"""
runner.py
----------
Thin wrapper around spicelib's SimRunner + AscEditor/SpiceEditor for
launching parameter-varied LTspice runs in parallel and reacting to each
result as soon as it's ready (rather than waiting for the whole batch),
which is what keeps the number of .raw/.log files sitting on disk small
even for large Monte Carlo batches.

spicelib is only imported inside this module's functions/methods (not at
module import time), so the rest of ltspice_mc -- parameter discovery,
distributions, metrics, visualization -- can be imported and unit-tested
even on a machine that doesn't have spicelib or LTspice installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .param_utils import is_asc_file

_logger = logging.getLogger("ltspice_mc.runner")

RunCallback = Callable[[Optional[str], Optional[str]], None]


class LTspiceRunner:
    """
    Manages one schematic/netlist and a pool of parallel LTspice runs.

    This class deliberately never touches component reference designators:
    every value it sets is a `.param` (via `editor.set_parameters(...)`),
    matching the "circuit-agnostic, .param-only" design constraint -- it has
    no idea whether your circuit has an R1 or an M17, and it doesn't need to.
    """

    def __init__(
        self,
        schematic_path: Union[str, Path],
        output_folder: Union[str, Path] = "./ltspice_mc_temp",
        parallel_sims: int = 4,
        ltspice_exe: Optional[str] = None,
        sim_timeout: float = 600.0,
        extra_instructions: Optional[List[str]] = None,
    ):
        try:
            from spicelib import SimRunner, AscEditor, SpiceEditor
            from spicelib.simulators.ltspice_simulator import LTspice
        except ImportError as exc:
            raise ImportError(
                "spicelib is required to actually run simulations. Install it with:\n"
                "    pip install spicelib\n"
                "(Parameter discovery, distributions, and metric math all work fine "
                "without spicelib -- only launching LTspice itself needs it.)"
            ) from exc

        self.schematic_path = Path(schematic_path)
        if not self.schematic_path.exists():
            raise FileNotFoundError(f"Schematic/netlist not found: {self.schematic_path}")

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        if ltspice_exe:
            try:
                LTspice.spice_exe = [str(ltspice_exe)]
            except Exception:
                _logger.warning(
                    "Could not override the LTspice executable path via the "
                    "mechanism this version of spicelib exposes; falling back "
                    "to spicelib's auto-detected path. Check spicelib's docs "
                    "for the exact override attribute/method in your installed "
                    "version if auto-detection doesn't find your install.",
                    exc_info=True,
                )

        self.editor = (
            AscEditor(str(self.schematic_path))
            if is_asc_file(self.schematic_path)
            else SpiceEditor(str(self.schematic_path))
        )

        if extra_instructions:
            for instruction in extra_instructions:
                self.editor.add_instruction(instruction)

        self.runner = SimRunner(
            simulator=LTspice,
            parallel_sims=parallel_sims,
            timeout=sim_timeout,
            output_folder=str(output_folder),
        )

    def run_case(self, param_values: Dict[str, float], callback: RunCallback) -> None:
        """
        Set `.param` values for one Monte Carlo sample and queue it for
        (possibly parallel) execution. `callback(raw_path, log_path)` is
        invoked once this specific run finishes; both arguments are None if
        the run aborted before producing files.

        NOTE: this must be called from a single (e.g. the main) thread, in a
        simple loop -- that's the documented spicelib usage pattern for
        parallel batches: mutate the editor, call run(), repeat. SimRunner
        snapshots the netlist into its own run-numbered file on each call,
        so looping here is safe even though multiple LTspice instances end
        up executing concurrently in the background.
        """

        self.editor.set_parameters(**param_values)

        def _wrapped_callback(raw_file, log_file):
            try:
                callback(str(raw_file) if raw_file else None, str(log_file) if log_file else None)
            except Exception:
                _logger.exception("Unhandled exception inside user callback")

        self.runner.run(self.editor, callback=_wrapped_callback, callback_on_error=True)

    def wait_all(self, timeout: Optional[float] = None) -> bool:
        """Block until every queued run has finished. Returns True iff every
        run completed successfully (mirrors spicelib's own return value)."""
        return bool(self.runner.wait_completion(timeout=timeout))

    def run_case_blocking(self, param_values: Dict[str, float]) -> "tuple[Optional[str], Optional[str]]":
        """
        Runs exactly one simulation and blocks until it's done, returning
        (raw_path, log_path) (either may be None if the run aborted). Meant
        for sequential callers like an optimizer, which need one result
        before deciding the next point -- don't mix with concurrent
        run_case() calls on the same instance.
        """
        result: Dict[str, Optional[str]] = {"raw": None, "log": None}

        def _callback(raw_path, log_path):
            result["raw"], result["log"] = raw_path, log_path

        self.run_case(param_values, callback=_callback)
        self.wait_all()
        return result["raw"], result["log"]

    @property
    def n_ok(self) -> int:
        return self.runner.okSim

    @property
    def n_total(self) -> int:
        return self.runner.runno

    def cleanup(self) -> None:
        """Delete any run/log files spicelib is still tracking (best-effort)."""
        try:
            self.runner.cleanup_files()
        except Exception:
            _logger.debug("cleanup_files() failed or is unavailable in this spicelib version", exc_info=True)
