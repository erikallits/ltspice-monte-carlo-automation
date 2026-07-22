from pathlib import Path
import time 
import matplotlib.pyplot as plt
import numpy as np

from ltspice_mc.engine import MonteCarloEngine
from ltspice_mc.metrics import compute_bandwidth
from ltspice_mc.netlist_parameterizer import interactive_parameterize
from ltspice_mc.visualization import plot_summary_dashboard

# 1. Ορίζω το αρχικό αρχείο
original_circuit = Path("circuit.asc")

if not original_circuit.exists():
    print(f"\n[Σφάλμα] Το αρχείο '{original_circuit}' δεν βρέθηκε!")
    exit(1)

# 2. Καλώ την interactive_parameterize
print("--- Αυτόματη Ανακάλυψη Παραμέτρων από το Netlist ---")
result = interactive_parameterize(original_circuit)

if not result.distributions:
    print("\n[Σφάλμα] Δεν επιλέχθηκε κανένα εξάρτημα. Τερματισμός.")
    exit(1)

print("---------------------------------------------------\n")

# 3. Δημιουργία του Engine πάνω στο νέο αρχείο
engine = MonteCarloEngine(result.output_path)

# 4. Περνάμε τις κατανομές
for param_name, dist in result.distributions.items():
    engine.add_parameter(param_name, dist)

# 5. Ορισμός Μέτρησης
engine.add_metric("my_bandwidth", lambda raw: compute_bandwidth(raw, "V(vout)", "V(vin)"))

# 6. Προσομοίωση
start_time = time.time()
print("Οι προσομοιώσεις ξεκίνησαν στο LTspice...")

results = engine.run(n_samples=50, seed=42)

total_duration = time.time() - start_time
print(f"\nΌλες οι προσομοιώσεις ολοκληρώθηκαν σε {total_duration:.2f} δευτερόλεπτα.")

# 7. Αποτελέσματα
print("\n=== ΣΤΑΤΙΣΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ===")
print(results.summary())
print("===============================\n")

# ------------------------------------------------------------------------------
# 8. ΔΙΑΔΡΑΣΤΙΚΗ ΕΙΣΑΓΩΓΗ ΟΡΙΩΝ YIELD ΑΠΟ ΤΟΝ ΧΡΗΣΤΗ
# ------------------------------------------------------------------------------
print("=== ΟΡΙΣΜΟΣ ΠΡΟΔΙΑΓΡΑΦΩΝ (SPECS) ΓΙΑ YIELD ===")
print("(Πιέστε Enter χωρίς τιμή αν δεν θέλετε να ορίσετε κάποιο όριο)\n")

lower_str = input("Δώστε κατώτερο όριο Bandwidth σε Hz (Low Limit) [Enter για κανένα]: ").strip()
upper_str = input("Δώστε ανώτερο όριο Bandwidth σε Hz (High Limit) [Enter για κανένα]: ").strip()

spec_limits = {}
bandwidth_limits = {}

if lower_str:
    try:
        bandwidth_limits["lower"] = float(lower_str)
    except ValueError:
        print("[Προειδοποίηση]: Μη έγκυρη τιμή για το κατώτερο όριο. Αγνοείται.")

if upper_str:
    try:
        bandwidth_limits["upper"] = float(upper_str)
    except ValueError:
        print("[Προειδοποίηση]: Μη έγκυρη τιμή για το ανώτερο όριο. Αγνοείται.")

if bandwidth_limits:
    spec_limits["my_bandwidth"] = bandwidth_limits

    bandwidth_values = [v for v in results.metrics["my_bandwidth"] if v is not None]
    
    success_count = 0
    for val in bandwidth_values:
        is_valid = True
        if "lower" in bandwidth_limits and val < bandwidth_limits["lower"]:
            is_valid = False
        if "upper" in bandwidth_limits and val > bandwidth_limits["upper"]:
            is_valid = False
        if is_valid:
            success_count += 1

    yield_percentage = (success_count / len(bandwidth_values)) * 100
    print(f"\n---> YIELD RESULT: {success_count}/{len(bandwidth_values)} εντός ορίων ({yield_percentage:.2f}%)\n")
else:
    print("\nΔεν ορίστηκαν όρια. Παράλειψη υπολογισμού Yield.\n")

# ------------------------------------------------------------------------------
# 9. ΒΑΣΙΚΟ DASHBOARD
# ------------------------------------------------------------------------------
if spec_limits:
    fig1 = plot_summary_dashboard(results, metrics=["my_bandwidth"], limits=spec_limits)
else:
    fig1 = plot_summary_dashboard(results, metrics=["my_bandwidth"])

plt.savefig("my_circuit_monte_carlo.png", dpi=150)
print("1. Το βασικό γράφημα αποθηκεύτηκε ως 'my_circuit_monte_carlo.png'")

# ------------------------------------------------------------------------------
# 10. ΠΡΟΣΘΕΤΑ ΧΡΗΣΙΜΑ ΔΙΑΓΡΑΜΜΑΤΑ (Βελτιστοποιημένα για 14" Οθόνες - Grid 2x3)
# ------------------------------------------------------------------------------
bw_vals = np.array([v for v in results.metrics["my_bandwidth"] if v is not None])
param_names = list(results.parameters.keys())
n_params = len(param_names)

if n_params > 0 and len(bw_vals) > 0:
    # Υπολογισμός γραμμών/στηλών για πλέγμα (Grid Layout 2x3)
    total_plots = n_params + 1  # 4 params + 1 CDF = 5 plots
    ncols = 3
    nrows = 2

    # Δημιουργία παραθύρου με αναλογίες κατάλληλες για 14" οθόνη
    fig2, axes = plt.subplots(nrows, ncols, figsize=(11, 7))
    fig2.suptitle("Ανάλυση Ευαισθησίας (Sensitivity) & Αθροιστική Κατανομή (CDF)", fontsize=12, fontweight='bold')
    
    # Μετατροπή του πίνακα axes σε 1D array για εύκολη προσπέλαση
    axes_flat = axes.flatten()

    # Scatter Plots για κάθε εξάρτημα
    for i, p_name in enumerate(param_names):
        p_vals = np.array(results.parameters[p_name])
        ax = axes_flat[i]
        ax.scatter(p_vals, bw_vals, alpha=0.75, c='#1f77b4', edgecolors='none', s=20)
        ax.set_title(f"Ευαισθησία: {p_name}", fontsize=10)
        ax.set_xlabel(f"Τιμή {p_name}", fontsize=8)
        ax.set_ylabel("Bandwidth (Hz)", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    # CDF Plot (στο 5ο κουτί του πλέγματος)
    ax_cdf = axes_flat[n_params]
    sorted_bw = np.sort(bw_vals)
    cdf_vals = np.arange(1, len(sorted_bw) + 1) / len(sorted_bw)
    ax_cdf.plot(sorted_bw, cdf_vals, marker='.', linestyle='none', color='darkred', ms=4)
    ax_cdf.set_title("Αθροιστική Κατανομή (CDF)", fontsize=10)
    ax_cdf.set_xlabel("Bandwidth (Hz)", fontsize=8)
    ax_cdf.set_ylabel("Πιθανότητα (0 - 1)", fontsize=8)
    ax_cdf.tick_params(axis='both', which='major', labelsize=8)
    ax_cdf.grid(True, linestyle="--", alpha=0.4)

    # Προσθήκη ορίων Yield στο CDF αν υπάρχουν
    if "lower" in bandwidth_limits:
        ax_cdf.axvline(bandwidth_limits["lower"], color='red', linestyle='--', label='Low')
    if "upper" in bandwidth_limits:
        ax_cdf.axvline(bandwidth_limits["upper"], color='red', linestyle='--', label='High')
    if bandwidth_limits:
        ax_cdf.legend(fontsize=7)

    # Κρύβουμε το 6ο άδειο κουτί που περισσεύει στο 2x3 πλέγμα
    for j in range(total_plots, len(axes_flat)):
        fig2.delaxes(axes_flat[j])

    # Αυτόματη στοίχιση με απόσταση για να μην κρύβονται τα labels
    plt.tight_layout()
    plt.savefig("my_circuit_sensitivity_analysis.png", dpi=120)
    print("2. Το γράφημα ευαισθησίας & CDF αποθηκεύτηκε ως 'my_circuit_sensitivity_analysis.png'")

plt.show()