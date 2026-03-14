from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List
import numpy as np

# Define slices here, interval (index) of N1
    POR_irn = slice()
    PCR_irn = slice()
    POR_hp_pt = slice()
    PCR_hp_pt = slice() 
    POR_hp_ctrl = slice() 
    PCR_hp_ctrl = slice()
    CR_rho_pm_pt = slice()
    CR_rho_pm_ctrl = slice() 
    CR_rho_mp_pt = slice() 
    CR_rho_mp_ctrl = slice()


@dataclass
class BootstrapTVectorResult:
    """Stores the results of a Studentized Bootstrap analysis."""
    theta_hat: np.ndarray       # [peak_latency_ms, peak_amplitude]
    se_hat: np.ndarray          # [standard_error_latency, standard_error_amplitude]
    ci_latency: Tuple[float, float]
    ci_amplitude: Tuple[float, float]


# ---------------------------------------------------------------------
# Core Statistical Functions
# ---------------------------------------------------------------------

def find_min_statistic(data_matrix: np.ndarray, 
                        time_points: np.ndarray, 
                        idx_slice: slice) -> np.ndarray:
    """Finds the peak (minimum) latency and amplitude within a time window."""
    mean_curve = data_matrix.mean(axis=1)
    curve_in_range = mean_curve[idx_slice]
    time_in_range = time_points[idx_slice]
    
    min_idx = np.argmin(curve_in_range)
    return np.array([time_in_range[min_idx], curve_in_range[min_idx]], dtype=float)


def bootstrap_se_vector(x: np.ndarray, 
                        b_se: int, 
                        stat_func: Callable, 
                        rng: np.random.Generator) -> np.ndarray:
    """Estimates standard error using bootstrap resampling."""
    thetas = np.empty((b_se, 2))
    n_subjects = x.shape[1]
    
    for k in range(b_se):
        idx = rng.integers(0, n_subjects, size=n_subjects)
        thetas[k] = stat_func(x[:, idx])
    return np.std(thetas, axis=0, ddof=1)


def bootstrap_t_interval_vector(x_matrix: np.ndarray, 
                                 time_points: np.ndarray, 
                                 idx_slice: slice, 
                                 alpha=0.05, 
                                 b_total=10000, 
                                 b_se=1000, 
                                 seed=None) -> BootstrapTVectorResult:
    """Calculates Studentized Bootstrap Confidence Intervals for Latency and Amplitude."""
    rng = np.random.default_rng(seed)
    
    def current_stat_func(d): 
        return find_min_statistic(d, time_points, idx_slice)
    
    # Point estimates
    theta_hat = current_stat_func(x_matrix)
    se_hat = bootstrap_se_vector(x_matrix, b_se, current_stat_func, rng)
    
    # T-statistic distribution
    t_stars = np.full((b_total, 2), np.nan)
    n_subjects = x_matrix.shape[1]
    
    for i in range(b_total):
        idx = rng.integers(0, n_subjects, size=n_subjects)
        x_resampled = x_matrix[:, idx]
        
        theta_star = current_stat_func(x_resampled)
        se_star = bootstrap_se_vector(x_resampled, b_se, current_stat_func, rng)
        
        if np.all(se_star > 0):
            t_stars[i, :] = (theta_star - theta_hat) / se_star
    
    # Filter valid resamples
    t_stars = t_stars[~np.isnan(t_stars).any(axis=1)]
    
    intervals = []
    for j in range(2):
        q_lo = np.quantile(t_stars[:, j], alpha / 2)
        q_hi = np.quantile(t_stars[:, j], 1 - alpha / 2)
        # CI formula: [theta - q_hi * se, theta - q_lo * se]
        intervals.append((theta_hat[j] - q_hi * se_hat[j], 
                          theta_hat[j] - q_lo * se_hat[j]))
                          
    return BootstrapTVectorResult(theta_hat, se_hat, intervals[0], intervals[1])


def get_individual_minima(matrix: np.ndarray, t_pts: np.ndarray, idx_slice: slice):
    """Extracts peak latency and amplitude for every subject individually."""
    sub_m, sub_t = matrix[idx_slice, :], t_pts[idx_slice]
    idx = np.argmin(sub_m, axis=0)
    return sub_t[idx], sub_m[idx, np.arange(matrix.shape[1])]


def calculate_hedges_g(group_pat, group_ctrl):
    """Calculates Hedges' g (positive value means Patients > Controls)."""
    n1, n2 = len(group_pat), len(group_ctrl)
    pooled_var = ((n1-1) * np.var(group_pat, ddof=1) + 
                  (n2-1) * np.var(group_ctrl, ddof=1)) / (n1 + n2 - 2)
    s_pooled = np.sqrt(pooled_var)
    
    if s_pooled == 0:
        return 0.0
        
    d = (np.mean(group_pat) - np.mean(group_ctrl)) / s_pooled
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return d * correction


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Constants
    T_UNIT, ZERO_INDEX = 1.0, 500
    OUTPUT_FILE = "bootstrap_t_results.txt"
    B_TOTAL, B_SE, ALPHA = 10000, 1000, 0.05

    # Organized Analysis Setup
    ANALYSIS_SETUP = {
        "huggins": {
            "K": 4000,
            "Patient": {"file": ".npy", "analyses": [
                {"name": "POR", "range": POR_hp_pt, "stim_start": 750},
                {"name": "PCR", "range": PCR_hp_pt, "stim_start": 1500}]},
            "Control": {"file": ".npy", "analyses": [
                {"name": "POR", "range": POR_hp_ctrl, "stim_start": 750},
                {"name": "PCR", "range": PCR_hp_ctrl, "stim_start": 1500}]}
        },
        "irn": {
            "K": 4000,
            "Patient": {"file": ".npy", "analyses": [
                {"name": "POR", "range": POR_irn, "stim_start": 750},
                {"name": "PCR", "range": PCR_irn, "stim_start": 1500}]},
            "Control": {"file": ".npy", "analyses": [
                {"name": "POR", "range": POR_irn, "stim_start": 750},
                {"name": "PCR", "range": PCR_irn, "stim_start": 1500}]}
        }
        # rhopm and rhomp follow the same pattern...
    }

    results_storage = {}

    with open(OUTPUT_FILE, 'w') as f:
        def log(msg): 
            print(msg)
            f.write(msg + '\n')

        for condition, setup in ANALYSIS_SETUP.items():
            log(f"\n{'='*70}\nCONDITION: {condition.upper()}\n{'='*70}")
            
            time_points = (np.arange(setup['K']) - ZERO_INDEX) * T_UNIT
            results_storage[condition] = {}
            matrices = {}

            # 1. Individual Group Analysis
            for group in ["Patient", "Control"]:
                try:
                    raw_data = np.load(setup[group]["file"])
                    # Ensure 2D (samples x subjects)
                    matrices[group] = raw_data.mean(axis=1) if raw_data.ndim == 3 else raw_data
                    results_storage[condition][group] = {}

                    for ana in setup[group]["analyses"]:
                        res = bootstrap_t_interval_vector(
                            matrices[group], time_points, ana["range"], ALPHA, B_TOTAL, B_SE
                        )
                        results_storage[condition][group][ana["name"]] = res
                        
                        stim_ms = ana["stim_start"]
                        rel_latency = res.theta_hat[0] - stim_ms
                        ci_rel_lat = (res.ci_latency[0] - stim_ms, res.ci_latency[1] - stim_ms)

                        log(f"\n--- {group} | {ana['name']} ---")
                        log(f"  Peak Location (ms): {res.theta_hat[0]:.1f}")
                        log(f"  Relative Latency:   {rel_latency:.1f} ms")
                        log(f"  Peak Amplitude:     {res.theta_hat[1]:.6f}")
                        log(f"  CI Latency (rel):  [{ci_rel_lat[0]:.1f}, {ci_rel_lat[1]:.1f}] ms")
                        log(f"  CI Amplitude:      [{res.ci_amplitude[0]:.6f}, {res.ci_amplitude[1]:.6f}]")
                
                except Exception as e:
                    log(f"  Error processing {group}: {e}")

            # 2. Group Comparisons
            log(f"\n--- GROUP COMPARISON: {condition.upper()} ---")
            for ana_name in [a["name"] for a in setup["Patient"]["analyses"]]:
                p_res = results_storage[condition]["Patient"].get(ana_name)
                c_res = results_storage[condition]["Control"].get(ana_name)
                
                if p_res and c_res:
                    # Significance via CI Overlap
                    sig_lat = not (max(p_res.ci_latency[0], c_res.ci_latency[0]) < 
                                   min(p_res.ci_latency[1], c_res.ci_latency[1]))
                    sig_amp = not (max(p_res.ci_amplitude[0], c_res.ci_amplitude[0]) < 
                                   min(p_res.ci_amplitude[1], c_res.ci_amplitude[1]))
                    
                    # Effect Size (Hedges' g)
                    sl_p = next(a["range"] for a in setup["Patient"]["analyses"] if a["name"] == ana_name)
                    sl_c = next(a["range"] for a in setup["Control"]["analyses"] if a["name"] == ana_name)
                    
                    ind_lat_p, ind_amp_p = get_individual_minima(matrices["Patient"], time_points, sl_p)
                    ind_lat_c, ind_amp_c = get_individual_minima(matrices["Control"], time_points, sl_c)
                    
                    g_lat = calculate_hedges_g(ind_lat_p, ind_lat_c)
                    # Flip amplitude g so positive means Patients are "stronger" (more negative)
                    g_amp = -calculate_hedges_g(ind_amp_p, ind_amp_c) 

                    log(f"  [{ana_name}]")
                    log(f"    Latency:   {'**SIGNIFICANT**' if sig_lat else 'Overlap (n.s.)'} | g = {g_lat:.3f}")
                    log(f"    Amplitude: {'**SIGNIFICANT**' if sig_amp else 'Overlap (n.s.)'} | g = {g_amp:.3f}")

    print(f"\nAnalysis complete. Results written to {OUTPUT_FILE}")
