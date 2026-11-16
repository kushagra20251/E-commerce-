

import os #part1
import numpy as np

def run_part1():
    os.makedirs("outputs", exist_ok=True)

    # Load data
    data = np.genfromtxt("data/records.csv", delimiter=",", skip_header=1)
    # Columns: record_id(0), c1(1), c2(2), c3(3), c4(4), c5(5), b1(6), b2(7), outcome(8), value(9)
    value = data[:, 9]
    X4 = data[:, 4]
    n = len(value)

    # Task 1.1
    mean_value = np.mean(value)
    sd_value = np.std(value, ddof=1)
    se_value = sd_value / np.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    # Task 1.2
    mu_hyp = 8.0
    z_stat = (mean_value - mu_hyp) / se_value
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    # Task 1.3
    E = 0.5
    n_required = int(np.ceil((1.96 * sd_value / E) ** 2))

    # Task 1.4
    cov_val = np.cov(X4, value, ddof=1)[0, 1]
    var_X4 = np.var(X4, ddof=1)
    reg_slope = cov_val / var_X4
    reg_intercept = np.mean(value) - reg_slope * np.mean(X4)
    fitted = reg_slope * X4 + reg_intercept

    tss = np.sum((value - mean_value) ** 2)
    ess = np.sum((fitted - mean_value) ** 2)
    rss = np.sum((value - fitted) ** 2)
    tss_equals_ess_plus_rss = "yes" if abs(tss - (ess + rss)) < 1e-6 * tss else "no"
    r2 = 1.0 - (rss / tss)
    rmse = np.sqrt(rss / n)
    mae = np.mean(np.abs(value - fitted))

    # Task 1.5
    mean_X4 = np.mean(X4)
    sd_X4 = np.std(X4, ddof=1)
    z = (X4 - mean_X4) / sd_X4

    m, c = 0.0, 0.0
    def compute_cost(m_val, c_val):
        preds = m_val * z + c_val
        return np.mean((value - preds) ** 2)

    gd_J0 = compute_cost(m, c)
    eta = 0.1
    costs = {}
    for epoch in range(1, 501):
        preds = m * z + c
        e = value - preds
        dm = - (2.0 / n) * np.sum(z * e)
        dc = - (2.0 / n) * np.sum(e)
        m -= eta * dm
        c -= eta * dc
        if epoch == 1: costs[1] = compute_cost(m, c)
        if epoch == 10: costs[10] = compute_cost(m, c)
        if epoch == 100: costs[100] = compute_cost(m, c)
        if epoch == 500: costs[500] = compute_cost(m, c)

    gd_slope = m / sd_X4
    gd_intercept = c - m * mean_X4 / sd_X4
    gd_slope_gap = abs(gd_slope - reg_slope)

    # Task 1.6
    sort_idx = np.argsort(fitted, kind='stable')
    sorted_value = value[sort_idx]
    sorted_fitted = fitted[sort_idx]
    sorted_resids = sorted_value - sorted_fitted

    lower_resids = sorted_resids[:400]
    upper_resids = sorted_resids[400:]
    resid_sd_lower_raw = np.std(lower_resids, ddof=1)
    resid_sd_upper_raw = np.std(upper_resids, ddof=1)
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    log_val = np.log(value)
    log_X4 = np.log(X4)
    log_cov = np.cov(log_X4, log_val, ddof=1)[0, 1]
    log_var = np.var(log_X4, ddof=1)
    loglog_slope = log_cov / log_var
    loglog_intercept = np.mean(log_val) - loglog_slope * np.mean(log_X4)
    log_fitted = loglog_slope * log_X4 + loglog_intercept
    log_rss = np.sum((log_val - log_fitted) ** 2)
    log_tss = np.sum((log_val - np.mean(log_val)) ** 2)
    loglog_r2 = 1.0 - (log_rss / log_tss)
    loglog_rmse = np.sqrt(log_rss / n)

    log_sort_idx = np.argsort(log_fitted, kind='stable')
    log_sorted_resids = (log_val - log_fitted)[log_sort_idx]
    resid_sd_lower_log = np.std(log_sorted_resids[:400], ddof=1)
    resid_sd_upper_log = np.std(log_sorted_resids[400:], ddof=1)
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

    # Task 1.7
    ones = np.ones(n)
    X_mat = np.column_stack((ones, log_X4, data[:, 1], data[:, 2], data[:, 3], data[:, 5], data[:, 6], data[:, 7]))
    beta = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ log_val)

    mr_preds = X_mat @ beta
    mr_rss = np.sum((log_val - mr_preds) ** 2)
    mr_r2 = 1.0 - (mr_rss / log_tss)
    p = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p - 1)
    loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1)

    # Write results
    results = [
        f"n = {n}",
        f"mean_value = {mean_value:.4f}",
        f"sd_value = {sd_value:.4f}",
        f"se_value = {se_value:.4f}",
        f"ci95_low = {ci95_low:.4f}",
        f"ci95_high = {ci95_high:.4f}",
        f"z_stat = {z_stat:.4f}",
        f"reject_h0 = {reject_h0}",
        f"n_required = {n_required}",
        f"reg_slope = {reg_slope:.4f}",
        f"reg_intercept = {reg_intercept:.4f}",
        f"tss = {tss:.4f}",
        f"ess = {ess:.4f}",
        f"rss = {rss:.4f}",
        f"tss_equals_ess_plus_rss = {tss_equals_ess_plus_rss}",
        f"r2 = {r2:.4f}",
        f"rmse = {rmse:.4f}",
        f"mae = {mae:.4f}",
        f"gd_J0 = {gd_J0:.4f}",
        f"gd_J1 = {costs[1]:.4f}",
        f"gd_J10 = {costs[10]:.4f}",
        f"gd_J100 = {costs[100]:.4f}",
        f"gd_J500 = {costs[500]:.4f}",
        f"gd_slope = {gd_slope:.4f}",
        f"gd_intercept = {gd_intercept:.4f}",
        f"gd_slope_gap = {gd_slope_gap:.6f}",
        f"resid_sd_lower_raw = {resid_sd_lower_raw:.4f}",
        f"resid_sd_upper_raw = {resid_sd_upper_raw:.4f}",
        f"resid_sd_ratio_raw = {resid_sd_ratio_raw:.4f}",
        f"loglog_intercept = {loglog_intercept:.4f}",
        f"loglog_slope = {loglog_slope:.4f}",
        f"loglog_r2 = {loglog_r2:.4f}",
        f"loglog_rmse = {loglog_rmse:.4f}",
        f"resid_sd_lower_log = {resid_sd_lower_log:.4f}",
        f"resid_sd_upper_log = {resid_sd_upper_log:.4f}",
        f"resid_sd_ratio_log = {resid_sd_ratio_log:.4f}",
        f"mr_beta_0 = {beta[0]:.4f}",
        f"mr_beta_1 = {beta[1]:.4f}",
        f"mr_beta_2 = {beta[2]:.4f}",
        f"mr_beta_3 = {beta[3]:.4f}",
        f"mr_beta_4 = {beta[4]:.4f}",
        f"mr_beta_5 = {beta[5]:.4f}",
        f"mr_beta_6 = {beta[6]:.4f}",
        f"mr_beta_7 = {beta[7]:.4f}",
        f"mr_r2 = {mr_r2:.4f}",
        f"mr_adj_r2 = {mr_adj_r2:.4f}",
        f"loglog_adj_r2 = {loglog_adj_r2:.4f}"
    ]

    with open("outputs/part1_results.txt", "w") as f:
        f.write("\n".join(results) + "\n")

if __name__ == "__main__":
    run_part1()

