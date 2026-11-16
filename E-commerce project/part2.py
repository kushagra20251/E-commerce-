

import os#part2
import numpy as np

def run_part2():
    data = np.genfromtxt("data/records.csv", delimiter=",", skip_header=1)
    outcomes = data[:, 8].astype(int)
    features = data[:, 1:8] # X1 to X7

    # Task 2.1
    idx_0 = np.where(outcomes == 0)[0]
    idx_1 = np.where(outcomes == 1)[0]

    train_0_count = int(np.floor(7 * len(idx_0) / 10))
    train_1_count = int(np.floor(7 * len(idx_1) / 10))

    train_idx = np.concatenate([idx_0[:train_0_count], idx_1[:train_1_count]])
    test_idx = np.concatenate([idx_0[train_0_count:], idx_1[train_1_count:]])

    X_train, y_train = features[train_idx], outcomes[train_idx]
    X_test, y_test = features[test_idx], outcomes[test_idx]

    train_rows = len(X_train)
    train_positives = int(np.sum(y_train))
    test_rows = len(X_test)
    test_positives = int(np.sum(y_test))

    train_mean = np.mean(X_train, axis=0)
    train_sd = np.std(X_train, axis=0, ddof=1)

    X_train_std = (X_train - train_mean) / train_sd
    X_test_std = (X_test - train_mean) / train_sd

    # Task 2.2
    n_tr = len(X_train_std)
    weights = np.zeros(7)
    bias = 0.0

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def bce_loss(y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    z_tr = X_train_std @ weights + bias
    p_tr = sigmoid(z_tr)
    loss_epoch_0 = bce_loss(y_train, p_tr)

    losses = {}
    eta = 0.1
    for epoch in range(1, 1001):
        z = X_train_std @ weights + bias
        preds = sigmoid(z)
        error = preds - y_train
        dw = (1.0 / n_tr) * (X_train_std.T @ error)
        db = (1.0 / n_tr) * np.sum(error)
        weights -= eta * dw
        bias -= eta * db

        if epoch == 1: losses[1] = bce_loss(y_train, preds)
        if epoch == 10: losses[10] = bce_loss(y_train, preds)
        if epoch == 100: losses[100] = bce_loss(y_train, preds)
        if epoch == 1000: losses[1000] = bce_loss(y_train, preds)

    test_preds = sigmoid(X_test_std @ weights + bias)
    test_bce = bce_loss(y_test, test_preds)

    # Task 2.3
    def run_lr_eta(test_eta):
        w = np.zeros(7)
        b = 0.0
        for _ in range(200):
            preds = sigmoid(X_train_std @ w + b)
            error = preds - y_train
            w -= test_eta * (1.0 / n_tr) * (X_train_std.T @ error)
            b -= test_eta * (1.0 / n_tr) * np.sum(error)
        return bce_loss(y_train, sigmoid(X_train_std @ w + b))

    loss_epoch_200_eta_001 = run_lr_eta(0.01)
    loss_epoch_200_eta_01 = run_lr_eta(0.1)
    loss_epoch_200_eta_10 = run_lr_eta(1.0)

    # Task 2.4
    odds_ratios = np.exp(weights)

    # Task 2.5 & Metrics helper
    def get_confusion_cells(y_true, p_vals, tau):
        preds = (p_vals >= tau).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        return tp, fp, tn, fn

    def calc_metrics(tp, fp, tn, fn):
        acc = (tp + tn) / max(1, (tp + fp + tn + fn))
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, (tp + fn))
        f1 = (2 * prec * rec) / max(1e-15, (prec + rec)) if (prec + rec) > 0 else 0.0
        spec = tn / max(1, (tn + fp))
        return acc, prec, rec, f1, spec

    test_probs = test_preds
    tp_5, fp_5, tn_5, fn_5 = get_confusion_cells(y_test, test_probs, 0.5)
    acc_5, prec_5, rec_5, f1_5, spec_5 = calc_metrics(tp_5, fp_5, tn_5, fn_5)

    tp_3, fp_3, tn_3, fn_3 = get_confusion_cells(y_test, test_probs, 0.3)
    acc_3, prec_3, rec_3, f1_3, spec_3 = calc_metrics(tp_3, fp_3, tn_3, fn_3)

    blind_acc = float(np.sum(y_test == 0)) / len(y_test)

    # Task 2.6
    thresholds = np.round(np.arange(0.1, 1.0, 0.1), 1)
    thresh_rows = []
    for tau in thresholds:
        tp, fp, tn, fn = get_confusion_cells(y_test, test_probs, tau)
        tpr = tp / max(1, (tp + fn))
        fpr = fp / max(1, (fp + tn))
        thresh_rows.append(f"{tau:.1f},{tp},{fp},{tn},{fn},{tpr:.4f},{fpr:.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/part2_thresholds.csv", "w") as f:
        f.write("tau,tp,fp,tn,fn,tpr,fpr\n" + "\n".join(thresh_rows) + "\n")

    # Task 2.7 (KNN)
    def knn_predict(X_ref, y_ref, x_query, K):
        dists = np.sqrt(np.sum((X_ref - x_query) ** 2, axis=1))
        # Stable sort: primary distance, secondary index
        sort_order = np.lexsort((np.arange(len(dists)), dists))
        sorted_labels = y_ref[sort_order[:K]]
        pos_count = np.sum(sorted_labels == 1)
        needed = (K + 1) / 2
        return 1 if pos_count >= needed else 0, float(pos_count) / K

    k_vals = list(range(1, 22, 2))
    knn_cv_rows = []
    best_k_f1 = -1
    knn_best_k = 1

    folds = np.arange(n_tr) % 5
    for K in k_vals:
        fold_f1s = []
        for fold in range(5):
            val_mask = (folds == fold)
            train_mask = ~val_mask
            X_fold_tr, y_fold_tr = X_train_std[train_mask], y_train[train_mask]
            X_fold_val, y_fold_val = X_train_std[val_mask], y_train[val_mask]

            preds = []
            for query in X_fold_val:
                dists = np.sqrt(np.sum((X_fold_tr - query) ** 2, axis=1))
                sort_order = np.lexsort((np.arange(len(dists)), dists))
                neighbors = y_fold_tr[sort_order[:K]]
                pred = 1 if np.sum(neighbors == 1) >= ((K + 1) / 2) else 0
                preds.append(pred)
            preds = np.array(preds)

            tp = int(np.sum((preds == 1) & (y_fold_val == 1)))
            fp = int(np.sum((preds == 1) & (y_fold_val == 0)))
            fn = int(np.sum((preds == 0) & (y_fold_val == 1)))
            prec = tp / max(1, (tp + fp))
            rec = tp / max(1, (tp + fn))
            f1 = (2 * prec * rec) / max(1e-15, (prec + rec)) if (prec + rec) > 0 else 0.0
            fold_f1s.append(f1)

        mean_f1 = np.mean(fold_f1s)
        knn_cv_rows.append(f"{K},{mean_f1:.4f}")
        if mean_f1 > best_k_f1:
            best_k_f1 = mean_f1
            knn_best_k = K

    with open("outputs/part2_knn_cv.csv", "w") as f:
        f.write("K,mean_cv_f1\n" + "\n".join(knn_cv_rows) + "\n")

    # Refit KNN on all training rows
    knn_test_preds = []
    for query in X_test_std:
        dists = np.sqrt(np.sum((X_train_std - query) ** 2, axis=1))
        sort_order = np.lexsort((np.arange(len(dists)), dists))
        neighbors = y_train[sort_order[:knn_best_k]]
        pred = 1 if np.sum(neighbors == 1) >= ((knn_best_k + 1) / 2) else 0
        knn_test_preds.append(pred)
    knn_test_preds = np.array(knn_test_preds)

    knn_tp = int(np.sum((knn_test_preds == 1) & (y_test == 1)))
    knn_fp = int(np.sum((knn_test_preds == 1) & (y_test == 0)))
    knn_tn = int(np.sum((knn_test_preds == 0) & (y_test == 0)))
    knn_fn = int(np.sum((knn_test_preds == 0) & (y_test == 1)))
    knn_acc, knn_prec, knn_rec, knn_f1, _ = calc_metrics(knn_tp, knn_fp, knn_tn, knn_fn)

    # Task 2.8 (Money optimization)
    train_probs = sigmoid(X_train_std @ weights + bias)
    taus = np.arange(0.05, 1.0, 0.05)
    best_tau = 0.05
    min_train_cost = float('inf')

    for tau in taus:
        tp, fp, tn, fn = get_confusion_cells(y_train, train_probs, tau)
        cost = 50 * fp + 400 * fn
        if cost < min_train_cost:
            min_train_cost = cost
            best_tau = round(float(tau), 2)

    cost_tau = best_tau
    cost_train_at_tau = min_train_cost

    tp_t, fp_t, tn_t, fn_t = get_confusion_cells(y_test, test_probs, cost_tau)
    cost_test_at_tau = 50 * fp_t + 400 * fn_t

    tp_5s, fp_5s, tn_5s, fn_5s = get_confusion_cells(y_test, test_probs, 0.5)
    cost_test_at_05 = 50 * fp_5s + 400 * fn_5s

    # Flag nobody
    fn_nobody = int(np.sum(y_test == 1))
    cost_test_flag_nobody = 400 * fn_nobody

    results = [
        f"train_rows = {train_rows}",
        f"train_positives = {train_positives}",
        f"test_rows = {test_rows}",
        f"test_positives = {test_positives}",
        f"loss_epoch_0 = {loss_epoch_0:.4f}",
        f"loss_epoch_1 = {losses[1]:.4f}",
        f"loss_epoch_10 = {losses[10]:.4f}",
        f"loss_epoch_100 = {losses[100]:.4f}",
        f"loss_epoch_1000 = {losses[1000]:.4f}",
        f"weight_X1 = {weights[0]:.4f}",
        f"weight_X2 = {weights[1]:.4f}",
        f"weight_X3 = {weights[2]:.4f}",
        f"weight_X4 = {weights[3]:.4f}",
        f"weight_X5 = {weights[4]:.4f}",
        f"weight_X6 = {weights[5]:.4f}",
        f"weight_X7 = {weights[6]:.4f}",
        f"bias = {bias:.4f}",
        f"test_bce = {test_bce:.4f}",
        f"loss_epoch_200_eta_0.01 = {loss_epoch_200_eta_001:.4f}",
        f"loss_epoch_200_eta_0.1 = {loss_epoch_200_eta_01:.4f}",
        f"loss_epoch_200_eta_1.0 = {loss_epoch_200_eta_10:.4f}",
        f"odds_ratio_X1 = {odds_ratios[0]:.4f}",
        f"odds_ratio_X2 = {odds_ratios[1]:.4f}",
        f"odds_ratio_X3 = {odds_ratios[2]:.4f}",
        f"odds_ratio_X4 = {odds_ratios[3]:.4f}",
        f"odds_ratio_X5 = {odds_ratios[4]:.4f}",
        f"odds_ratio_X6 = {odds_ratios[5]:.4f}",
        f"odds_ratio_X7 = {odds_ratios[6]:.4f}",
        f"tau_0.5_tp = {tp_5}",
        f"tau_0.5_fp = {fp_5}",
        f"tau_0.5_tn = {tn_5}",
        f"tau_0.5_fn = {fn_5}",
        f"tau_0.5_accuracy = {acc_5:.4f}",
        f"tau_0.5_precision = {prec_5:.4f}",
        f"tau_0.5_recall = {rec_5:.4f}",
        f"tau_0.5_f1 = {f1_5:.4f}",
        f"tau_0.5_specificity = {spec_5:.4f}",
        f"tau_0.3_tp = {tp_3}",
        f"tau_0.3_fp = {fp_3}",
        f"tau_0.3_tn = {tn_3}",
        f"tau_0.3_fn = {fn_3}",
        f"tau_0.3_accuracy = {acc_3:.4f}",
        f"tau_0.3_precision = {prec_3:.4f}",
        f"tau_0.3_recall = {rec_3:.4f}",
        f"tau_0.3_f1 = {f1_3:.4f}",
        f"tau_0.3_specificity = {spec_3:.4f}",
        f"blind_accuracy = {blind_acc:.4f}",
        f"knn_best_k = {knn_best_k}",
        f"knn_tp = {knn_tp}",
        f"knn_fp = {knn_fp}",
        f"knn_tn = {knn_tn}",
        f"knn_fn = {knn_fn}",
        f"knn_accuracy = {knn_acc:.4f}",
        f"knn_precision = {knn_prec:.4f}",
        f"knn_recall = {knn_rec:.4f}",
        f"knn_f1 = {knn_f1:.4f}",
        f"cost_tau = {cost_tau:.2f}",
        f"cost_train_at_tau = {cost_train_at_tau}",
        f"cost_test_at_tau = {cost_test_at_tau}",
        f"cost_test_at_0.5 = {cost_test_at_05}",
        f"cost_test_flag_nobody = {cost_test_flag_nobody}"
    ]

    with open("outputs/part2_results.txt", "w") as f:
        f.write("\n".join(results) + "\n")

if __name__ == "__main__":
    run_part2()

