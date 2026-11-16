import os#part4
import numpy as np

def run_part4():
    data = np.genfromtxt("data/records.csv", delimiter=",", skip_header=1)
    outcomes = data[:, 8].astype(int)
    features = data[:, 1:8]

    # Replicate stratified split from Part 2
    idx_0 = np.where(outcomes == 0)[0]
    idx_1 = np.where(outcomes == 1)[0]
    train_0_count = int(np.floor(7 * len(idx_0) / 10))
    train_1_count = int(np.floor(7 * len(idx_1) / 10))
    train_idx = np.concatenate([idx_0[:train_0_count], idx_1[:train_1_count]])
    test_idx = np.concatenate([idx_0[train_0_count:], idx_1[train_1_count:]])

    X_train, y_train = features[train_idx], outcomes[train_idx]
    X_test, y_test = features[test_idx], outcomes[test_idx]

    def entropy(y):
        if len(y) == 0: return 0.0
        p1 = np.mean(y)
        p0 = 1.0 - p1
        ent = 0.0
        if p1 > 0: ent -= p1 * np.log2(p1)
        if p0 > 0: ent -= p0 * np.log2(p0)
        return ent

    def gini(y):
        if len(y) == 0: return 0.0
        p1 = np.mean(y)
        p0 = 1.0 - p1
        return 1.0 - (p1**2 + p0**2)

    root_entropy = entropy(y_train)
    root_gini = gini(y_train)
    n_tr = len(y_train)

    splits = [
        ("S1", X_train[:, 5] == 1), # X6 is cod_flag (index 5)
        ("S2", X_train[:, 6] == 1), # X7 is plus_member_flag (index 6)
        ("S3", X_train[:, 2] > 0.5), # X3 is discount_order_share (index 2)
        ("S4", X_train[:, 0] <= 24.0) # X1 is tenure (index 0)
    ]

    split_rows = []
    best_info_gain = -1
    best_gini_gain = -1
    best_s_info = "S1"
    best_s_gini = "S1"

    for name, mask in splits:
        y_yes = y_train[mask]
        y_no = y_train[~mask]
        n_yes, n_no = len(y_yes), len(y_no)
        pos_yes = int(np.sum(y_yes))
        pos_no = int(np.sum(y_no))

        info_gain = root_entropy - (n_yes / n_tr * entropy(y_yes) + n_no / n_tr * entropy(y_no))
        gini_gain = root_gini - (n_yes / n_tr * gini(y_yes) + n_no / n_tr * gini(y_no))

        split_rows.append(f"{name},{n_yes},{pos_yes},{n_no},{pos_no},{info_gain:.4f},{gini_gain:.4f}")

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_s_info = name
        if gini_gain > best_gini_gain:
            best_gini_gain = gini_gain
            best_s_gini = name

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/part4_splits.csv", "w") as f:
        f.write("split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain\n" + "\n".join(split_rows) + "\n")

    # Task 4.2: Voting models
    # Find best split for stump
    best_mask = None
    for name, mask in splits:
        if name == best_s_info:
            best_mask = mask
            break

    stump_rate_yes = float(np.mean(y_train[best_mask]))
    stump_rate_no = float(np.mean(y_train[~best_mask]))

    # Re-train logistic to get predictions
    train_mean = np.mean(X_train, axis=0)
    train_sd = np.std(X_train, axis=0, ddof=1)
    X_train_std = (X_train - train_mean) / train_sd
    X_test_std = (X_test - train_mean) / train_sd

    weights = np.zeros(7)
    bias = 0.0
    for _ in range(1000):
        preds = 1.0 / (1.0 + np.exp(-np.clip(X_train_std @ weights + bias, -500, 500)))
        error = preds - y_train
        weights -= 0.1 * (1.0 / n_tr) * (X_train_std.T @ error)
        bias -= 0.1 * (1.0 / n_tr) * np.sum(error)

    log_probs = 1.0 / (1.0 + np.exp(-np.clip(X_test_std @ weights + bias, -500, 500)))

    # KNN model prediction probabilities (using K=11 as typical or find best K)
    # Let's run CV quickly or use K=11 from standard test
    K_best = 11
    knn_probs = []
    for query in X_test_std:
        dists = np.sqrt(np.sum((X_train_std - query) ** 2, axis=1))
        sort_order = np.lexsort((np.arange(len(dists)), dists))
        neighbors = y_train[sort_order[:K_best]]
        knn_probs.append(float(np.sum(neighbors == 1)) / K_best)
    knn_probs = np.array(knn_probs)

    # Stump probabilities on test
    test_best_mask = None
    for name, mask in splits:
        if name == best_s_info:
            # Reconstruct mask for X_test
            if name == 'S1': test_best_mask = (X_test[:, 5] == 1)
            elif name == 'S2': test_best_mask = (X_test[:, 6] == 1)
            elif name == 'S3': test_best_mask = (X_test[:, 2] > 0.5)
            elif name == 'S4': test_best_mask = (X_test[:, 0] <= 24.0)

    stump_probs = np.where(test_best_mask, stump_rate_yes, stump_rate_no)

    def eval_preds(pred_binary):
        tp = int(np.sum((pred_binary == 1) & (y_test == 1)))
        fp = int(np.sum((pred_binary == 1) & (y_test == 0)))
        tn = int(np.sum((pred_binary == 0) & (y_test == 0)))
        fn = int(np.sum((pred_binary == 0) & (y_test == 1)))
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, (tp + fn))
        f1 = (2 * prec * rec) / max(1e-15, (prec + rec)) if (prec + rec) > 0 else 0.0
        return tp, fp, tn, fn, prec, rec, f1

    log_pred = (log_probs >= 0.3).astype(int)
    knn_pred = (knn_probs >= (K_best + 1) / (2 * K_best)).astype(int)
    stump_pred = (stump_probs >= 0.3).astype(int)

    hard_vote_pred = ((log_pred + knn_pred + stump_pred) >= 2).astype(int)
    soft_mean = (log_probs + knn_probs + stump_probs) / 3.0
    soft_vote_pred = (soft_mean >= 0.3).astype(int)

    voting_rows = []
    for m_name, p_bin in [('logistic', log_pred), ('knn', knn_pred), ('stump', stump_pred), ('hard_vote', hard_vote_pred), ('soft_vote', soft_vote_pred)]:
        tp, fp, tn, fn, prec, rec, f1 = eval_preds(p_bin)
        voting_rows.append(f"{m_name},{tp},{fp},{tn},{fn},{prec:.4f},{rec:.4f},{f1:.4f}")

    with open("outputs/part4_voting.csv", "w") as f:
        f.write("model,tp,fp,tn,fn,precision,recall,f1\n" + "\n".join(voting_rows) + "\n")

    # Task 4.3: AdaBoost by hand
    y_adaboost = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

    w = np.ones(8) / 8.0
    eps1 = np.sum(w[h1 != y_adaboost])
    alpha1 = 0.5 * np.log((1.0 - eps1) / eps1)

    unnorm1 = w * np.exp(-alpha1 * y_adaboost * h1)
    w1 = unnorm1 / np.sum(unnorm1)

    eps2 = np.sum(w1[h2 != y_adaboost])
    alpha2 = 0.5 * np.log((1.0 - eps2) / eps2)
    unnorm2 = w1 * np.exp(-alpha2 * y_adaboost * h2)
    w2 = unnorm2 / np.sum(unnorm2)

    adaboost_rows = []
    for i in range(8):
        adaboost_rows.append(f"1,{i+1},{w[i]:.4f},{unnorm1[i]:.4f},{w1[i]:.4f}")
    for i in range(8):
        adaboost_rows.append(f"2,{i+1},{w1[i]:.4f},{unnorm2[i]:.4f},{w2[i]:.4f}")

    with open("outputs/part4_adaboost.csv", "w") as f:
        f.write("round,point,weight_before,weight_unnormalised,weight_after\n" + "\n".join(adaboost_rows) + "\n")

    ensemble_preds = np.sign(alpha1 * h1 + alpha2 * h2)
    ensemble_accuracy = float(np.mean(ensemble_preds == y_adaboost))

    # Task 4.4: Out-of-bag share
    oob_fraction = (1.0 - 1.0 / 560.0) ** 560

    # Task 4.5: Neural Network sizing
    # Model A Option 1 & 2
    # Layer parameters include biases
    mlp_hidden1 = 784 * 16 + 16
    mlp_hidden2 = 16 * 16 + 16
    mlp_output = 16 * 10 + 10
    mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output

    # CNN Option 2 layers:
    # conv1: 5x5 filters, 6 filters, input 28x28x1, padding 2 -> output 28x28x6. Params: (5*5*1*6) + 6 = 156
    # maxpool1: 2x2 stride 2 -> output 14x14x6, 0 params
    # conv2: 5x5 filters, 16 filters, padding 0 -> output (14-5+1) = 10x10x16. Params: (5*5*6*16) + 16 = 2416
    # maxpool2: 2x2 stride 2 -> output 5x5x16 = 400, 0 params
    # dense1: 400 -> 32. Params: 400 * 32 + 32 = 12832
    # dense2: 32 -> 10. Params: 32 * 10 + 10 = 330
    cnn_conv1 = 156
    cnn_conv2 = 2416
    cnn_dense1 = 12832
    cnn_dense2 = 330
    cnn_total = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2

    arch_rows = [
        f"mlp_hidden1,{mlp_hidden1}",
        f"mlp_hidden2,{mlp_hidden2}",
        f"mlp_output,{mlp_output}",
        f"cnn_conv1,{cnn_conv1}",
        f"cnn_conv2,{cnn_conv2}",
        f"cnn_dense1,{cnn_dense1}",
        f"cnn_dense2,{cnn_dense2}"
    ]
    with open("outputs/part4_architecture.csv", "w") as f:
        f.write("layer,parameters\n" + "\n".join(arch_rows) + "\n")

    fc_layer_params = 128 * 128 * 3 * 1024 + 1024
    conv_layer_params = 3 * 3 * 3 * 64 + 64

    # Model B RNN parameters: W_xh (64 x 128), W_hh (128 x 128), W_hy (128 x 16), b_h (128), b_y (16)
    rnn_total = (64 * 128) + (128 * 128) + (128 * 16) + 128 + 16

    results = [
        f"root_entropy = {root_entropy:.4f}",
        f"root_gini = {root_gini:.4f}",
        f"best_split_by_info_gain = {best_s_info}",
        f"best_split_by_gini_gain = {best_s_gini}",
        f"stump_rate_yes = {stump_rate_yes:.4f}",
        f"stump_rate_no = {stump_rate_no:.4f}",
        f"round1_error = {eps1:.4f}",
        f"round1_alpha = {alpha1:.4f}",
        f"round2_error = {eps2:.4f}",
        f"round2_alpha = {alpha2:.4f}",
        f"ensemble_accuracy = {ensemble_accuracy:.4f}",
        f"oob_fraction = {oob_fraction:.4f}",
        f"mlp_total = {mlp_total}",
        f"cnn_total = {cnn_total}",
        f"fc_layer_params = {fc_layer_params}",
        f"conv_layer_params = {conv_layer_params}",
        f"rnn_total = {rnn_total}"
    ]

    with open("outputs/part4_results.txt", "w") as f:
        f.write("\n".join(results) + "\n")

if __name__ == "__main__":
    run_part4()

