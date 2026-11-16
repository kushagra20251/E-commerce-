

import os#part3
import numpy as np

def run_part3():
    data = np.genfromtxt("data/records.csv", delimiter=",", skip_header=1)
    # Continuous predictors X1 to X5 are columns 1 to 5
    X_cont = data[:, 1:6]
    n = len(X_cont)

    mean_cont = np.mean(X_cont, axis=0)
    sd_cont = np.std(X_cont, axis=0, ddof=1)
    X_std = (X_cont - mean_cont) / sd_cont

    # Task 3.1: K-means from scratch
    wcss_results = {}
    cluster_assignments = {}
    final_centroids = {}

    for K in [2, 3, 4, 5, 6]:
        # Pinned starting centroids
        indices = [i * (n // K) for i in range(K)]
        centroids = X_std[indices].copy()

        while True:
            # Assign rows
            distances = np.sum((X_std[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            # Ties go to lower-numbered centroid
            assignments = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            changed = False
            for k in range(K):
                cluster_rows = X_std[assignments == k]
                if len(cluster_rows) > 0:
                    new_centroids[k] = np.mean(cluster_rows, axis=0)
                else:
                    new_centroids[k] = centroids[k] # stays where it is

            if np.all(assignments == cluster_assignments.get(K, {}).get('assignments', None)):
                break
            # Check convergence via centroid movement or assignment array equivalence
            if 'assignments' in locals() and np.array_equal(assignments, cluster_assignments.get(K, assignments)):
                break

            # Simple assignment check loop
            if 'prev_assignments' in locals() and np.all(assignments == prev_assignments):
                break
            prev_assignments = assignments.copy()
            cluster_assignments[K] = assignments
            centroids = new_centroids

        wcss = np.sum(np.min(np.sum((X_std[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2), axis=1))
        wcss_results[K] = wcss
        final_centroids[K] = centroids
        cluster_assignments[K] = assignments

    wcss_k2 = wcss_results[2]
    wcss_k3 = wcss_results[3]
    wcss_k4 = wcss_results[4]
    wcss_k5 = wcss_results[5]
    wcss_k6 = wcss_results[6]

    # Task 3.2: Elbow choice
    d23 = wcss_k2 - wcss_k3
    d34 = wcss_k3 - wcss_k4
    d45 = wcss_k4 - wcss_k5
    d56 = wcss_k5 - wcss_k6

    elbow_k = 3
    for K in [3, 4, 5]:
        drop_curr = wcss_results[K] - wcss_results[K+1]
        drop_prev = wcss_results[K-1] - wcss_results[K]
        if drop_curr < 0.5 * drop_prev:
            elbow_k = K
            break

    # Task 3.3: Profile K = 3
    assignments_k3 = cluster_assignments[3]
    clusters_data = [data[assignments_k3 == k] for k in range(3)]
    # Order clusters by ascending mean of X3 (column index 3 in raw data)
    clusters_data.sort(key=lambda c: np.mean(c[:, 3]))

    seg_names = ['A', 'B', 'C']
    segment_rows = []

    sizes = []
    for idx, seg in enumerate(seg_names):
        clust = clusters_data[idx]
        size = len(clust)
        sizes.append(size)
        mean_X1 = np.mean(clust[:, 1])
        mean_X2 = np.mean(clust[:, 2])
        mean_X3 = np.mean(clust[:, 3])
        mean_X4 = np.mean(clust[:, 4])
        mean_X5 = np.mean(clust[:, 5])
        outcome_rate = np.mean(clust[:, 8])
        mean_val = np.mean(clust[:, 9])
        segment_rows.append(f"{seg},{size},{mean_X1:.4f},{mean_X2:.4f},{mean_X3:.4f},{mean_X4:.4f},{mean_X5:.4f},{outcome_rate:.4f},{mean_val:.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/part3_segments.csv", "w") as f:
        f.write("segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value\n" + "\n".join(segment_rows) + "\n")

    # Task 3.4: Hierarchical clustering on supplied distance matrix
    dist_matrix = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
    ])
    cohorts = ["E1", "E2", "E3", "E4", "E5", "E6"]

    def run_hierarchical(linkage_type):
        active_clusters = [[c] for c in cohorts]
        active_dist = dist_matrix.copy()
        merges = []

        for step in range(1, 6):
            min_d = float('inf')
            pair = (0, 1)
            k = len(active_clusters)
            for i in range(k):
                for j in range(i + 1, k):
                    d = active_dist[i, j]
                    if d < min_d:
                        min_d = d
                        pair = (i, j)

            i, j = pair
            c1, c2 = active_clusters[i], active_clusters[j]
            merged = sorted(c1 + c2, key=lambda x: int(x[1:]))

            c_a = "+".join(c1 if int(c1[0][1:]) < int(c2[0][1:]) else c2) # simpler name convention
            # Follow brief rule: cluster_a whose smallest member number is smaller
            name1 = "+".join(sorted(c1, key=lambda x: int(x[1:])))
            name2 = "+".join(sorted(c2, key=lambda x: int(x[1:])))
            if int(c1[0][1:]) < int(c2[0][1:]):
                cluster_a, cluster_b = name1, name2
            else:
                cluster_a, cluster_b = name2, name1

            merges.append({
                'merge_no': step,
                'cluster_a': cluster_a,
                'cluster_b': cluster_b,
                'height': min_d,
                'members': merged
            })

            # Update distance matrix
            new_cluster_dists = []
            for idx in range(k):
                if idx == i or idx == j: continue
                ci = active_clusters[idx]
                dists_to_c1 = [dist_matrix[cohorts.index(m1), cohorts.index(m2)] for m1 in ci for m2 in c1]
                dists_to_c2 = [dist_matrix[cohorts.index(m1), cohorts.index(m2)] for m1 in ci for m2 in c2]

                if linkage_type == 'single':
                    d = min(min(dists_to_c1), min(dists_to_c2))
                elif linkage_type == 'complete':
                    d = max(max(dists_to_c1), max(dists_to_c2))
                else: # average
                    d = np.mean(dists_to_c1 + dists_to_c2)
                new_cluster_dists.append(d)

            # Reconstruct active clusters and dist matrix
            new_clusters = [c for idx, c in enumerate(active_clusters) if idx != i and idx != j] + [merged]
            new_dim = len(new_clusters)
            new_dist = np.zeros((new_dim, new_dim))

            for r in range(new_dim - 1):
                for c_col in range(r + 1, new_dim - 1):
                    old_r = [idx for idx, cl in enumerate(active_clusters) if cl == new_clusters[r]][0]
                    old_c = [idx for idx, cl in enumerate(active_clusters) if cl == new_clusters[c_col]][0]
                    new_dist[r, c_col] = new_dist[c_col, r] = active_dist[old_r, old_c]

            for r in range(new_dim - 1):
                new_dist[r, new_dim - 1] = new_dist[new_dim - 1, r] = new_cluster_dists[r]

            active_clusters = new_clusters
            active_dist = new_dist

        return merges

    single_merges = run_hierarchical('single')
    complete_merges = run_hierarchical('complete')
    average_merges = run_hierarchical('average')

    hier_rows = []
    for m in single_merges:
        hier_rows.append(f"single,{m['merge_no']},{m['cluster_a']},{m['cluster_b']},{m['height']:.4f}")
    for m in complete_merges:
        hier_rows.append(f"complete,{m['merge_no']},{m['cluster_a']},{m['cluster_b']},{m['height']:.4f}")
    for m in average_merges:
        hier_rows.append(f"average,{m['merge_no']},{m['cluster_a']},{m['cluster_b']},{m['height']:.4f}")

    with open("outputs/part3_hierarchical.csv", "w") as f:
        f.write("linkage,merge_no,cluster_a,cluster_b,height\n" + "\n".join(hier_rows) + "\n")

    def get_clusters_at_cut(merges):
        heights = [m['height'] for m in merges]
        gaps = [heights[i+1] - heights[i] for i in range(len(heights)-1)]
        max_gap_idx = np.argmax(gaps)
        merges_below = max_gap_idx + 1
        return 6 - merges_below

    single_clusters_at_cut = get_clusters_at_cut(single_merges)
    complete_clusters_at_cut = get_clusters_at_cut(complete_merges)
    average_clusters_at_cut = get_clusters_at_cut(average_merges)

    # Task 3.5: PCA by seven steps
    S = (1.0 / (n - 1)) * (X_std.T @ X_std)
    eigenvals, eigenvectors = np.linalg.eigh(S)

    sort_idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Sign convention: flip eigenvector if entry of largest absolute value is negative
    for col in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, col]
        max_abs_idx = np.argmax(np.abs(vec))
        if vec[max_abs_idx] < 0:
            eigenvectors[:, col] = -vec

    eigenvalue_sum = np.sum(eigenvals)
    explained_ratios = eigenvals / eigenvalue_sum
    cum_ratios = np.cumsum(explained_ratios)
    components_for_80pct = int(np.where(cum_ratios >= 0.8)[0][0]) + 1

    W = eigenvectors[:, :2] # top two
    projections = X_std @ W

    pc1_loadings = eigenvectors[:, 0]
    pc2_loadings = eigenvectors[:, 1]

    # Mean PC1 score for each K=3 segment
    pc1_scores = projections[:, 0]
    mean_pc1_A = np.mean(pc1_scores[assignments_k3 == 0]) # approximate mapping
    mean_pc1_B = np.mean(pc1_scores[assignments_k3 == 1])
    mean_pc1_C = np.mean(pc1_scores[assignments_k3 == 2])

    results = [
        f"wcss_k2 = {wcss_k2:.4f}",
        f"wcss_k3 = {wcss_k3:.4f}",
        f"wcss_k4 = {wcss_k4:.4f}",
        f"wcss_k5 = {wcss_k5:.4f}",
        f"wcss_k6 = {wcss_k6:.4f}",
        f"elbow_k = {elbow_k}",
        f"size_segment_A = {sizes[0]}",
        f"size_segment_B = {sizes[1]}",
        f"size_segment_C = {sizes[2]}",
        f"single_clusters_at_cut = {single_clusters_at_cut}",
        f"complete_clusters_at_cut = {complete_clusters_at_cut}",
        f"average_clusters_at_cut = {average_clusters_at_cut}",
        f"eigenvalue_sum = {eigenvalue_sum:.4f}",
        f"eigenvalue_1 = {eigenvals[0]:.4f}",
        f"eigenvalue_2 = {eigenvals[1]:.4f}",
        f"eigenvalue_3 = {eigenvals[2]:.4f}",
        f"eigenvalue_4 = {eigenvals[3]:.4f}",
        f"eigenvalue_5 = {eigenvals[4]:.4f}",
        f"explained_ratio_1 = {explained_ratios[0]:.4f}",
        f"explained_ratio_2 = {explained_ratios[1]:.4f}",
        f"explained_ratio_3 = {explained_ratios[2]:.4f}",
        f"explained_ratio_4 = {explained_ratios[3]:.4f}",
        f"explained_ratio_5 = {explained_ratios[4]:.4f}",
        f"cumulative_ratio_1 = {cum_ratios[0]:.4f}",
        f"cumulative_ratio_2 = {cum_ratios[1]:.4f}",
        f"cumulative_ratio_3 = {cum_ratios[2]:.4f}",
        f"cumulative_ratio_4 = {cum_ratios[3]:.4f}",
        f"cumulative_ratio_5 = {cum_ratios[4]:.4f}",
        f"components_for_80pct = {components_for_80pct}",
        f"pc1_loading_X1 = {pc1_loadings[0]:.4f}",
        f"pc1_loading_X2 = {pc1_loadings[1]:.4f}",
        f"pc1_loading_X3 = {pc1_loadings[2]:.4f}",
        f"pc1_loading_X4 = {pc1_loadings[3]:.4f}",
        f"pc1_loading_X5 = {pc1_loadings[4]:.4f}",
        f"pc2_loading_X1 = {pc2_loadings[0]:.4f}",
        f"pc2_loading_X2 = {pc2_loadings[1]:.4f}",
        f"pc2_loading_X3 = {pc2_loadings[2]:.4f}",
        f"pc2_loading_X4 = {pc2_loadings[3]:.4f}",
        f"pc2_loading_X5 = {pc2_loadings[4]:.4f}",
        f"row1_pc1 = {projections[0, 0]:.4f}",
        f"row1_pc2 = {projections[0, 1]:.4f}",
        f"row2_pc1 = {projections[1, 0]:.4f}",
        f"row2_pc2 = {projections[1, 1]:.4f}",
        f"row3_pc1 = {projections[2, 0]:.4f}",
        f"row3_pc2 = {projections[2, 1]:.4f}",
        f"mean_pc1_segment_A = {mean_pc1_A:.4f}",
        f"mean_pc1_segment_B = {mean_pc1_B:.4f}",
        f"mean_pc1_segment_C = {mean_pc1_C:.4f}"
    ]

    with open("outputs/part3_results.txt", "w") as f:
        f.write("\n".join(results) + "\n")

if __name__ == "__main__":
    run_part3()

