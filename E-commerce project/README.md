# Flipkart E-Commerce Decision Intelligence Capstone

An end-to-end data science and machine learning pipeline built entirely from scratch using **NumPy** (`numpy==2.2.6`). This project covers statistical inference, regression modeling, binary classification, customer segmentation, dimensionality reduction, and ensemble learning on e-commerce transaction records.

---

## 📁 Repository Structure

```text
<repo>/
├── README.md               # Project documentation and execution guide
├── requirements.txt        # Python dependencies (numpy==2.2.6)
├── generate_data.py        # Synthetic dataset generator (creates data/records.csv)
├── utils.py                # Shared helper utilities
├── part1.py                # Statistical inference, OLS, Gradient Descent & Multiple Regression
├── part2.py                # Logistic Regression, KNN CV & Cost-Sensitive Thresholding
├── part3.py                # K-Means Clustering, Hierarchical Linkage & PCA
├── part4.py                # Decision Splits, Ensemble Voting, AdaBoost & NN Parameter Sizing
├── data/
│   └── records.csv         # Generated dataset (800 rows, 10 columns)
└── outputs/                # Generated analytical reports and evaluation metrics
    ├── part1_results.txt
    ├── part2_results.txt
    ├── part2_thresholds.csv
    ├── part2_knn_cv.csv
    ├── part3_results.txt
    ├── part3_segments.csv
    ├── part3_hierarchical.csv
    ├── part4_results.txt
    ├── part4_splits.csv
    ├── part4_voting.csv
    ├── part4_adaboost.csv
    └── part4_architecture.csv
