#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDG-PET connectome machine-learning analysis pipeline.

This script performs five complementary classification analyses:
1. Regional FDG-PET uptake -> XGBoost + leave-one-out cross-validation (LOOCV)
2. Metabolic connectome edges -> PCA + XGBoost + LOOCV
3. Metabolic connectome edges -> recursive feature elimination (RFE) + XGBoost + LOOCV
4. Global graph metrics -> XGBoost + LOOCV
5. Metabolic connectome edges -> connectome-based predictive modelling (CPM) + logistic regression + LOOCV

All model selection and feature selection steps are performed within the training data of each LOOCV fold.
Permutation tests are performed on the complete analysis pipeline and therefore preserve the cross-validation structure.
The permutation p-value uses the standard +1 correction: p = (1 + count(null AUC >= observed AUC)) / (1 + number of valid permutations).
With 500 valid permutations, the minimum attainable p-value is therefore 1/501 (~0.002).

The metabolic connectome is defined as C_ij = |log(x_i / x_j)|, where x_i and x_j are regional FDG-PET uptake values.
For N ROIs, the number of unique undirected edges is N_edges = N * (N - 1) / 2.

Input requirements
------------------
The input Excel workbook must contain:
1. An uptake sheet: one row per subject, one subject identifier column, one column per ROI, and numeric FDG-PET uptake values.
2. A clinical outcome sheet: one row per subject, the same subject identifier column, and one or more binary outcome columns.

Example command
---------------
python fdg_pet_connectome_analysis.py --input-excel data.xlsx --uptake-sheet FDGPET --labels-sheet clinical --id-column subject_id --labels "UPDRS=UPDRS_binary,MMSE=MMSE_binary,Hallucinations=hallucinations" --output-dir results

Dependencies
------------
numpy, pandas, scipy, scikit-learn, xgboost, networkx, joblib, openpyxl
"""

import argparse
import json
import logging
import warnings
from collections import Counter
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.linalg import eigvalsh
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

LOGGER = logging.getLogger("fdg_pet_analysis")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run FDG-PET connectome machine-learning analyses.")
    parser.add_argument("--input-excel", required=True, help="Path to the input Excel workbook.")
    parser.add_argument("--uptake-sheet", required=True, help="Excel sheet containing subject-level ROI uptake values.")
    parser.add_argument("--labels-sheet", required=True, help="Excel sheet containing binary clinical outcomes.")
    parser.add_argument("--id-column", required=True, help="Column containing the unique subject identifier.")
    parser.add_argument("--labels", required=True, help="Comma-separated outcome specifications in the form LABEL_NAME=COLUMN_NAME. Example: \"UPDRS=UPDRS_binary,MMSE=MMSE_binary,Dementia=Dementia_binary\".")
    parser.add_argument("--output-dir", required=True, help="Directory in which analysis results will be saved.")
    parser.add_argument("--n-permutations", type=int, default=500, help="Number of permutations for each permutation test. Default: 500.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs used for outcome-level analyses and permutation tests. Default: -1 (all available CPUs).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed. Default: 42.")
    parser.add_argument("--graph-density", type=float, default=0.20, help="Proportion of strongest edges retained for graph analysis. Must be between 0 and 1. Default: 0.20.")
    parser.add_argument("--graph-randomizations", type=int, default=20, help="Number of random weight permutations used for normalized graph efficiency. Default: 20.")
    parser.add_argument("--pca-variance", type=float, default=0.70, help="Cumulative variance retained by PCA. Default: 0.70.")
    parser.add_argument("--rfe-features", type=int, default=20, help="Number of features selected by RFE in each fold. Default: 20.")
    parser.add_argument("--rfe-stability", type=float, default=0.50, help="Minimum proportion of LOOCV folds in which an RFE feature must be selected to enter the consensus set. Default: 0.50.")
    parser.add_argument("--cpm-p-thresholds", default="0.05,0.01,0.001", help="Comma-separated point-biserial p-value thresholds for CPM. Default: 0.05,0.01,0.001.")
    parser.add_argument("--cpm-consensus", type=float, default=0.50, help="Minimum proportion of LOOCV folds in which a CPM edge must be selected to enter the consensus set. Default: 0.50.")
    parser.add_argument("--xgb-estimators", type=int, default=200, help="Number of XGBoost estimators. Default: 200.")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05, help="XGBoost learning rate. Default: 0.05.")
    parser.add_argument("--xgb-max-depth", type=int, default=3, help="XGBoost maximum tree depth. Default: 3.")
    return parser.parse_args()


def configure_logging():
    """Configure console logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def parse_label_mapping(label_argument):
    """Parse outcome specifications in the form LABEL_NAME=COLUMN_NAME."""
    mapping = {}
    for item in label_argument.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid label specification '{item}'. Expected format LABEL_NAME=COLUMN_NAME.")
        label_name, column_name = item.split("=", 1)
        label_name, column_name = label_name.strip(), column_name.strip()
        if not label_name or not column_name:
            raise ValueError(f"Invalid label specification '{item}'.")
        if label_name in mapping:
            raise ValueError(f"Duplicate outcome label: '{label_name}'.")
        mapping[label_name] = column_name
    if not mapping:
        raise ValueError("No outcome variables were provided.")
    return mapping


def validate_parameters(args):
    """Validate command-line parameters."""
    if not 0 < args.graph_density <= 1:
        raise ValueError("--graph-density must be > 0 and <= 1.")
    if not 0 < args.pca_variance <= 1:
        raise ValueError("--pca-variance must be > 0 and <= 1.")
    if not 0 < args.rfe_stability <= 1:
        raise ValueError("--rfe-stability must be > 0 and <= 1.")
    if not 0 < args.cpm_consensus <= 1:
        raise ValueError("--cpm-consensus must be > 0 and <= 1.")
    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be >= 1.")
    if args.rfe_features < 1:
        raise ValueError("--rfe-features must be >= 1.")
    if args.graph_randomizations < 1:
        raise ValueError("--graph-randomizations must be >= 1.")
    if args.xgb_estimators < 1:
        raise ValueError("--xgb-estimators must be >= 1.")
    if args.xgb_learning_rate <= 0:
        raise ValueError("--xgb-learning-rate must be > 0.")
    if args.xgb_max_depth < 1:
        raise ValueError("--xgb-max-depth must be >= 1.")
    p_thresholds = [float(x.strip()) for x in args.cpm_p_thresholds.split(",") if x.strip()]
    if not p_thresholds:
        raise ValueError("At least one CPM p-value threshold is required.")
    if any(p <= 0 or p >= 1 for p in p_thresholds):
        raise ValueError("All CPM p-value thresholds must be between 0 and 1.")
    return p_thresholds


def load_input_data(input_excel, uptake_sheet, labels_sheet, id_column):
    """Load and validate input data."""
    input_excel = Path(input_excel)
    if not input_excel.exists():
        raise FileNotFoundError(f"Input Excel file not found: {input_excel}")
    LOGGER.info("Loading input workbook: %s", input_excel)
    uptake_df = pd.read_excel(input_excel, sheet_name=uptake_sheet, index_col=None)
    labels_df = pd.read_excel(input_excel, sheet_name=labels_sheet, index_col=None)
    if id_column not in uptake_df.columns:
        raise ValueError(f"ID column '{id_column}' was not found in uptake sheet.")
    if id_column not in labels_df.columns:
        raise ValueError(f"ID column '{id_column}' was not found in labels sheet.")
    if uptake_df[id_column].duplicated().any():
        raise ValueError("Duplicate subject identifiers were found in the uptake sheet.")
    if labels_df[id_column].duplicated().any():
        raise ValueError("Duplicate subject identifiers were found in the labels sheet.")
    uptake_df = uptake_df.set_index(id_column)
    labels_df = labels_df.set_index(id_column)
    common_subjects = uptake_df.index.intersection(labels_df.index)
    if len(common_subjects) < 10:
        raise ValueError("Fewer than 10 subjects are shared between the uptake and clinical sheets.")
    uptake_df = uptake_df.loc[common_subjects].copy()
    labels_df = labels_df.loc[common_subjects].copy()
    LOGGER.info("Found %d subjects shared between uptake and clinical data.", len(common_subjects))
    return uptake_df, labels_df


def prepare_uptake_matrix(uptake_df):
    """Prepare the ROI uptake matrix."""
    if uptake_df.shape[1] < 2:
        raise ValueError("The uptake sheet must contain at least two ROI columns.")
    numeric_df = uptake_df.apply(pd.to_numeric, errors="coerce")
    non_numeric_columns = [column for column in uptake_df.columns if numeric_df[column].isna().all() and not uptake_df[column].isna().all()]
    if non_numeric_columns:
        raise ValueError("The following uptake columns are not numeric: " + ", ".join(map(str, non_numeric_columns)))
    if numeric_df.isna().all(axis=0).any():
        invalid_columns = numeric_df.columns[numeric_df.isna().all(axis=0)].tolist()
        raise ValueError("The following ROI columns contain no numeric values: " + ", ".join(map(str, invalid_columns)))
    if numeric_df.columns.duplicated().any():
        raise ValueError("Duplicate ROI column names were found.")
    if (numeric_df <= 0).any().any():
        invalid_count = int((numeric_df <= 0).sum().sum())
        raise ValueError(f"Found {invalid_count} non-positive uptake values. Log-ratio connectivity requires strictly positive uptake values.")
    if numeric_df.isna().any().any():
        LOGGER.warning("Missing uptake values were detected. Subjects with missing values will be excluded before model fitting.")
    return numeric_df


def build_logratio_connectomes(data):
    """Construct subject-specific metabolic connectomes using C_ij = |log(x_i / x_j)|."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Input uptake data must be a 2-dimensional matrix.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Uptake data contain non-finite values.")
    if np.any(data <= 0):
        raise ValueError("Uptake values must be strictly positive.")
    log_data = np.log(data)
    connectomes = np.abs(log_data[:, :, None] - log_data[:, None, :])
    connectomes = np.transpose(connectomes, (1, 2, 0))
    for subject_index in range(connectomes.shape[2]):
        np.fill_diagonal(connectomes[:, :, subject_index], 0.0)
    return connectomes


def build_edge_to_roi(roi_names):
    """Return the mapping between unique edges and ROI indices."""
    n_roi = len(roi_names)
    return [(i, j) for i in range(n_roi) for j in range(i + 1, n_roi)]


def extract_upper_triangle(connectomes):
    """Extract unique undirected edges from subject-specific connectomes."""
    n_roi = connectomes.shape[0]
    upper_triangle = np.triu_indices(n_roi, k=1)
    return np.asarray([connectomes[:, :, subject_index][upper_triangle] for subject_index in range(connectomes.shape[2])])


def create_xgb_classifier(random_state):
    """Create the standard XGBoost classifier used throughout the pipeline."""
    return XGBClassifier(eval_metric="logloss", n_estimators=200, learning_rate=0.05, max_depth=3, random_state=random_state, n_jobs=1)


def validate_binary_outcome(y, label_name):
    """Validate a binary outcome vector."""
    y = np.asarray(y)
    valid_mask = ~pd.isna(y)
    y_valid = y[valid_mask]
    if y_valid.size < 10:
        raise ValueError(f"Outcome '{label_name}' contains fewer than 10 non-missing subjects.")
    unique_values = np.unique(y_valid)
    if len(unique_values) != 2:
        raise ValueError(f"Outcome '{label_name}' must contain exactly two classes. Found: {unique_values.tolist()}")
    try:
        y_numeric = y_valid.astype(int)
    except Exception as exc:
        raise ValueError(f"Outcome '{label_name}' could not be converted to integers.") from exc
    unique_numeric = np.unique(y_numeric)
    if not np.array_equal(unique_numeric, np.array([0, 1])):
        mapping = {value: index for index, value in enumerate(unique_numeric)}
        y_numeric = np.asarray([mapping[value] for value in y_numeric], dtype=int)
        LOGGER.warning("Outcome '%s' was automatically encoded as %s.", label_name, mapping)
    counts = Counter(y_numeric)
    if min(counts.values()) < 2:
        raise ValueError(f"Outcome '{label_name}' must contain at least two subjects in each class for LOOCV.")
    return valid_mask, y_numeric


def collect_results(y_true, y_pred, y_score):
    """Collect standard classification metrics."""
    y_true, y_pred, y_score = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_score)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = np.nan
    return {"accuracy": float(accuracy), "balanced_accuracy": float(balanced_accuracy), "roc_auc": float(roc_auc), "confusion_matrix": confusion, "true_labels": y_true.tolist(), "predicted_labels": y_pred.tolist(), "scores": y_score}


def permutation_test_auc(pipeline_function, X, y, n_permutations, random_state, n_jobs, **kwargs):
    """Perform a complete-pipeline permutation test by permuting the outcome labels."""
    observed = pipeline_function(X, y, **kwargs)
    observed_auc = observed.get("roc_auc", np.nan)
    if not np.isfinite(observed_auc):
        return {"auc_real": np.nan, "perm_aucs": np.array([]), "perm_mean": np.nan, "perm_std": np.nan, "p_value": np.nan}
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**31 - 1, size=n_permutations)
    if n_jobs == 1:
        permutation_aucs = [_run_permutation(pipeline_function, X, y, seed, kwargs) for seed in seeds]
    else:
        permutation_aucs = Parallel(n_jobs=n_jobs)(delayed(_run_permutation)(pipeline_function, X, y, seed, kwargs) for seed in seeds)
    permutation_aucs = np.asarray(permutation_aucs, dtype=float)
    permutation_aucs = permutation_aucs[np.isfinite(permutation_aucs)]
    if permutation_aucs.size == 0:
        p_value, permutation_mean, permutation_std = np.nan, np.nan, np.nan
    else:
        p_value = (1.0 + np.sum(permutation_aucs >= observed_auc)) / (1.0 + permutation_aucs.size)
        permutation_mean = float(np.mean(permutation_aucs))
        permutation_std = float(np.std(permutation_aucs, ddof=1)) if permutation_aucs.size > 1 else 0.0
    return {"auc_real": float(observed_auc), "perm_aucs": permutation_aucs, "perm_mean": permutation_mean, "perm_std": permutation_std, "p_value": float(p_value)}


def _run_permutation(pipeline_function, X, y, seed, kwargs):
    """Run one permutation replicate."""
    rng = np.random.RandomState(int(seed))
    y_permuted = rng.permutation(y)
    result = pipeline_function(X, y_permuted, **kwargs)
    return result.get("roc_auc", np.nan)


def permutation_test_cpm(X, y, p_thresholds, n_permutations, random_state, n_jobs, consensus_fraction):
    """Perform permutation testing for all CPM thresholds."""
    observed_results = classify_loocv_cpm_all_thresholds(X, y, p_thresholds, consensus_fraction=consensus_fraction)
    observed_auc = {result["p_thresh"]: result["roc_auc"] for result in observed_results}
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**31 - 1, size=n_permutations)
    if n_jobs == 1:
        permutation_results = [_run_cpm_permutation(X, y, p_thresholds, consensus_fraction, seed) for seed in seeds]
    else:
        permutation_results = Parallel(n_jobs=n_jobs)(delayed(_run_cpm_permutation)(X, y, p_thresholds, consensus_fraction, seed) for seed in seeds)
    output = []
    for threshold_index, threshold in enumerate(p_thresholds):
        null_auc = np.asarray([result[threshold_index] for result in permutation_results], dtype=float)
        null_auc = null_auc[np.isfinite(null_auc)]
        observed = observed_auc[threshold]
        if null_auc.size == 0 or not np.isfinite(observed):
            p_value, null_mean, null_std = np.nan, np.nan, np.nan
        else:
            p_value = (1.0 + np.sum(null_auc >= observed)) / (1.0 + null_auc.size)
            null_mean = float(np.mean(null_auc))
            null_std = float(np.std(null_auc, ddof=1)) if null_auc.size > 1 else 0.0
        output.append({"p_thresh": threshold, "auc_real": observed, "perm_aucs": null_auc, "perm_mean": null_mean, "perm_std": null_std, "p_value": float(p_value)})
    return observed_results, output


def _run_cpm_permutation(X, y, p_thresholds, consensus_fraction, seed):
    """Run one CPM permutation across all p-value thresholds."""
    rng = np.random.RandomState(int(seed))
    y_permuted = rng.permutation(y)
    results = classify_loocv_cpm_all_thresholds(X, y_permuted, p_thresholds, consensus_fraction=consensus_fraction)
    return [result.get("roc_auc", np.nan) for result in results]


def proportional_threshold_matrix(matrix, density):
    """Retain the strongest proportion of edges in a weighted matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    n_roi = matrix.shape[0]
    upper_triangle = np.triu_indices(n_roi, k=1)
    edge_values = np.abs(matrix[upper_triangle])
    if edge_values.size == 0:
        return np.zeros_like(matrix)
    n_edges_to_keep = max(1, int(np.floor(density * edge_values.size)))
    threshold = np.partition(edge_values, -n_edges_to_keep)[-n_edges_to_keep]
    thresholded = np.where(np.abs(matrix) >= threshold, matrix, 0.0)
    thresholded = np.triu(thresholded, k=1)
    thresholded = thresholded + thresholded.T
    np.fill_diagonal(thresholded, 0.0)
    return thresholded


def compute_weighted_global_metrics(adjacency, n_randomizations=20, random_state=42):
    """Compute weighted and binary global graph metrics."""
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    n_roi = adjacency.shape[0]
    weights = np.abs(adjacency).copy()
    np.fill_diagonal(weights, 0.0)
    metric_names = ["mean_strength", "strength_variance", "global_efficiency", "normalized_global_efficiency", "mean_clustering", "degree_variance", "algebraic_connectivity", "assortativity", "transitivity", "spectral_radius"]
    if not np.any(weights > 0):
        return {name: np.nan for name in metric_names}
    strength = np.sum(weights, axis=0)
    mean_strength = float(np.mean(strength))
    strength_variance = float(np.var(strength))
    distances = np.full_like(weights, np.inf, dtype=float)
    positive_edges = weights > 0
    distances[positive_edges] = 1.0 / weights[positive_edges]
    np.fill_diagonal(distances, 0.0)
    try:
        shortest_paths = shortest_path(distances, directed=False, unweighted=False)
        n_pairs = n_roi * (n_roi - 1) / 2
        upper_triangle = np.triu_indices(n_roi, k=1)
        finite_pairs = np.isfinite(shortest_paths[upper_triangle])
        global_efficiency = float(np.sum(1.0 / shortest_paths[upper_triangle][finite_pairs]) / n_pairs) if np.any(finite_pairs) else 0.0
    except Exception:
        global_efficiency = np.nan
    normalized_global_efficiency = np.nan
    try:
        upper_triangle = np.triu_indices(n_roi, k=1)
        edge_mask = weights[upper_triangle] > 0
        edge_weights = weights[upper_triangle][edge_mask]
        random_efficiencies = []
        rng = np.random.RandomState(random_state)
        rows, columns = upper_triangle[0][edge_mask], upper_triangle[1][edge_mask]
        for _ in range(n_randomizations):
            randomized = np.zeros_like(weights)
            shuffled_weights = rng.permutation(edge_weights)
            randomized[rows, columns] = shuffled_weights
            randomized[columns, rows] = shuffled_weights
            random_distances = np.full_like(randomized, np.inf, dtype=float)
            positive_random_edges = randomized > 0
            random_distances[positive_random_edges] = 1.0 / randomized[positive_random_edges]
            np.fill_diagonal(random_distances, 0.0)
            random_shortest_paths = shortest_path(random_distances, directed=False, unweighted=False)
            random_pair_distances = random_shortest_paths[upper_triangle]
            finite_random = np.isfinite(random_pair_distances)
            if np.any(finite_random):
                random_efficiencies.append(float(np.sum(1.0 / random_pair_distances[finite_random]) / n_pairs))
        if random_efficiencies and np.isfinite(global_efficiency):
            mean_random_efficiency = np.mean(random_efficiencies)
            if mean_random_efficiency > 0:
                normalized_global_efficiency = global_efficiency / mean_random_efficiency
    except Exception:
        normalized_global_efficiency = np.nan
    try:
        graph = nx.from_numpy_array(weights)
        clustering = nx.clustering(graph, weight="weight")
        mean_clustering = float(np.mean(list(clustering.values())))
    except Exception:
        mean_clustering = np.nan
    binary_adjacency = (weights > 0).astype(int)
    degree = np.sum(binary_adjacency, axis=0)
    degree_variance = float(np.var(degree))
    try:
        laplacian = np.diag(np.sum(weights, axis=0)) - weights
        eigenvalues = eigvalsh(laplacian)
        algebraic_connectivity = float(np.sort(eigenvalues)[1]) if eigenvalues.size >= 2 else np.nan
    except Exception:
        algebraic_connectivity = np.nan
    try:
        binary_graph = nx.from_numpy_array(binary_adjacency)
        assortativity = float(nx.degree_pearson_correlation_coefficient(binary_graph))
    except Exception:
        assortativity = np.nan
    try:
        transitivity = float(nx.transitivity(binary_graph))
    except Exception:
        transitivity = np.nan
    try:
        spectral_radius = float(np.max(np.abs(eigvalsh(weights))))
    except Exception:
        spectral_radius = np.nan
    return {"mean_strength": mean_strength, "strength_variance": strength_variance, "global_efficiency": global_efficiency, "normalized_global_efficiency": normalized_global_efficiency, "mean_clustering": mean_clustering, "degree_variance": degree_variance, "algebraic_connectivity": algebraic_connectivity, "assortativity": assortativity, "transitivity": transitivity, "spectral_radius": spectral_radius}


def graph_global_metrics_from_connectomes(connectomes, density, n_randomizations, random_state):
    """Calculate graph-level features for every subject."""
    n_subjects = connectomes.shape[2]
    feature_rows, metric_names = [], None
    for subject_index in range(n_subjects):
        adjacency = connectomes[:, :, subject_index].copy()
        np.fill_diagonal(adjacency, 0.0)
        thresholded = proportional_threshold_matrix(adjacency, density=density)
        metrics = compute_weighted_global_metrics(thresholded, n_randomizations=n_randomizations, random_state=random_state + subject_index)
        if metric_names is None:
            metric_names = list(metrics.keys())
        feature_rows.append([metrics[name] for name in metric_names])
    return np.asarray(feature_rows, dtype=float), metric_names


def classify_loocv_xgb(X, y, random_state=42, top_k=(5, 10), xgb_estimators=200, xgb_learning_rate=0.05, xgb_max_depth=3):
    """Run MinMax scaling + XGBoost classification under LOOCV."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional matrix.")
    if X.shape[0] != len(y):
        raise ValueError("X and y have incompatible numbers of subjects.")
    n_subjects, n_features = X.shape
    loo = LeaveOneOut()
    y_true, y_pred, y_scores = [], [], []
    feature_importances = np.zeros((n_subjects, n_features), dtype=float)
    feature_ranks = np.zeros((n_subjects, n_features), dtype=float)
    top_k = tuple(sorted({int(k) for k in top_k if int(k) > 0 and int(k) <= n_features}))
    top_k_counts = {k: np.zeros(n_features, dtype=float) for k in top_k}
    for fold_index, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        scaler = MinMaxScaler()
        X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
        classifier = create_xgb_classifier(random_state=random_state, n_estimators=xgb_estimators, learning_rate=xgb_learning_rate, max_depth=xgb_max_depth)
        classifier.fit(X_train_scaled, y_train)
        score = float(classifier.predict_proba(X_test_scaled)[0, 1])
        prediction = int(classifier.predict(X_test_scaled)[0])
        y_true.append(int(y_test[0])); y_pred.append(prediction); y_scores.append(score)
        importance = np.asarray(classifier.feature_importances_, dtype=float)
        feature_importances[fold_index] = importance
        ranks = np.argsort(np.argsort(-importance)) + 1
        feature_ranks[fold_index] = ranks
        for k in top_k:
            top_k_counts[k] += (ranks <= k).astype(float)
    results = collect_results(y_true, y_pred, y_scores)
    results["num_features"] = int(n_features)
    results["feature_importance"] = {"mean_importance": np.mean(feature_importances, axis=0), "std_importance": np.std(feature_importances, axis=0), "mean_rank": np.mean(feature_ranks, axis=0)}
    for k in top_k:
        results["feature_importance"][f"freq_top{k}"] = top_k_counts[k] / n_subjects
    return results


def classify_pca_xgb(X, y, variance_threshold=0.70, random_state=42, xgb_estimators=200, xgb_learning_rate=0.05, xgb_max_depth=3):
    """Run PCA + XGBoost under LOOCV and derive fold-wise edge and ROI contributions from retained PCA components."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    loo = LeaveOneOut()
    y_true, y_pred, y_scores = [], [], []
    all_components, all_explained_variance, all_n_components, all_edge_contributions = [], [], [], []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        scaler = MinMaxScaler()
        X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
        pca_full = PCA().fit(X_train_scaled)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
        n_components = min(n_components, X_train_scaled.shape[0], X_train_scaled.shape[1])
        pca = PCA(n_components=n_components)
        X_train_pca, X_test_pca = pca.fit_transform(X_train_scaled), pca.transform(X_test_scaled)
        classifier = create_xgb_classifier(random_state=random_state, n_estimators=xgb_estimators, learning_rate=xgb_learning_rate, max_depth=xgb_max_depth)
        classifier.fit(X_train_pca, y_train)
        score = float(classifier.predict_proba(X_test_pca)[0, 1]); prediction = int(classifier.predict(X_test_pca)[0])
        y_true.append(int(y_test[0])); y_pred.append(prediction); y_scores.append(score)
        components = np.asarray(pca.components_, dtype=float)
        explained_variance = np.asarray(pca.explained_variance_ratio_, dtype=float)
        edge_contribution = np.sum((components ** 2) * explained_variance[:, None], axis=0)
        all_components.append(components); all_explained_variance.append(explained_variance); all_n_components.append(n_components); all_edge_contributions.append(edge_contribution)
    max_components = max(component.shape[0] for component in all_components)
    n_features = X.shape[1]
    component_stack = np.zeros((len(all_components), max_components, n_features), dtype=float)
    variance_stack = np.zeros((len(all_explained_variance), max_components), dtype=float)
    for index, (components, explained_variance) in enumerate(zip(all_components, all_explained_variance)):
        component_stack[index, :components.shape[0], :] = components
        variance_stack[index, :explained_variance.shape[0]] = explained_variance
    edge_contribution_matrix = np.vstack(all_edge_contributions)
    results = collect_results(y_true, y_pred, y_scores)
    results["num_features"] = int(round(np.mean(all_n_components)))
    results["pca"] = {"mean_n_components": int(round(np.mean(all_n_components))), "mean_components": np.mean(component_stack, axis=0), "mean_explained_variance": np.mean(variance_stack, axis=0), "n_components_per_fold": all_n_components, "edge_contribution_per_fold": edge_contribution_matrix, "mean_edge_contribution": np.mean(edge_contribution_matrix, axis=0), "std_edge_contribution": np.std(edge_contribution_matrix, axis=0)}
    return results


def map_edge_contribution_to_rois(edge_contribution, edge_to_roi, n_rois):
    """Aggregate edge-level PCA contribution to the two incident ROIs."""
    roi_contribution = np.zeros(n_rois, dtype=float)
    for edge_index, (roi_i, roi_j) in enumerate(edge_to_roi):
        roi_contribution[roi_i] += edge_contribution[edge_index]
        roi_contribution[roi_j] += edge_contribution[edge_index]
    return roi_contribution


def classify_loocv_rfe_xgb(X, y, n_features_to_select=20, stability_threshold=0.50, random_state=42, xgb_estimators=200, xgb_learning_rate=0.05, xgb_max_depth=3):
    """Run per-fold RFE + XGBoost under LOOCV; report only fold-wise selection and selection-frequency summaries."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    loo = LeaveOneOut()
    y_true, y_pred, y_scores, selected_features_all = [], [], [], []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        scaler = MinMaxScaler()
        X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
        n_features = min(n_features_to_select, X_train_scaled.shape[1])
        base_classifier = create_xgb_classifier(random_state=random_state, n_estimators=xgb_estimators, learning_rate=xgb_learning_rate, max_depth=xgb_max_depth)
        selector = RFE(estimator=base_classifier, n_features_to_select=n_features, step=0.20)
        selector.fit(X_train_scaled, y_train)
        selected_indices = np.flatnonzero(selector.support_)
        selected_features_all.append(selected_indices)
        X_train_selected, X_test_selected = selector.transform(X_train_scaled), selector.transform(X_test_scaled)
        final_classifier = create_xgb_classifier(random_state=random_state, n_estimators=xgb_estimators, learning_rate=xgb_learning_rate, max_depth=xgb_max_depth)
        final_classifier.fit(X_train_selected, y_train)
        score = float(final_classifier.predict_proba(X_test_selected)[0, 1]); prediction = int(final_classifier.predict(X_test_selected)[0])
        y_true.append(int(y_test[0])); y_pred.append(prediction); y_scores.append(score)
    results = collect_results(y_true, y_pred, y_scores)
    n_folds = len(selected_features_all)
    selection_counter = Counter(feature for fold_features in selected_features_all for feature in fold_features)
    minimum_count = int(np.ceil(stability_threshold * n_folds))
    stable_features = sorted([feature for feature, count in selection_counter.items() if count >= minimum_count])
    if not stable_features:
        stable_features = [feature for feature, _ in selection_counter.most_common(min(n_features_to_select, X.shape[1]))]
    consensus_features = np.asarray(stable_features, dtype=int)
    selection_frequency = np.zeros(X.shape[1], dtype=float)
    for feature, count in selection_counter.items():
        selection_frequency[feature] = count / n_folds
    results.update({"selected_features_all": selected_features_all, "consensus_features": consensus_features, "selection_frequency": selection_frequency, "num_features": int(consensus_features.size)})
    return results


def classify_loocv_cpm_all_thresholds(X_edges, y, p_thresholds, consensus_fraction=0.50):
    """Run CPM for multiple p-value thresholds with feature selection restricted to each LOOCV training fold."""
    X, y = np.asarray(X_edges, dtype=float), np.asarray(y, dtype=int)
    n_edges = X.shape[1]
    all_results = []
    for p_threshold in p_thresholds:
        y_true, y_pred, y_scores, selected_edges_per_fold = [], [], [], []
        positive_counter, negative_counter = Counter(), Counter()
        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(X):
            X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
            correlations, p_values = np.zeros(n_edges, dtype=float), np.ones(n_edges, dtype=float)
            for edge_index in range(n_edges):
                edge_values = X_train[:, edge_index]
                if np.nanstd(edge_values) == 0:
                    continue
                correlation, p_value = stats.pointbiserialr(y_train, edge_values)
                if np.isfinite(correlation): correlations[edge_index] = correlation
                if np.isfinite(p_value): p_values[edge_index] = p_value
            positive_indices = np.flatnonzero((p_values < p_threshold) & (correlations > 0)); negative_indices = np.flatnonzero((p_values < p_threshold) & (correlations < 0))
            selected_edges_per_fold.append({"positive": positive_indices, "negative": negative_indices}); positive_counter.update(positive_indices.tolist()); negative_counter.update(negative_indices.tolist())
            positive_train = np.sum(X_train[:, positive_indices], axis=1) if positive_indices.size else np.zeros(X_train.shape[0]); positive_test = np.sum(X_test[:, positive_indices], axis=1) if positive_indices.size else np.zeros(X_test.shape[0])
            negative_train = np.sum(X_train[:, negative_indices], axis=1) if negative_indices.size else np.zeros(X_train.shape[0]); negative_test = np.sum(X_test[:, negative_indices], axis=1) if negative_indices.size else np.zeros(X_test.shape[0])
            X_train_features, X_test_features = np.column_stack([positive_train, negative_train]), np.column_stack([positive_test, negative_test])
            scaler = MinMaxScaler(); X_train_scaled, X_test_scaled = scaler.fit_transform(X_train_features), scaler.transform(X_test_features)
            classifier = LogisticRegression(solver="liblinear", class_weight="balanced"); classifier.fit(X_train_scaled, y_train)
            score = float(classifier.predict_proba(X_test_scaled)[0, 1]); prediction = int(classifier.predict(X_test_scaled)[0])
            y_true.append(int(y_test[0])); y_pred.append(prediction); y_scores.append(score)
        n_folds = len(selected_edges_per_fold); minimum_count = int(np.ceil(consensus_fraction * n_folds))
        positive_consensus = np.asarray(sorted([edge for edge, count in positive_counter.items() if count >= minimum_count]), dtype=int); negative_consensus = np.asarray(sorted([edge for edge, count in negative_counter.items() if count >= minimum_count]), dtype=int); consensus_edges = np.concatenate([positive_consensus, negative_consensus])
        metrics = collect_results(y_true, y_pred, y_scores); metrics.update({"p_thresh": float(p_threshold), "selected_edges_per_fold": selected_edges_per_fold, "positive_consensus": positive_consensus, "negative_consensus": negative_consensus, "consensus_edges": consensus_edges, "num_features": int(consensus_edges.size)}); all_results.append(metrics)
    return all_results


def save_pickle(obj, path):
    """Save an object as a compressed pickle file."""
    joblib.dump(obj, path, compress=3)


def create_results_record(label_name, method, classifier_name, results):
    """Create a compact row for the summary table."""
    confusion = results.get("confusion_matrix", np.full((2, 2), np.nan))
    return {"Label": label_name, "Method": method, "Classifier": classifier_name, "Accuracy": results.get("accuracy", np.nan), "Balanced Accuracy": results.get("balanced_accuracy", np.nan), "ROC-AUC": results.get("roc_auc", np.nan), "TP": confusion[1, 1] if confusion.shape == (2, 2) else np.nan, "FN": confusion[1, 0] if confusion.shape == (2, 2) else np.nan, "FP": confusion[0, 1] if confusion.shape == (2, 2) else np.nan, "TN": confusion[0, 0] if confusion.shape == (2, 2) else np.nan, "NumFeatures": results.get("num_features", np.nan), "Permutation Mean AUC": results.get("perm_mean_auc", np.nan), "Permutation SD AUC": results.get("perm_std_auc", np.nan), "Permutation P-value": results.get("perm_p_value", np.nan)}


def create_roc_record(method, classifier_name, results):
    """Create an ROC curve record."""
    try:
        false_positive_rate, true_positive_rate, _ = roc_curve(results["true_labels"], results["scores"]); roc_auc = float(roc_auc_score(results["true_labels"], results["scores"]))
    except ValueError:
        false_positive_rate, true_positive_rate, roc_auc = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.nan
    return {"method": method, "classifier": classifier_name, "fpr": false_positive_rate, "tpr": true_positive_rate, "auc": roc_auc}


def run_single_outcome(label_name, outcome_column, uptake_df, connectomes, labels_df, output_dir, graph_density, graph_randomizations, pca_variance, rfe_features, rfe_stability, cpm_p_thresholds, cpm_consensus, n_permutations, random_state, n_jobs, xgb_estimators, xgb_learning_rate, xgb_max_depth):
    """Run all analyses for one clinical outcome."""
    LOGGER.info("Processing outcome '%s' (%s).", label_name, outcome_column)
    if outcome_column not in labels_df.columns:
        raise ValueError(f"Outcome column '{outcome_column}' was not found.")
    outcome = labels_df[outcome_column]
    valid_outcome_mask, y = validate_binary_outcome(outcome, label_name)
    subject_ids = labels_df.index[valid_outcome_mask]
    uptake_subset = uptake_df.loc[subject_ids]
    if uptake_subset.isna().any().any():
        missing_subjects = uptake_subset.index[uptake_subset.isna().any(axis=1)]
        LOGGER.warning("Outcome '%s': excluding %d subjects with missing uptake values.", label_name, len(missing_subjects))
        complete_mask = ~uptake_subset.isna().any(axis=1); uptake_subset = uptake_subset.loc[complete_mask]; y = y[complete_mask.to_numpy()]; subject_ids = uptake_subset.index
    if len(subject_ids) < 10:
        LOGGER.warning("Skipping '%s': fewer than 10 complete subjects.", label_name); return None
    if len(np.unique(y)) != 2:
        LOGGER.warning("Skipping '%s': fewer than two classes remain after missing-data filtering.", label_name); return None
    subject_indices = labels_df.index.get_indexer(subject_ids)
    if np.any(subject_indices < 0):
        raise RuntimeError("Subject alignment failed.")
    X_roi = uptake_subset.to_numpy(dtype=float); X_edges = extract_upper_triangle(connectomes[:, :, subject_indices]); connectomes_subset = connectomes[:, :, subject_indices]
    outcome_output_dir = output_dir / _safe_filename(label_name); outcome_output_dir.mkdir(parents=True, exist_ok=True)
    results_records, roc_records = [], []

    LOGGER.info("[%s] 1/5 Regional uptake -> XGBoost", label_name)
    regional_results = classify_loocv_xgb(X_roi, y, random_state=random_state)
    regional_permutation = permutation_test_auc(classify_loocv_xgb, X_roi, y, n_permutations=n_permutations, random_state=random_state, n_jobs=n_jobs)
    regional_results.update({"perm_mean_auc": regional_permutation["perm_mean"], "perm_std_auc": regional_permutation["perm_std"], "perm_p_value": regional_permutation["p_value"]})
    results_records.append(create_results_record(label_name, "Regional_raw", "XGBoost", regional_results)); roc_records.append(create_roc_record("Regional_raw", "XGBoost", regional_results)); save_pickle(regional_results, outcome_output_dir / "regional_raw_results.pkl")

    LOGGER.info("[%s] 2/5 Connectome -> PCA -> XGBoost", label_name)
    pca_results = classify_pca_xgb(X_edges, y, variance_threshold=pca_variance, random_state=random_state)
    pca_permutation = permutation_test_auc(classify_pca_xgb, X_edges, y, n_permutations=n_permutations, random_state=random_state, n_jobs=n_jobs, variance_threshold=pca_variance)
    edge_to_roi = build_edge_to_roi(list(uptake_df.columns)); pca_results["pca"]["mean_roi_contribution"] = map_edge_contribution_to_rois(pca_results["pca"]["mean_edge_contribution"], edge_to_roi, X_roi.shape[1]); pca_results["pca"]["std_roi_contribution"] = np.std(np.vstack([map_edge_contribution_to_rois(edge_values, edge_to_roi, X_roi.shape[1]) for edge_values in pca_results["pca"]["edge_contribution_per_fold"]]), axis=0); pca_results.update({"perm_mean_auc": pca_permutation["perm_mean"], "perm_std_auc": pca_permutation["perm_std"], "perm_p_value": pca_permutation["p_value"]})
    results_records.append(create_results_record(label_name, "Connectome_PCA", "XGBoost", pca_results)); roc_records.append(create_roc_record("Connectome_PCA", "XGBoost", pca_results)); save_pickle(pca_results, outcome_output_dir / "connectome_pca_results.pkl")

    LOGGER.info("[%s] 3/5 Connectome -> RFE -> XGBoost", label_name)
    rfe_results = classify_loocv_rfe_xgb(X_edges, y, n_features_to_select=rfe_features, stability_threshold=rfe_stability, random_state=random_state)
    rfe_permutation = permutation_test_auc(classify_loocv_rfe_xgb, X_edges, y, n_permutations=n_permutations, random_state=random_state, n_jobs=n_jobs, n_features_to_select=rfe_features, stability_threshold=rfe_stability)
    rfe_results.update({"perm_mean_auc": rfe_permutation["perm_mean"], "perm_std_auc": rfe_permutation["perm_std"], "perm_p_value": rfe_permutation["p_value"]})
    results_records.append(create_results_record(label_name, "Connectome_RFE", "XGBoost", rfe_results)); roc_records.append(create_roc_record("Connectome_RFE", "XGBoost", rfe_results)); save_pickle(rfe_results, outcome_output_dir / "connectome_rfe_results.pkl")

    LOGGER.info("[%s] 4/5 Global graph metrics -> XGBoost", label_name)
    graph_features, graph_metric_names = graph_global_metrics_from_connectomes(connectomes_subset, density=graph_density, n_randomizations=graph_randomizations, random_state=random_state)
    graph_results = classify_loocv_xgb(graph_features, y, random_state=random_state); graph_results["metric_names"] = graph_metric_names
    graph_permutation = permutation_test_auc(classify_loocv_xgb, graph_features, y, n_permutations=n_permutations, random_state=random_state, n_jobs=n_jobs)
    graph_results.update({"perm_mean_auc": graph_permutation["perm_mean"], "perm_std_auc": graph_permutation["perm_std"], "perm_p_value": graph_permutation["p_value"]})
    results_records.append(create_results_record(label_name, "Graph_global_metrics", "XGBoost", graph_results)); roc_records.append(create_roc_record("Graph_global_metrics", "XGBoost", graph_results)); save_pickle(graph_results, outcome_output_dir / "graph_global_metrics_results.pkl")

    LOGGER.info("[%s] 5/5 Connectome -> CPM", label_name)
    cpm_results, cpm_permutation = permutation_test_cpm(X_edges, y, cpm_p_thresholds, n_permutations=n_permutations, random_state=random_state, n_jobs=n_jobs, consensus_fraction=cpm_consensus)
    for observed, permutation in zip(cpm_results, cpm_permutation):
        observed.update({"perm_mean_auc": permutation["perm_mean"], "perm_std_auc": permutation["perm_std"], "perm_p_value": permutation["p_value"]}); threshold_name = _format_threshold(observed["p_thresh"]); method_name = f"CPM_p{threshold_name}"; results_records.append(create_results_record(label_name, method_name, "Logistic Regression", observed)); roc_records.append(create_roc_record(method_name, "Logistic Regression", observed))
    save_pickle(cpm_results, outcome_output_dir / "cpm_results.pkl"); save_pickle(cpm_permutation, outcome_output_dir / "cpm_permutation_results.pkl")
    pd.DataFrame(results_records).to_excel(outcome_output_dir / "results_table.xlsx", index=False); save_pickle(roc_records, outcome_output_dir / "roc_curves.pkl"); save_pickle({"subject_ids": list(subject_ids), "roi_names": list(uptake_df.columns), "edge_to_roi": edge_to_roi}, outcome_output_dir / "feature_metadata.pkl")
    master_row = {"Label": label_name, "Outcome_column": outcome_column, "N": len(y), "Class_0": int(np.sum(y == 0)), "Class_1": int(np.sum(y == 1))}
    for record in results_records: master_row[f"{record['Method']}_{record['Classifier']}"] = record["ROC-AUC"]
    LOGGER.info("Completed outcome '%s'.", label_name)
    return master_row


def _safe_filename(value):
    """Convert a label into a filesystem-safe name."""
    safe = "".join(character if character.isalnum() or character in ("-", "_", ".") else "_" for character in str(value))
    return safe.strip("_") or "outcome"


def _format_threshold(value):
    """Create a compact threshold representation for filenames."""
    return f"{value:.6g}".replace(".", "p")


def save_configuration(output_dir, args, label_mapping, n_subjects, n_rois):
    """Save the complete analysis configuration."""
    configuration = vars(args).copy(); configuration["labels"] = label_mapping; configuration["n_subjects"] = int(n_subjects); configuration["n_rois"] = int(n_rois); configuration["n_edges"] = int(n_rois * (n_rois - 1) / 2)
    with open(output_dir / "analysis_configuration.json", "w", encoding="utf-8") as file: json.dump(configuration, file, indent=2, default=str)


def run_analysis(args):
    """Run the complete analysis pipeline."""
    p_thresholds = validate_parameters(args); label_mapping = parse_label_mapping(args.labels); output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    uptake_df, labels_df = load_input_data(args.input_excel, args.uptake_sheet, args.labels_sheet, args.id_column); uptake_df = prepare_uptake_matrix(uptake_df)
    complete_uptake_mask = ~uptake_df.isna().any(axis=1)
    if not complete_uptake_mask.all():
        LOGGER.warning("Removing %d subjects with missing ROI uptake values.", int((~complete_uptake_mask).sum())); uptake_df = uptake_df.loc[complete_uptake_mask]; labels_df = labels_df.loc[labels_df.index.intersection(uptake_df.index)]
    common_subjects = uptake_df.index.intersection(labels_df.index); uptake_df = uptake_df.loc[common_subjects]; labels_df = labels_df.loc[common_subjects]
    if len(common_subjects) < 10:
        raise ValueError("Fewer than 10 complete subjects remain.")
    uptake_matrix = uptake_df.to_numpy(dtype=float); connectomes = build_logratio_connectomes(uptake_matrix); n_rois = uptake_matrix.shape[1]; n_edges = int(n_rois * (n_rois - 1) / 2)
    LOGGER.info("Dataset: %d subjects, %d ROIs, %d unique connectome edges.", len(common_subjects), n_rois, n_edges); save_configuration(output_dir, args, label_mapping, len(common_subjects), n_rois)
    master_results = []; outcome_items = list(label_mapping.items())
    if args.n_jobs == 1:
        for outcome_index, (label_name, outcome_column) in enumerate(outcome_items):
            result = run_single_outcome(label_name=label_name, outcome_column=outcome_column, uptake_df=uptake_df, connectomes=connectomes, labels_df=labels_df, output_dir=output_dir, graph_density=args.graph_density, graph_randomizations=args.graph_randomizations, pca_variance=args.pca_variance, rfe_features=args.rfe_features, rfe_stability=args.rfe_stability, cpm_p_thresholds=p_thresholds, cpm_consensus=args.cpm_consensus, n_permutations=args.n_permutations, random_state=args.random_state + outcome_index, n_jobs=1, xgb_estimators=args.xgb_estimators, xgb_learning_rate=args.xgb_learning_rate, xgb_max_depth=args.xgb_max_depth)
            if result is not None: master_results.append(result)
    else:
        outcome_results = Parallel(n_jobs=args.n_jobs)(delayed(run_single_outcome)(label_name=label_name, outcome_column=outcome_column, uptake_df=uptake_df, connectomes=connectomes, labels_df=labels_df, output_dir=output_dir, graph_density=args.graph_density, graph_randomizations=args.graph_randomizations, pca_variance=args.pca_variance, rfe_features=args.rfe_features, rfe_stability=args.rfe_stability, cpm_p_thresholds=p_thresholds, cpm_consensus=args.cpm_consensus, n_permutations=args.n_permutations, random_state=args.random_state + outcome_index, n_jobs=1, xgb_estimators=args.xgb_estimators, xgb_learning_rate=args.xgb_learning_rate, xgb_max_depth=args.xgb_max_depth) for outcome_index, (label_name, outcome_column) in enumerate(outcome_items)); master_results = [result for result in outcome_results if result is not None]
    pd.DataFrame(master_results).to_excel(output_dir / "master_summary_table.xlsx", index=False); save_pickle({"subject_ids": list(common_subjects), "roi_names": list(uptake_df.columns), "edge_to_roi": build_edge_to_roi(list(uptake_df.columns))}, output_dir / "dataset_metadata.pkl")
    LOGGER.info("Analysis completed successfully."); LOGGER.info("Results saved to: %s", output_dir.resolve())


def main():
    """Entry point."""
    configure_logging(); warnings.filterwarnings("default", category=RuntimeWarning); args = parse_arguments()
    try:
        run_analysis(args)
    except Exception as exc:
        LOGGER.exception("Analysis failed: %s", exc); raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
