
ML_VangVND_pipeline.py

import os
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    cohen_kappa_score, balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION & CONSTANTS
# =========================================================
DATA_DIR = "data"
PATH_TRAIN = os.path.join(DATA_DIR, "train_2020_2023_scaled.csv")
PATH_VAL   = os.path.join(DATA_DIR, "val_2024_scaled.csv")
PATH_TEST  = os.path.join(DATA_DIR, "test_2025_scaled.csv")

OUT_METRICS_CSV = "output_ket_qua_tong_hop.csv"
OUT_VIF_CSV = "output_VIF.csv"
RANDOM_STATE = 42
TARGET_COL = "y_up_5d"

EXCLUDE_COLS = {
    "Date", "year", "XAU_USD", "USD_VND", "P_VND_LUONG",
    "SP500", "VIXCLS", "DTWEXBGS", "DCOILWTICO", "DGS10",
    "future_ret_5d", TARGET_COL
}

# DATA UTILITIES
def load_dataset(path: str) -> pd.DataFrame:
    """Đọc dữ liệu và chuẩn hóa định dạng cột ngày tháng."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {path}")
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df


def extract_features_and_target(
    df: pd.DataFrame, feature_cols: List[str], y_col: str = TARGET_COL
) -> Tuple[np.ndarray, np.ndarray]:
    """Tách ma trận đặc trưng X và nhãn mục tiêu y."""
    X = df[feature_cols].copy().values
    y = df[y_col].astype(int).values
    return X, y


# DIAGNOSTICS & THRESHOLD OPTIMIZATION

def run_vif_analysis(X_train_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Kiểm định đa cộng tuyến (VIF) trên tập huấn luyện."""
    vif_records = [
        variance_inflation_factor(X_train_df.values, i)
        for i in range(X_train_df.shape[1])
    ]
    vif_df = pd.DataFrame({"Feature": feature_cols, "VIF": vif_records})
    vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

    def classify_vif(v: float) -> str:
        if v >= 10:
            return "Nghiêm trọng (>=10)"
        elif v >= 5:
            return "Cần chú ý (5-10)"
        return "An toàn (<5)"

    vif_df["Muc_do"] = vif_df["VIF"].apply(classify_vif)
    vif_df.to_csv(OUT_VIF_CSV, index=False, encoding="utf-8-sig")
    return vif_df


def optimize_classification_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str = "balanced_acc"
) -> Tuple[float, float]:
    """Tìm ngưỡng cắt xác suất tối ưu trên tập Validation."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_thr, best_score = 0.5, -1.0

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_acc":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError("Metric phải là 'f1' hoặc 'balanced_acc'")

        if score > best_score:
            best_score, best_thr = score, thr

    return best_thr, best_score


def evaluate_model_performance(
    model: Any, X: np.ndarray, y: np.ndarray, name: str, threshold: float = 0.5
) -> Dict[str, Any]:
    """Đánh giá toàn diện các chỉ số phân loại ngoài mẫu."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec = recall_score(y, y_pred, zero_division=0)
    bal_acc = 0.5 * (rec + specificity)
    fpr, tpr, _ = roc_curve(y, y_prob)

    metrics = {
        "Mo hinh": name,
        "Threshold": round(float(threshold), 3),
        "Accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "Precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "Recall": round(float(rec), 4),
        "F1-Score": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "AUC-ROC": round(float(roc_auc_score(y, y_prob)), 4),
        "Specificity": round(float(specificity), 4),
        "Bal_Accuracy": round(float(bal_acc), 4),
        "Kappa": round(float(cohen_kappa_score(y, y_pred)), 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

    return {
        "name": name,
        "threshold": threshold,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "metrics": metrics
    }

# VISUALIZATIONS

def plot_evaluation_charts(results: List[Dict[str, Any]], metrics_df: pd.DataFrame) -> None:
    """Tạo biểu đồ ma trận nhầm lẫn, đường cong ROC và đối soát chỉ số."""
    # 1. Confusion Matrix
    fig, axes = plt.subplots(1, len(results), figsize=(16, 5))
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = res["cm"]
        name = res["name"]
        acc = res["metrics"]["Accuracy"]
        thr = res["metrics"]["Threshold"]
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{name}\nAcc={acc:.3f} | Thr={thr:.3f}", fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0 (Giảm)", "1 (Tăng)"])
        ax.set_yticklabels(["0 (Giảm)", "1 (Tăng)"])
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")

        for (i, j), v in np.ndenumerate(cm):
            ax.text(
                j, i, f"{v}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white" if cm_pct[i, j] > 50 else "black"
            )

    plt.suptitle("Confusion Matrix — Test Set 2025", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plot_confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.close()

    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    colors = ["tab:blue", "tab:purple", "tab:green"]
    for res, color in zip(results, colors):
        auc = res["metrics"]["AUC-ROC"]
        plt.plot(res["fpr"], res["tpr"], color=color, lw=2.2, label=f"{res['name']} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.500)")
    plt.title("ROC Curve — So sánh mô hình (Test 2025)", fontweight="bold")
    plt.xlabel("1 - Specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("plot_roc_curve.png", dpi=130)
    plt.close()

    # 3. Bar Metrics
    cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "Bal_Accuracy"]
    x = np.arange(len(cols))
    width = 0.25

    plt.figure(figsize=(12, 5))
    for i, (res, color) in enumerate(zip(results, colors)):
        model_name = res["name"]
        vals = metrics_df.loc[metrics_df["Mo hinh"] == model_name, cols].values[0]
        plt.bar(x + (i - 1) * width, vals, width, label=model_name, color=color, alpha=0.85)

    plt.xticks(x, cols)
    plt.ylim(0, 1.1)
    plt.title("So sánh hiệu suất đa chỉ tiêu — Test Set 2025", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_metrics_comparison.png", dpi=130)
    plt.close()


# MAIN EXECUTION PIPELINE

def main():
    print("=== BẮT ĐẦU PIPELINE MACHINE LEARNING DỰ BÁO GIÁ VÀNG ===")

    # 1. Nạp dữ liệu
    train_df = load_dataset(PATH_TRAIN)
    val_df   = load_dataset(PATH_VAL)
    test_df  = load_dataset(PATH_TEST)

    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]
    print(f"✔ Số biến đặc trưng sử dụng: {len(feature_cols)}")

    X_train, y_train = extract_features_and_target(train_df, feature_cols)
    X_val, y_val     = extract_features_and_target(val_df, feature_cols)
    X_test, y_test   = extract_features_and_target(test_df, feature_cols)

    # Dữ liệu gộp Train + Val để huấn luyện mô hình cuối
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    # 2. Kiểm định VIF
    X_train_df = pd.DataFrame(X_train, columns=feature_cols)
    run_vif_analysis(X_train_df, feature_cols)
    print("✔ Đã hoàn thành kiểm định đa cộng tuyến VIF.")

    # 3. Tinh chỉnh siêu tham số trên Validation (Grid Search AUC)
    print("\n=== TUNING HYPERPARAMETERS (Validation Set 2024) ===")

    # Logistic
    best_logit_auc, best_c = -1.0, 1.0
    for c in [1e-3, 1e-2, 1e-1, 1, 10, 100]:
        clf = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE, C=c)
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        if auc > best_logit_auc:
            best_logit_auc, best_c = auc, c

    # Tree Bagging
    best_bag_auc, best_bag_params = -1.0, {}
    for n_est in [200, 300, 500]:
        for ms in [0.7, 0.8, 1.0]:
            bag = BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                n_estimators=n_est, max_samples=ms, bootstrap=True,
                n_jobs=-1, random_state=RANDOM_STATE
            )
            bag.fit(X_train, y_train)
            auc = roc_auc_score(y_val, bag.predict_proba(X_val)[:, 1])
            if auc > best_bag_auc:
                best_bag_auc = auc
                best_bag_params = {"n_estimators": n_est, "max_samples": ms}

    # Random Forest
    best_rf_auc, best_rf_params = -1.0, {}
    for md in [5, 10]:
        for mf in [3, 5, 7, "sqrt"]:
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=md, max_features=mf,
                random_state=RANDOM_STATE, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
            if auc > best_rf_auc:
                best_rf_auc = auc
                best_rf_params = {"n_estimators": 200, "max_depth": md, "max_features": mf}

    print(f"✔ Best Logistic: C={best_c} | Val AUC={best_logit_auc:.4f}")
    print(f"✔ Best Bagging : {best_bag_params} | Val AUC={best_bag_auc:.4f}")
    print(f"✔ Best RF      : {best_rf_params} | Val AUC={best_rf_auc:.4f}")

    # 4. Tối ưu hóa ngưỡng phân loại (Balanced Accuracy)
    print("\n=== TỐI ƯU THRESHOLD TRÊN VALIDATION ===")
    m_logit = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE, C=best_c).fit(X_train, y_train)
    m_bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        **best_bag_params, bootstrap=True, n_jobs=-1, random_state=RANDOM_STATE
    ).fit(X_train, y_train)
    m_rf = RandomForestClassifier(**best_rf_params, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)

    thr_logit, _ = optimize_classification_threshold(y_val, m_logit.predict_proba(X_val)[:, 1])
    thr_bag, _   = optimize_classification_threshold(y_val, m_bag.predict_proba(X_val)[:, 1])
    thr_rf, _    = optimize_classification_threshold(y_val, m_rf.predict_proba(X_val)[:, 1])

    print(f"  Threshold tối ưu: Logistic={thr_logit:.3f} | Bagging={thr_bag:.3f} | RF={thr_rf:.3f}")

    # 5. Huấn luyện lại trên Train+Val & Đánh giá trên Test 2025
    print("\n=== HUẤN LUYỆN LẠI & ĐÁNH GIÁ TRÊN TEST 2025 ===")
    final_logit = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE, C=best_c).fit(X_tv, y_tv)
    final_bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        **best_bag_params, bootstrap=True, n_jobs=-1, random_state=RANDOM_STATE
    ).fit(X_tv, y_tv)
    final_rf = RandomForestClassifier(**best_rf_params, random_state=RANDOM_STATE, n_jobs=-1).fit(X_tv, y_tv)

    res_logit = evaluate_model_performance(final_logit, X_test, y_test, "Logistic Regression", threshold=thr_logit)
    res_bag   = evaluate_model_performance(final_bag, X_test, y_test, "Tree Bagging", threshold=thr_bag)
    res_rf    = evaluate_model_performance(final_rf, X_test, y_test, "Random Forest", threshold=thr_rf)

    results = [res_logit, res_bag, res_rf]
    metrics_df = pd.DataFrame([r["metrics"] for r in results])
    metrics_df.to_csv(OUT_METRICS_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(metrics_df[["Mo hinh", "Threshold", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "Bal_Accuracy"]].to_string(index=False))
    print("=" * 80)

    # 6. Xuất trực quan hóa
    plot_evaluation_charts(results, metrics_df)
    print("✔ Đã xuất toàn bộ biểu đồ và file báo cáo tổng hợp.")


if __name__ == "__main__":
    main()
