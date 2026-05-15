# ============================
# ML_VangVND_full.py (Balanced Accuracy version)
# Logistic (baseline) + Tree Bagging + Random Forest
# Time split: Train 2020-2023 | Val 2024 | Test 2025
# Tune hyperparams on Val (AUC), choose threshold on Val (Balanced Accuracy),
# Retrain on Train+Val, evaluate on Test.
# Dùng file SCALED — không cần StandardScaler
# ============================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    cohen_kappa_score, balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.patches import Patch


# =========================================================
# 0) PATHS - CHỈ CẦN SỬA 3 DÒNG NÀY
# =========================================================
PATH_TRAIN = r"C:\Users\Admin\Downloads\Khoa luan\train_2020_2023_scaled.csv"
PATH_VAL   = r"C:\Users\Admin\Downloads\Khoa luan\val_2024_scaled.csv"
PATH_TEST  = r"C:\Users\Admin\Downloads\Khoa luan\test_2025_scaled.csv"

OUT_METRICS_CSV = "output_ket_qua_tong_hop.csv"


# =========================================================
# 1) UTILITIES
# =========================================================
def load_data(path):
    df = pd.read_csv(path)
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        except Exception:
            pass
    return df


def build_features(df, feature_cols, y_col="y_up_5d"):
    X = df[feature_cols].copy()
    y = df[y_col].astype(int).values
    return X, y


def pick_threshold(y_true, y_prob, metric="balanced_acc"):
    """
    Chọn threshold tối ưu dựa trên Validation.
    metric: 'balanced_acc' hoặc 'f1'
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_thr, best_score = 0.5, -1
    rows = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_acc":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError("metric must be 'f1' or 'balanced_acc'")

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) else 0
        rec  = tp / (tp + fn) if (tp + fn) else 0
        rows.append([thr, score, rec, spec])

        if score > best_score:
            best_score, best_thr = score, thr

    thr_df = pd.DataFrame(rows, columns=["threshold", metric, "recall", "specificity"])
    return best_thr, best_score, thr_df


def evaluate(model, X, y, name, threshold=0.5):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    rec         = recall_score(y, y_pred, zero_division=0)
    bal_acc     = 0.5 * (rec + specificity)
    fpr, tpr, _ = roc_curve(y, y_prob)

    metrics = {
        "Mo hinh"     : name,
        "Threshold"   : float(np.round(threshold, 3)),
        "Accuracy"    : float(np.round(accuracy_score(y, y_pred), 4)),
        "Precision"   : float(np.round(precision_score(y, y_pred, zero_division=0), 4)),
        "Recall"      : float(np.round(rec, 4)),
        "F1-Score"    : float(np.round(f1_score(y, y_pred, zero_division=0), 4)),
        "AUC-ROC"     : float(np.round(roc_auc_score(y, y_prob), 4)),
        "Specificity" : float(np.round(specificity, 4)),
        "Bal_Accuracy": float(np.round(bal_acc, 4)),
        "Kappa"       : float(np.round(cohen_kappa_score(y, y_pred), 4)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

    return {
        "name"     : name,
        "threshold": threshold,
        "y_prob"   : y_prob,
        "y_pred"   : y_pred,
        "cm"       : cm,
        "fpr"      : fpr,
        "tpr"      : tpr,
        "metrics"  : metrics
    }


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, len(results), figsize=(16, 5))
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm   = res["cm"]
        name = res["name"]
        acc  = res["metrics"]["Accuracy"]
        thr  = res["metrics"]["Threshold"]

        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{name}\nAcc={acc:.3f} | thr={thr:.3f}", fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["0 (Giảm)", "1 (Tăng)"])
        ax.set_yticklabels(["0 (Giảm)", "1 (Tăng)"])
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")

        for (i, j), v in np.ndenumerate(cm):
            ax.text(
                j, i, f"{v}\n({cm_pct[i,j]:.1f}%)",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if cm_pct[i,j] > 50 else "black"
            )

    plt.suptitle("Confusion Matrix — Tập Test 2025", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plot_confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("✔ Đã lưu: plot_confusion_matrix.png")


def plot_roc(results):
    colors = ["tab:blue", "tab:purple", "tab:green"]
    plt.figure(figsize=(8, 6))
    for res, color in zip(results, colors):
        auc = res["metrics"]["AUC-ROC"]
        plt.plot(res["fpr"], res["tpr"], color=color, lw=2.5,
                 label=f'{res["name"]} (AUC={auc:.3f})')

    plt.plot([0,1],[0,1],"k--", lw=1.2, label="Random (AUC=0.500)")
    plt.title("ROC Curve — So sánh 3 mô hình (Test 2025)", fontweight="bold")
    plt.xlabel("1 - Specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("plot_roc_curve.png", dpi=130)
    plt.show()
    print("✔ Đã lưu: plot_roc_curve.png")


def plot_metrics_bar(metrics_df):
    cols  = ["Accuracy","Precision","Recall","F1-Score",
             "AUC-ROC","Specificity","Bal_Accuracy","Kappa"]
    x     = np.arange(len(cols))
    width = 0.25
    colors = ["tab:blue", "tab:purple", "tab:green"]
    models = metrics_df["Mo hinh"].tolist()
    vals   = [metrics_df.loc[metrics_df["Mo hinh"]==m, cols].values[0] for m in models]

    plt.figure(figsize=(14, 6))
    for i, (m, color) in enumerate(zip(models, colors)):
        plt.bar(x + (i-1)*width, vals[i], width, label=m,
                color=color, alpha=0.85)
        for j, v in enumerate(vals[i]):
            plt.text(x[j] + (i-1)*width, v + 0.01, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.xticks(x, cols, rotation=0)
    plt.ylim(0, 1.12)
    plt.title("So sánh hiệu suất 3 mô hình — Test 2025\n"
              "(Threshold tối ưu hóa trên Val 2024 theo Balanced Accuracy)", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_metrics_comparison.png", dpi=130)
    plt.show()
    print("✔ Đã lưu: plot_metrics_comparison.png")


# =========================================================
# 2) LOAD DATA
# =========================================================
train = load_data(PATH_TRAIN)
val   = load_data(PATH_VAL)
test  = load_data(PATH_TEST)

for name, df in [("Train", train), ("Val", val), ("Test", test)]:
    y = df["y_up_5d"].astype(int)
    print(f"{name}: {df.shape[0]} obs | pos_rate = {y.mean()*100:.1f}%")


# =========================================================
# 3) FEATURE LIST
# =========================================================
exclude = {
    "Date", "year",
    "XAU_USD", "USD_VND", "P_VND_LUONG",
    "SP500", "VIXCLS", "DTWEXBGS", "DCOILWTICO", "DGS10",
    "future_ret_5d", "y_up_5d"
}
feature_cols = [c for c in train.columns if c not in exclude]
print(f"\n✔ Số biến đặc trưng: {len(feature_cols)}")

X_train, y_train = build_features(train, feature_cols)
X_val,   y_val   = build_features(val,   feature_cols)
X_test,  y_test  = build_features(test,  feature_cols)

# Train + Val để retrain final model
X_tv = pd.concat([
    pd.DataFrame(X_train, columns=feature_cols),
    pd.DataFrame(X_val,   columns=feature_cols)
], axis=0).values
y_tv = np.concatenate([y_train, y_val])


# =========================================================
# 3.5) KIỂM ĐỊNH ĐA CỘNG TUYẾN — VIF
# Chạy trên tập Train trước khi huấn luyện mô hình
# VIF < 5: An toàn | 5–10: Cần chú ý | >10: Nghiêm trọng
# =========================================================
print("\n=== KIỂM ĐỊNH ĐA CỘNG TUYẾN (VIF) ===")

X_train_df = pd.DataFrame(X_train, columns=feature_cols)

vif_data = pd.DataFrame()
vif_data["Feature"] = feature_cols
vif_data["VIF"]     = [
    variance_inflation_factor(X_train_df.values, i)
    for i in range(X_train_df.shape[1])
]
vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

def classify_vif(v):
    if v >= 10:   return "❌ Nghiêm trọng (>10)"
    elif v >= 5:  return "⚠️  Cần chú ý (5–10)"
    else:         return "✅ An toàn (<5)"

vif_data["Muc_do"] = vif_data["VIF"].apply(classify_vif)
print(vif_data.to_string(index=False))

n_severe  = (vif_data["VIF"] >= 10).sum()
n_warning = ((vif_data["VIF"] >= 5) & (vif_data["VIF"] < 10)).sum()
n_safe    = (vif_data["VIF"] < 5).sum()
print(f"\n📊 Tóm tắt VIF:")
print(f"  ❌ Nghiêm trọng (VIF ≥ 10) : {n_severe}  biến")
print(f"  ⚠️  Cần chú ý  (5 ≤ VIF < 10): {n_warning} biến")
print(f"  ✅ An toàn     (VIF < 5)    : {n_safe}  biến")

vif_data.to_csv("output_VIF.csv", index=False, encoding="utf-8-sig")
print("✔ Đã lưu: output_VIF.csv")

# Biểu đồ VIF
fig, ax = plt.subplots(figsize=(10, 9))
bar_colors = ["#DC2626" if v >= 10 else "#F59E0B" if v >= 5 else "#22C55E"
              for v in vif_data["VIF"]]
bars = ax.barh(vif_data["Feature"], vif_data["VIF"],
               color=bar_colors, alpha=0.88, height=0.7)
ax.axvline(5,  color="#F59E0B", ls="--", lw=1.5)
ax.axvline(10, color="#DC2626", ls="--", lw=1.5)
for bar, v in zip(bars, vif_data["VIF"]):
    ax.text(v + 0.1, bar.get_y() + bar.get_height()/2,
            f"{v:.2f}", va="center", fontsize=8.5)
ax.set_xlabel("VIF (Variance Inflation Factor)")
ax.set_title("Kiểm định Đa cộng tuyến — VIF\nTập Train 2020–2023",
             fontweight="bold", pad=12)
legend_patches = [
    Patch(color="#DC2626", label="Nghiêm trọng (VIF ≥ 10)"),
    Patch(color="#F59E0B", label="Cần chú ý (5 ≤ VIF < 10)"),
    Patch(color="#22C55E", label="An toàn (VIF < 5)"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("plot_VIF.png", dpi=130, bbox_inches="tight")
plt.show()
print("✔ Đã lưu: plot_VIF.png")

# =========================================================
# 4) TUNE HYPERPARAMS ON VAL (AUC)
# =========================================================
print("\n=== HYPERPARAMETER TUNING (Val 2024 làm holdout) ===")

# ---- Logistic Regression
Cs = [1e-3, 1e-2, 1e-1, 1, 10, 100]
best_logit_auc, best_C = -1, None

for C in Cs:
    logit = LogisticRegression(max_iter=3000, solver="lbfgs",
                               random_state=42, C=C)
    logit.fit(X_train, y_train)
    auc = roc_auc_score(y_val, logit.predict_proba(X_val)[:, 1])
    if auc > best_logit_auc:
        best_logit_auc, best_C = auc, C

print(f"✔ Logistic — Best C={best_C} | AUC(val)={best_logit_auc:.4f}")

# ---- Tree Bagging
bag_grid = [(n, ms)
            for n  in [200, 300, 500]
            for ms in [0.7, 0.8, 1.0]]
best_bag_auc, best_bag_params = -1, None

for n_est, ms in bag_grid:
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=n_est, max_samples=ms,
        bootstrap=True, n_jobs=-1, random_state=42
    )
    bag.fit(X_train, y_train)
    auc = roc_auc_score(y_val, bag.predict_proba(X_val)[:, 1])
    if auc > best_bag_auc:
        best_bag_auc    = auc
        best_bag_params = {"n_estimators": n_est, "max_samples": ms}

print(f"✔ Bagging  — Best {best_bag_params} | AUC(val)={best_bag_auc:.4f}")

# ---- Random Forest
rf_grid = [(md, mf)
           for md in [5, 10]
           for mf in [3, 5, 7, "sqrt"]]
best_rf_auc, best_rf_params = -1, None

for md, mf in rf_grid:
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=md, max_features=mf,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    if auc > best_rf_auc:
        best_rf_auc    = auc
        best_rf_params = {"n_estimators": 200, "max_depth": md, "max_features": mf}

print(f"✔ RF       — Best {best_rf_params} | AUC(val)={best_rf_auc:.4f}")


# =========================================================
# 5) CHỌN THRESHOLD TỐI ƯU TRÊN VAL (Balanced Accuracy)
# =========================================================
THR_METRIC = "balanced_acc"   # ✅ CHỐT theo Balanced Accuracy

print(f"\n=== CHỌN THRESHOLD TỐI ƯU TRÊN VAL ({THR_METRIC}) ===")

# Fit trên TRAIN → predict Val → chọn threshold
logit_for_thr = LogisticRegression(max_iter=3000, solver="lbfgs",
                                   random_state=42, C=best_C)
logit_for_thr.fit(X_train, y_train)
val_prob_logit = logit_for_thr.predict_proba(X_val)[:, 1]
thr_logit, score_logit, _ = pick_threshold(y_val, val_prob_logit, metric=THR_METRIC)

bag_for_thr = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=best_bag_params["n_estimators"],
    max_samples=best_bag_params["max_samples"],
    bootstrap=True, n_jobs=-1, random_state=42
)
bag_for_thr.fit(X_train, y_train)
val_prob_bag = bag_for_thr.predict_proba(X_val)[:, 1]
thr_bag, score_bag, _ = pick_threshold(y_val, val_prob_bag, metric=THR_METRIC)

rf_for_thr = RandomForestClassifier(
    n_estimators=best_rf_params["n_estimators"],
    max_depth=best_rf_params["max_depth"],
    max_features=best_rf_params["max_features"],
    random_state=42, n_jobs=-1
)
rf_for_thr.fit(X_train, y_train)
val_prob_rf = rf_for_thr.predict_proba(X_val)[:, 1]
thr_rf, score_rf, _ = pick_threshold(y_val, val_prob_rf, metric=THR_METRIC)

print(f"  Logistic : threshold={thr_logit:.3f} | {THR_METRIC}={score_logit:.3f}")
print(f"  Bagging  : threshold={thr_bag:.3f}   | {THR_METRIC}={score_bag:.3f}")
print(f"  RF       : threshold={thr_rf:.3f}    | {THR_METRIC}={score_rf:.3f}")


# =========================================================
# 6) RETRAIN FINAL MODELS TRÊN TRAIN+VAL (2020–2024)
# =========================================================
print("\n=== RETRAIN TRÊN TRAIN+VAL (2020–2024) ===")

final_logit = LogisticRegression(max_iter=3000, solver="lbfgs",
                                 random_state=42, C=best_C)
final_bag   = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=best_bag_params["n_estimators"],
    max_samples=best_bag_params["max_samples"],
    bootstrap=True, n_jobs=-1, random_state=42
)
final_rf    = RandomForestClassifier(
    n_estimators=best_rf_params["n_estimators"],
    max_depth=best_rf_params["max_depth"],
    max_features=best_rf_params["max_features"],
    random_state=42, n_jobs=-1
)

final_logit.fit(X_tv, y_tv)
final_bag.fit(X_tv, y_tv)
final_rf.fit(X_tv, y_tv)
print("✔ Retrain hoàn tất")


# =========================================================
# 7) ĐÁNH GIÁ TRÊN TEST 2025 VỚI THRESHOLD TỐI ƯU
# =========================================================
print("\n=== ĐÁNH GIÁ TRÊN TẬP TEST 2025 ===")

res_logit = evaluate(final_logit, X_test, y_test, "Logistic Regression", threshold=thr_logit)
res_bag   = evaluate(final_bag,   X_test, y_test, "Tree Bagging",        threshold=thr_bag)
res_rf    = evaluate(final_rf,    X_test, y_test, "Random Forest",       threshold=thr_rf)

results    = [res_logit, res_bag, res_rf]
metrics_df = pd.DataFrame([r["metrics"] for r in results])

display_cols = ["Mo hinh","Threshold","Accuracy","Precision","Recall",
                "F1-Score","AUC-ROC","Specificity","Bal_Accuracy","Kappa",
                "TN","FP","FN","TP"]

print("\n" + "="*80)
print("  BẢNG TỔNG HỢP KẾT QUẢ — TẬP TEST 2025")
print("="*80)
print(metrics_df[display_cols].to_string(index=False))

best_auc = metrics_df.loc[metrics_df["AUC-ROC"].idxmax(),     "Mo hinh"]
best_bal = metrics_df.loc[metrics_df["Bal_Accuracy"].idxmax(), "Mo hinh"]
best_kap = metrics_df.loc[metrics_df["Kappa"].idxmax(),        "Mo hinh"]
print(f"\n🏆 Best AUC-ROC      : {best_auc}")
print(f"🏆 Best Bal_Accuracy : {best_bal}")
print(f"🏆 Best Kappa        : {best_kap}")

metrics_df[display_cols].to_csv(OUT_METRICS_CSV, index=False, encoding="utf-8-sig")
print(f"✔ Đã lưu: {OUT_METRICS_CSV}")


# =========================================================
# 8) BIỂU ĐỒ
# =========================================================
plot_confusion_matrices(results)
plot_roc(results)
plot_metrics_bar(metrics_df)


# =========================================================
# 9) DỰ BÁO THEO THỜI GIAN (Test 2025)
# =========================================================
if "Date" in test.columns:
    dates  = test["Date"].values
    colors = ["tab:blue", "tab:purple", "tab:green"]

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    for ax, res, color in zip(axes, results, colors):
        thr = res["threshold"]
        ax.plot(dates, res["y_prob"], lw=1.5, color=color, label=res["name"])
        ax.axhline(thr, color="gray", ls="--", lw=1,
                   label=f"Threshold = {thr:.3f}")
        ax.fill_between(dates, 0, 1,
                        where=(res["y_prob"] >= thr),
                        alpha=0.08, color="green", label="Dự báo: Tăng")
        ax.fill_between(dates, 0, 1,
                        where=(res["y_prob"] < thr),
                        alpha=0.08, color="red",   label="Dự báo: Giảm")
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(Y=1)")
        ax.set_title(f"{res['name']} (threshold={thr:.3f})", fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, ncol=4)
        ax.spines[["top","right"]].set_visible(False)

    axes[-1].set_xlabel("Date")
    plt.suptitle("Xác suất dự báo xu hướng tăng P(Y=1) — Tập Test 2025",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("plot_du_bao_theo_thoi_gian.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("✔ Đã lưu: plot_du_bao_theo_thoi_gian.png")
