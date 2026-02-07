import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
try:
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib is optional
    plt = None



DATA_DIR = Path("preprocessed_data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_OUTPUT_DIR = OUTPUT_DIR / "train"
VALIDATION_OUTPUT_DIR = OUTPUT_DIR / "validation"
TEST_OUTPUT_DIR = OUTPUT_DIR / "test"
TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_COLS = {"churn", "outreach", "member_id"}

# c/v (cost per outreach / value)
C_VALUES = [0.005, 0.01, 0.02, 0.03, 0.05]

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

# Output file names
CHURN_METRICS_FILE = "model_evaluation.csv"
VALIDATION_AUUC_CURVE_FILE = "uplift_gain_curve.csv"
VALIDATION_AUUC_VALUE_FILE = "auuc_score.csv"
VALIDATION_OPTIMAL_N_FILE = "optimal_n_by_cost.csv"

def select_numeric_features(train_df, test_df):
    """
    Return common numeric feature columns present in both train and test.

    Notes:
    - Excludes target/treatment/id columns defined in EXCLUDE_COLS.
    - Filters to numeric dtypes only to keep modeling simple and robust.
    """
    common = [c for c in train_df.columns if c not in EXCLUDE_COLS and c in test_df.columns]
    common = [c for c in common if pd.api.types.is_numeric_dtype(train_df[c])]
    return common


def evaluate_cv(models, X, y):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        aucs = []
        for tr_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            m = clone(model)
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, p))
        rows.append({
            "model": name,
            "roc_auc_mean": float(np.mean(aucs)),
            "roc_auc_std": float(np.std(aucs))
        })
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def uplift_two_model(base_model, X_train, y_train, t_train, X_score):
    """
    T-learner:
      p0 = P(churn | no outreach, X)
      p1 = P(churn | outreach, X)
      uplift = p0 - p1  (positive => outreach reduces churn)
    """
    X0, y0 = X_train[t_train == 0], y_train[t_train == 0]
    X1, y1 = X_train[t_train == 1], y_train[t_train == 1]

    if len(X0) == 0 or len(X1) == 0:
        return None

    model0 = clone(base_model)
    model1 = clone(base_model)
    model0.fit(X0, y0)
    model1.fit(X1, y1)

    p0 = model0.predict_proba(X_score)[:, 1]
    p1 = model1.predict_proba(X_score)[:, 1]

    uplift_reduction = p0 - p1
    return p0, p1, uplift_reduction


def uplift_curve(uplift_scores, y_true, t_true, n_bins=100):
    """
    Build a cumulative uplift (gain) curve for AUUC-style evaluation.

    We sort members by predicted uplift (descending) and compute cumulative "gain" as:
        gain(k) = E[Y_treated | counterfactual from control up to k] - Observed(Y_treated up to k)

    Returns: DataFrame with columns: top_n, fraction, gain, random_gain.
    """
    df = pd.DataFrame({
        "uplift": np.asarray(uplift_scores),
        "y": np.asarray(y_true).astype(int),
        "t": np.asarray(t_true).astype(int),
    }).sort_values("uplift", ascending=False).reset_index(drop=True)

    df["cum_t"] = df["t"].cumsum()
    df["cum_c"] = (1 - df["t"]).cumsum()
    df["cum_y_t"] = (df["y"] * df["t"]).cumsum()
    df["cum_y_c"] = (df["y"] * (1 - df["t"])).cumsum()

    control_rate = df["cum_y_c"] / df["cum_c"].replace(0, np.nan)
    gain = df["cum_t"] * control_rate - df["cum_y_t"]
    gain = gain.ffill().fillna(0.0)

    idx = np.linspace(0, len(df) - 1, n_bins).astype(int)
    sampled = df.iloc[idx].copy()
    sampled["top_n"] = sampled.index + 1
    sampled["fraction"] = sampled["top_n"] / len(df)
    sampled["gain"] = gain.iloc[idx].values
    total_gain = gain.iloc[-1]
    sampled["random_gain"] = sampled["fraction"] * total_gain

    return sampled[["top_n", "fraction", "gain", "random_gain"]]


def rank_and_cumsum_uplift(uplift_scores, c_value=None):
    """
    Sort scores descending and compute cumulative sums over the ranked users.
    Then compute cumulative net = cum_uplift - c * rank.
    """
    uplift_arr = np.asarray(uplift_scores, dtype=float)
    if uplift_arr.size == 0:
        raise ValueError("uplift_scores is empty.")

    uplift_arr_sorted = np.sort(uplift_arr)[::-1]
    rank = np.arange(1, uplift_arr_sorted.size + 1)
    cum_uplift = np.cumsum(uplift_arr_sorted)

    out = pd.DataFrame({
        "rank": rank,
        "uplift_reduction": uplift_arr_sorted,
        "cum_uplift_reduction": cum_uplift,
    })
    if c_value is not None:
        out["cum_net"] = cum_uplift - c_value * rank
        out["c"] = c_value
    return out


def calc_optimal_n(uplift_scores, c_value):
    """
    Given uplift u_(i) sorted descending, maximize:
      Net(n) = sum_{i=1..n} u_(i) - c * n
    Returns (best_n, best_net, curve_df).
    """
    ranked = rank_and_cumsum_uplift(uplift_scores, c_value=c_value)
    net_with_zero = np.r_[0.0, ranked["cum_net"].values]
    best_idx = int(np.argmax(net_with_zero))
    best_n = best_idx
    best_net = float(net_with_zero[best_idx])

    curve = pd.DataFrame({
        "n": np.r_[0, ranked["rank"].values],
        "cum_gain": np.r_[0.0, ranked["cum_uplift_reduction"].values],
        "net": net_with_zero,
        "c": c_value
    })
    return best_n, best_net, curve


def find_optimal_n(uplift_scores, c_values):
    rows = []
    for c in c_values:
        best_n, best_net, _ = calc_optimal_n(uplift_scores, c)
        rows.append({"c": c, "best_n": best_n, "best_net": best_net})
    return pd.DataFrame(rows).sort_values("c")



def main():
    np.random.seed(RANDOM_STATE)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    features = select_numeric_features(train, test)
    if not features:
        raise ValueError("No numeric features selected. Check input columns/dtypes and EXCLUDE_COLS.")

    X = train[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = train["churn"].astype(int).values
    t = train["outreach"].astype(int).values

    X_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(0).values

    # Churn model evaluation
    models = {
        "RF": RandomForestClassifier(n_estimators=400, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1),
        "GBM": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
        "Logit": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
            ]
        ),
    }

    metrics = evaluate_cv(models, X, y)
    metrics.to_csv(TRAIN_OUTPUT_DIR / CHURN_METRICS_FILE, index=False)

    # Uplift base model
    uplift_base = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=50,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Uplift validation split (used ONLY for choosing n)
    stratify_combo = train["churn"].astype(str) + "_" + train["outreach"].astype(str)
    min_combo = int(stratify_combo.value_counts().min())
    if min_combo < 2:
        # Fallback: stratify by churn only to avoid train_test_split errors when a combo class is too small.
        stratify_arg = train["churn"].astype(int).values
    else:
        stratify_arg = stratify_combo.values

    X_tr, X_val, y_tr, y_val, t_tr, t_val = train_test_split(
        X, y, t, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_arg
    )
    val = uplift_two_model(uplift_base, X_tr, y_tr, t_tr, X_val)
    if val is None:
        raise RuntimeError("One treatment arm empty — cannot train uplift.")
    _, _, uplift_val = val

    # AUUC evaluation (validation only)
    curve = uplift_curve(uplift_val, y_val, t_val, n_bins=100)
    curve.to_csv(VALIDATION_OUTPUT_DIR / VALIDATION_AUUC_CURVE_FILE, index=False)

    # Plot uplift gain curve vs random baseline (optional dependency)
    if plt is not None:
        plt.figure(figsize=(8, 6))

        plt.plot(curve["fraction"], curve["gain"], label="Uplift model", linewidth=2)
        plt.plot(curve["fraction"], curve["random_gain"], "--", label="Random targeting")

        plt.xlabel("Fraction targeted")
        plt.ylabel("Cumulative gain")
        plt.title("Uplift Gain Curve (Validation)")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(VALIDATION_OUTPUT_DIR / "uplift_gain_curve.png")
        plt.close()
    else:
        print("matplotlib not installed; skipping uplift_gain_curve.png plot.")
    auuc = float(np.trapz(curve["gain"], x=curve["fraction"]))
    pd.DataFrame([{"auuc": auuc}]).to_csv(VALIDATION_OUTPUT_DIR / VALIDATION_AUUC_VALUE_FILE, index=False)
    print(f"AUUC (validation): {auuc:.6f}")

    # Selection of n uses validation uplift (not test uplift) to avoid leakage
    opt_n = find_optimal_n(uplift_val, C_VALUES)
    opt_n.to_csv(VALIDATION_OUTPUT_DIR / VALIDATION_OPTIMAL_N_FILE, index=False)

    rep_c = C_VALUES[1]
    best_n_rep, best_net_rep, curve_rep = calc_optimal_n(uplift_val, rep_c)
    curve_rep.to_csv(VALIDATION_OUTPUT_DIR / f"cumulative_net_curve_c_{rep_c}.csv", index=False)
    rank_and_cumsum_uplift(uplift_val, c_value=rep_c).to_csv(
        VALIDATION_OUTPUT_DIR / f"ranked_uplift_cumsum_c_{rep_c}.csv", index=False
    )

    # Train uplift on FULL TRAIN and score TEST
    full = uplift_two_model(uplift_base, X, y, t, X_test)
    if full is None:
        raise RuntimeError("Full uplift training failed.")
    p0_test, p1_test, uplift_test = full

    out = pd.DataFrame({
        "member_id": test["member_id"].values,
        "p_churn_no_outreach": p0_test,
        "p_churn_with_outreach": p1_test,
        "uplift_reduction": uplift_test,
    }).sort_values("uplift_reduction", ascending=False)

    out["rank"] = np.arange(1, len(out) + 1)
    out["cum_uplift_reduction"] = out["uplift_reduction"].cumsum()
    out[f"cum_net_c_{rep_c}"] = out["cum_uplift_reduction"] - rep_c * out["rank"]

    out.to_csv(TEST_OUTPUT_DIR / "users_ranked_by_predicted_uplift.csv", index=False)
    out.head(best_n_rep).to_csv(TEST_OUTPUT_DIR / f"recommended_targets_topn_c_{rep_c}.csv", index=False)
    out[["rank", "cum_uplift_reduction", f"cum_net_c_{rep_c}"]].to_csv(
        TEST_OUTPUT_DIR / f"cumulative_uplift_and_net_c_{rep_c}.csv", index=False
    )

    print("Saved:")
    print(" -", TRAIN_OUTPUT_DIR / CHURN_METRICS_FILE)
    print(" -", VALIDATION_OUTPUT_DIR / VALIDATION_AUUC_CURVE_FILE)
    print(" -", VALIDATION_OUTPUT_DIR / VALIDATION_AUUC_VALUE_FILE)
    print(" -", VALIDATION_OUTPUT_DIR / VALIDATION_OPTIMAL_N_FILE)
    print(" -", VALIDATION_OUTPUT_DIR / f"cumulative_net_curve_c_{rep_c}.csv")
    print(" -", VALIDATION_OUTPUT_DIR / f"ranked_uplift_cumsum_c_{rep_c}.csv")
    print(" -", TEST_OUTPUT_DIR / "users_ranked_by_predicted_uplift.csv")
    print(" -", TEST_OUTPUT_DIR / f"recommended_targets_topn_c_{rep_c}.csv")
    print(" -", TEST_OUTPUT_DIR / f"cumulative_uplift_and_net_c_{rep_c}.csv")


if __name__ == "__main__":
    main()
