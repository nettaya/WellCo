from pathlib import Path
import re
import pandas as pd

# Observation window end (used for seniority feature at decision time)
OBS_END_DATE = pd.to_datetime("2025-07-14")

# WellCo engagement fields (from the brief)
FIELD_KEYWORDS = {
    "Physical_Activity": [
        "movement", "physical activity", "exercise", "aerobic", "strength training",
        "strength", "cardio", "cardiovascular", "fitness", "workout", "endurance",
    ],
    "Sleep_Health": [
        "sleep", "sleep hygiene", "restorative sleep", "sleep quality",
        "apnea", "sleep apnea", "insomnia",
    ],
    "Resilience_Wellbeing": [
        "resilience", "wellbeing", "wellness", "stress", "stress management",
        "mindfulness", "meditation", "mental health", "anxiety",
    ],
}

CRITICAL_ICD_CODES = {
    'flag_diabetes': 'E11.9',
    'flag_hypertension': 'I10',
    'flag_dietary_counseling': 'Z71.3'
}

def count_occurrences(series, phrases):
    """Count occurrences of words/phrases in a text series."""
    patterns = []
    for phrase in phrases:
        p = phrase.strip().lower()
        if " " in p:
            pattern = r"\b" + r"\s+".join(map(re.escape, p.split())) + r"\b"
        else:
            pattern = r"\b" + re.escape(p) + r"\b"
        patterns.append(pattern)

    if not patterns:
        return pd.Series(0, index=series.index, dtype=int)

    combined = "(" + "|".join(patterns) + ")"
    return series.str.count(combined, flags=re.IGNORECASE).fillna(0).astype(int)


def process_members(df_members):
    """
    Returns: anchor table (keeps churn/outreach if present)
    """
    df = df_members.copy()
    if "signup_date" in df.columns:
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
        df["membership_days"] = (OBS_END_DATE - df["signup_date"]).dt.days
        df["membership_days"] = (
            df["membership_days"]
            .clip(lower=0)
            .fillna(0)
            .astype(int)
        )
        df = df.drop(columns=["signup_date"], errors="ignore")
    return df


def process_claims(df_claims, member_ids):
    """
    Returns: DataFrame [member_id, flags..., claims_count, claims_unique_icd_count, critical_icd_count]
    """
    df_claims = df_claims.copy()
    df_claims['icd_code'] = df_claims['icd_code'].astype(str).str.upper()

    df_flags = pd.DataFrame({'member_id': member_ids})

    for col_name, icd_code in CRITICAL_ICD_CODES.items():
        sick_set = set(df_claims.loc[df_claims['icd_code'] == icd_code, 'member_id'].unique())
        df_flags[col_name] = df_flags['member_id'].isin(sick_set).astype(int)

    df_counts = df_claims.groupby('member_id').agg(
        claims_count=('icd_code', 'size'),
        claims_unique_icd_count=('icd_code', 'nunique')
    ).reset_index()

    critical_set = set(CRITICAL_ICD_CODES.values())
    df_critical = df_claims[df_claims['icd_code'].isin(critical_set)]
    df_critical_counts = df_critical.groupby('member_id').agg(
        critical_icd_count=('icd_code', 'size')
    ).reset_index()

    df_agg = df_flags.merge(
        df_counts[['member_id', 'claims_count', 'claims_unique_icd_count']],
        on='member_id', how='left'
    )
    df_agg = df_agg.merge(df_critical_counts, on='member_id', how='left')
    df_agg['claims_count'] = df_agg['claims_count'].fillna(0).astype(int)
    df_agg['claims_unique_icd_count'] = df_agg['claims_unique_icd_count'].fillna(0).astype(int)
    df_agg['critical_icd_count'] = df_agg['critical_icd_count'].fillna(0).astype(int)

    return df_agg


def process_web_visits(df_web, member_ids):
    """
    Returns: DataFrame with member-level engagement totals, volume, and diversity.
    """
    df = df_web.copy()

    df["url_text"] = df.get("url", "").fillna("").astype(str)
    df["full_text"] = (
        df.get("title", "").fillna("").astype(str)
        + " "
        + df.get("description", "").fillna("").astype(str)
    )

    field_cols = []
    total_count_cols = []
    for field, phrases in FIELD_KEYWORDS.items():
        total_col = f"{field}_total_count"

        url_count = count_occurrences(df["url_text"], phrases)
        text_count = count_occurrences(df["full_text"], phrases)
        df[total_col] = url_count + text_count

        field_cols.append(total_col)
        total_count_cols.append(total_col)

    # Member-level aggregates
    df_fields = df.groupby("member_id")[field_cols].sum().reset_index()
    df_basic = df.groupby("member_id").agg(web_visits_total=("member_id", "size")).reset_index()
    df_agg = df_fields.merge(df_basic, on="member_id", how="outer")
    df_agg["topic_diversity"] = (df_agg[total_count_cols] > 0).sum(axis=1)

    # Merge with full member list and fill defaults
    df_final = pd.DataFrame({"member_id": member_ids}).merge(df_agg, on="member_id", how="left")
    fill_zero_cols = field_cols + ["web_visits_total", "topic_diversity"]
    for col in fill_zero_cols:
        df_final[col] = df_final[col].fillna(0).astype(int)

    return df_final


def process_app_usage(df_app, member_ids):
    """
    Returns: DataFrame [member_id, total_sessions, avg_hours_between_sessions,
    avg_sessions_per_user, highly_active_user]
    """
    df = df_app.copy()
    df['timestamp'] = pd.to_datetime(df.get('timestamp'), errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # Total sessions
    df_sessions = df.groupby('member_id').agg(
        total_sessions=('timestamp', 'size'),
        active_days=('timestamp', lambda s: s.dt.date.nunique()),
    ).reset_index()
    df_sessions['avg_sessions_per_user'] = (
        df_sessions['total_sessions'] / df_sessions['active_days'].replace(0, pd.NA)
    )

    # Average time between sessions (in hours)
    df_sorted = df.sort_values(['member_id', 'timestamp'])
    df_sorted['prev_ts'] = df_sorted.groupby('member_id')['timestamp'].shift(1)
    df_sorted['gap_hours'] = (df_sorted['timestamp'] - df_sorted['prev_ts']).dt.total_seconds() / 3600.0
    df_gaps = df_sorted.groupby('member_id')['gap_hours'].mean().reset_index()

    df_agg = df_sessions.merge(df_gaps, on='member_id', how='left')
    df_agg = df_agg.rename(columns={'gap_hours': 'avg_hours_between_sessions'})

    # Merge with all members
    df_final = pd.DataFrame({'member_id': member_ids}).merge(
        df_agg[['member_id', 'total_sessions', 'avg_hours_between_sessions', 'avg_sessions_per_user']],
        on='member_id', how='left'
    )
    df_final['total_sessions'] = df_final['total_sessions'].fillna(0).astype(int)
    df_final['avg_sessions_per_user'] = df_final['avg_sessions_per_user'].fillna(0.0).astype(float)

    # Binary flag: 1 if member's average sessions/day is above the population average.
    population_avg_sessions = float(df_final['avg_sessions_per_user'].mean())
    df_final['highly_active_user'] = (
        df_final['avg_sessions_per_user'] > population_avg_sessions
    ).astype(int)

    df_final.loc[df_final['total_sessions'] < 2, 'avg_hours_between_sessions'] = -1
    df_final['avg_hours_between_sessions'] = df_final['avg_hours_between_sessions'].fillna(-1)

    return df_final


def build_master_table(df_members, df_claims, df_web, df_app):
    """
    Returns: Final Clean DataFrame (Merged Features)
    """
    unique_members = df_members['member_id'].unique()

    df_clean_members = process_members(df_members)
    df_clean_claims = process_claims(df_claims, unique_members)
    df_clean_web = process_web_visits(df_web, unique_members)
    df_clean_app = process_app_usage(df_app, unique_members)

    master_df = df_clean_members.merge(df_clean_claims, on='member_id', how='left')
    master_df = master_df.merge(df_clean_web, on='member_id', how='left')
    master_df = master_df.merge(df_clean_app, on='member_id', how='left')

    return master_df

if __name__ == "__main__":
    train_dir = Path("raw_data/train")
    test_dir = Path("raw_data/test")
    out_dir = Path("preprocessed_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # TRAIN
    df_churn_raw = pd.read_csv(train_dir / "churn_labels.csv")
    df_claims_raw = pd.read_csv(train_dir / "claims.csv")
    df_web_raw = pd.read_csv(train_dir / "web_visits.csv")
    df_app_raw = pd.read_csv(train_dir / "app_usage.csv")

    # TEST
    df_test_raw = pd.read_csv(test_dir / "test_members.csv")
    df_test_claims_raw = pd.read_csv(test_dir / "test_claims.csv")
    df_test_web_raw = pd.read_csv(test_dir / "test_web_visits.csv")
    df_test_app_raw = pd.read_csv(test_dir / "test_app_usage.csv")

    # Build & Save
    train_master = build_master_table(df_churn_raw, df_claims_raw, df_web_raw, df_app_raw)
    train_master.to_csv(out_dir / "train.csv", index=False)

    test_master = build_master_table(df_test_raw, df_test_claims_raw, df_test_web_raw, df_test_app_raw)
    test_master.to_csv(out_dir / "test.csv", index=False)
