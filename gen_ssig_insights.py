import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import (
    chi2_contingency,
    fisher_exact,
    ttest_ind,
    f_oneway,
    pearsonr
)

# -----------------------
# CONFIGURATION
# -----------------------
P_THRESHOLD = 0.05
MIN_COUNT = 20
INPUT_PATH = "toys_and_games.xlsx"
OUTPUT_PATH = "statistical_insights_segmented_tandg.csv"
CATEGORICAL_THRESHOLD = 15  # Unique value threshold for categoricals
SEGMENT_COLS = ["gender", "age"]  # Define your segment columns

# -----------------------
# TYPE INFERENCE
# -----------------------
def infer_column_types(df):
    types = {}
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if df[col].dtype == object or len(unique_vals) <= CATEGORICAL_THRESHOLD:
            types[col] = "categorical"
        else:
            types[col] = "continuous"
    return types

# -----------------------
# EFFECT SIZE
# -----------------------
def cramers_v(confusion_matrix):
    """
    Calculate Cramér's V with proper handling of edge cases
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.values.sum()
    r, k = confusion_matrix.shape
    
    # Handle edge cases
    if n == 0:
        return 0.0
    
    denominator = min(r-1, k-1)
    if denominator == 0:
        return 0.0  # Perfect association or no variation
    
    # Calculate Cramér's V
    cramers_v_value = np.sqrt(chi2 / (n * denominator))
    
    # Handle potential numerical issues
    if np.isnan(cramers_v_value) or np.isinf(cramers_v_value):
        return 0.0
    
    return cramers_v_value

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1)*np.std(x, ddof=1)**2 + (ny - 1)*np.std(y, ddof=1)**2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std

# -----------------------
# MAIN TEST FUNCTION
# -----------------------
def run_statistical_tests(df, types, segment_label=None):
    results = []

    for col1, col2 in combinations(df.columns, 2):
        if col1 in SEGMENT_COLS or col2 in SEGMENT_COLS:
            continue  # skip segment columns in test pairs

        t1, t2 = types.get(col1), types.get(col2)
        pair_type = f"{t1}_vs_{t2}"

        try:
            p = None
            effect_size = None
            test_name = None

            # Check for "Not Answered" or missing values in both columns
            df_cleaned = df.dropna(subset=[col1, col2])
            df_cleaned = df_cleaned[df_cleaned[col1] != "Not Answered"]
            df_cleaned = df_cleaned[df_cleaned[col2] != "Not Answered"]

            if df_cleaned[col1].empty or df_cleaned[col2].empty:
                continue  # skip if cleaned data is empty after filtering

            if pair_type == "categorical_vs_categorical":
                table = pd.crosstab(df_cleaned[col1], df_cleaned[col2])
                if table.values.sum() < MIN_COUNT or table.shape[0] < 2 or table.shape[1] < 2:
                    continue
                if (table == 0).all().any() or (table == 0).any().all():
                    continue

                if table.shape == (2, 2):
                    _, p = fisher_exact(table)
                    test_name = "Fisher's Exact"
                else:
                    _, p, _, _ = chi2_contingency(table)
                    test_name = "Chi-square"

                effect_size = cramers_v(table)

            elif pair_type in ["categorical_vs_continuous", "continuous_vs_categorical"]:
                # Ensure categorical is first
                if pair_type == "continuous_vs_categorical":
                    col1, col2 = col2, col1  # flip
                groups = df_cleaned.groupby(col1)[col2].apply(list)

                if len(groups) == 2:
                    g1, g2 = groups.iloc[0], groups.iloc[1]
                    if len(g1) < MIN_COUNT or len(g2) < MIN_COUNT:
                        continue
                    _, p = ttest_ind(g1, g2)
                    test_name = "T-test"
                    effect_size = cohens_d(g1, g2)
                elif len(groups) > 2:
                    valid_groups = [g for g in groups if len(g) >= MIN_COUNT]
                    if len(valid_groups) < 2:
                        continue
                    _, p = f_oneway(*valid_groups)
                    test_name = "ANOVA"

            elif pair_type == "continuous_vs_continuous":
                x = df_cleaned[col1]
                y = df_cleaned[col2]
                if len(x) < MIN_COUNT or len(y) < MIN_COUNT:
                    continue
                _, p = pearsonr(x, y)
                test_name = "Pearson Correlation"

            # Skip if no p-value or not significant
            if p is None or p >= P_THRESHOLD:
                continue

            # Get most common values for insight text
            value1 = df_cleaned[col1].mode().iloc[0] if not df_cleaned[col1].mode().empty else "N/A"
            value2 = df_cleaned[col2].mode().iloc[0] if not df_cleaned[col2].mode().empty else "N/A"

            q1_clean = col1.replace("_", " ").capitalize()
            q2_clean = col2.replace("_", " ").capitalize()

            # Format q2 label
            if q2_clean.lower().endswith("y"):
                q2_label = q2_clean[:-1] + "ies"
            else:
                q2_label = q2_clean + "s"

            # Format likelihood text from effect size
            if effect_size is not None:
                if effect_size >= 1:
                    likelihood_pct = round(effect_size * 100)
                    likelihood_phrase = f"{likelihood_pct}% more likely"
                else:
                    likelihood_pct = round((1 - effect_size) * 100)
                    likelihood_phrase = f"{likelihood_pct}% less likely"
            else:
                likelihood_phrase = "more likely"

            # Segment caveat (you can change this logic later)
            valid_in_all_segments = True
            insight_text = (
                f"{value1} fans are {likelihood_phrase} to say '{value2}' is their favourite when it comes to {q2_label.lower()}."
            )
            if not valid_in_all_segments:
                insight_text += " This insight may not hold across all age or gender groups."

            results.append({
                "question1": col1,
                "value1": value1,
                "question2": col2,
                "value2": value2,
                "p_value": p,
                "effect_size": effect_size,
                "test": test_name,
                "segment": segment_label or "ALL",
                "segment_valid": valid_in_all_segments,
                "insight": insight_text
            })

        except Exception as e:
            continue

    return results


# -----------------------
# SEGMENTED WRAPPER
# -----------------------
def run_with_segments(df):
    all_results = []

    # Type inference from full dataset
    types = infer_column_types(df)

    # Global (all data)
    all_results += run_statistical_tests(df, types, segment_label="ALL")

    # Segmented results
    for col in SEGMENT_COLS:
        if col not in df.columns:
            continue
        for val in df[col].dropna().unique():
            subset = df[df[col] == val]
            seg_label = f"{col}={val}"
            all_results += run_statistical_tests(subset, types, segment_label=seg_label)

    return pd.DataFrame(all_results)

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    df = pd.read_excel(INPUT_PATH)

    # Ensure numeric where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    # Optional: create age_group column if only age exists
    if "age" in df.columns and "age_group" not in df.columns:
        bins = [0, 5, 9, 12, 15, 18]
        labels = ["3-5", "6-9", "10-12", "13-15", "16-18"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

    results_df = run_with_segments(df)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ {len(results_df)} significant results saved to {OUTPUT_PATH}")
