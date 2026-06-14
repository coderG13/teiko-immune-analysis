"""
analysis.py - Runs all analytical tasks (Parts 2-4) and saves outputs.
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

DB_PATH = "teiko.db"
OUTPUT_DIR = "outputs"
POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]


def get_connection() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"{DB_PATH} not found. Run load_data.py first.")
    return sqlite3.connect(DB_PATH)


# ─────────────────────────────────────────────
# Part 2: Frequency Table
# ─────────────────────────────────────────────

def compute_frequency_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    For each sample, compute the relative frequency (%) of each cell population.
    Returns a long-format DataFrame with columns:
      sample, total_count, population, count, percentage
    """
    query = """
        SELECT
            s.sample_id  AS sample,
            cc.b_cell, cc.cd8_t_cell, cc.cd4_t_cell, cc.nk_cell, cc.monocyte
        FROM samples s
        JOIN cell_counts cc ON cc.sample_id = s.sample_id
    """
    df = pd.read_sql_query(query, conn)
    df["total_count"] = df[POPULATIONS].sum(axis=1)

    rows = []
    for _, row in df.iterrows():
        for pop in POPULATIONS:
            rows.append({
                "sample":      row["sample"],
                "total_count": int(row["total_count"]),
                "population":  pop,
                "count":       int(row[pop]),
                "percentage":  round(row[pop] / row["total_count"] * 100, 4),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Part 3: Statistical Analysis
# ─────────────────────────────────────────────

def get_melanoma_miraclib_pbmc(conn: sqlite3.Connection, freq_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter frequency table to melanoma + miraclib + PBMC samples only.
    Merges in response and sample_type from the database.
    """
    query = """
        SELECT s.sample_id AS sample, sub.response
        FROM samples s
        JOIN subjects sub ON sub.subject_id = s.subject_id
        WHERE sub.condition  = 'melanoma'
          AND sub.treatment  = 'miraclib'
          AND s.sample_type  = 'PBMC'
          AND sub.response  IS NOT NULL
    """
    meta = pd.read_sql_query(query, conn)
    merged = freq_df.merge(meta, on="sample", how="inner")
    return merged


def run_statistical_analysis(conn: sqlite3.Connection, freq_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mann-Whitney U test per population comparing responders vs non-responders.
    Returns a summary DataFrame.
    """
    df = get_melanoma_miraclib_pbmc(conn, freq_df)
    results = []
    for pop in POPULATIONS:
        pop_df = df[df["population"] == pop]
        resp    = pop_df[pop_df["response"] == "yes"]["percentage"].values
        nonresp = pop_df[pop_df["response"] == "no"]["percentage"].values
        u_stat, p_val = stats.mannwhitneyu(resp, nonresp, alternative="two-sided")
        results.append({
            "population":        pop,
            "n_responders":      len(resp),
            "n_non_responders":  len(nonresp),
            "median_responders": round(float(np.median(resp)), 4),
            "median_non_resp":   round(float(np.median(nonresp)), 4),
            "U_statistic":       round(float(u_stat), 4),
            "p_value":           round(float(p_val), 6),
            "significant":       p_val < 0.05,
        })
    return pd.DataFrame(results).sort_values("p_value")


def plot_boxplots(conn: sqlite3.Connection, freq_df: pd.DataFrame, out_dir: str) -> str:
    """
    Boxplot of relative frequencies per cell population: responders vs non-responders.
    """
    df = get_melanoma_miraclib_pbmc(conn, freq_df)
    stat_df = run_statistical_analysis(conn, freq_df)

    fig, axes = plt.subplots(1, 5, figsize=(18, 6))
    fig.suptitle(
        "Cell Population Frequencies: Responders vs Non-Responders\n"
        "(Melanoma patients, miraclib treatment, PBMC samples)",
        fontsize=13, fontweight="bold", y=1.02
    )

    colors = {"yes": "#4C9BE8", "no": "#E8754C"}
    labels = {"yes": "Responders", "no": "Non-Responders"}

    for ax, pop in zip(axes, POPULATIONS):
        pop_df = df[df["population"] == pop]
        data_resp    = pop_df[pop_df["response"] == "yes"]["percentage"].values
        data_nonresp = pop_df[pop_df["response"] == "no"]["percentage"].values

        bp = ax.boxplot(
            [data_resp, data_nonresp],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor(colors["yes"])
        bp["boxes"][1].set_facecolor(colors["no"])

        # Significance annotation
        row = stat_df[stat_df["population"] == pop].iloc[0]
        sig_label = f"p={row['p_value']:.4f}"
        if row["significant"]:
            sig_label += " *"
        ax.set_title(f"{pop}\n{sig_label}", fontsize=10)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Resp.", "Non-Resp."], fontsize=9)
        ax.set_ylabel("Relative Frequency (%)" if pop == POPULATIONS[0] else "")
        ax.grid(axis="y", alpha=0.3)

    patches = [mpatches.Patch(color=colors[k], label=labels[k]) for k in colors]
    fig.legend(handles=patches, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()

    path = os.path.join(out_dir, "boxplot_responders_vs_nonresponders.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# Part 4: Subset Analysis
# ─────────────────────────────────────────────

def run_subset_analysis(conn: sqlite3.Connection) -> dict:
    """
    Melanoma PBMC baseline (t=0) miraclib samples.
    Note: quintazide was considered as an alternative treatment comparator
    but is not present in this dataset; all analyses use miraclib as specified.
    """
    base_query = """
        SELECT
            sub.subject_id,
            sub.project_id,
            sub.sex,
            sub.response,
            cc.b_cell,
            s.sample_id
        FROM samples s
        JOIN subjects sub ON sub.subject_id = s.subject_id
        JOIN cell_counts cc ON cc.sample_id = s.sample_id
        WHERE sub.condition               = 'melanoma'
          AND s.sample_type               = 'PBMC'
          AND s.time_from_treatment_start = 0
          AND sub.treatment               = 'miraclib'
    """
    df = pd.read_sql_query(base_query, conn)

    samples_per_project  = df.groupby("project_id")["sample_id"].count().to_dict()
    subjects_by_response = df.drop_duplicates("subject_id").groupby("response")["subject_id"].count().to_dict()
    subjects_by_sex      = df.drop_duplicates("subject_id").groupby("sex")["subject_id"].count().to_dict()

    # Average B cells: melanoma male responders at t=0
    male_resp = df[(df["sex"] == "M") & (df["response"] == "yes")]
    avg_bcell = round(float(male_resp["b_cell"].mean()), 2) if len(male_resp) > 0 else None

    return {
        "dataframe":             df,
        "samples_per_project":   samples_per_project,
        "subjects_by_response":  subjects_by_response,
        "subjects_by_sex":       subjects_by_sex,
        "avg_bcell_male_resp":   avg_bcell,
        "n_male_resp_samples":   len(male_resp),
    }


# ─────────────────────────────────────────────
# Main pipeline runner
# ─────────────────────────────────────────────

def run_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = get_connection()

    print("=" * 60)
    print("PART 2: Computing frequency table...")
    freq_df = compute_frequency_table(conn)
    freq_path = os.path.join(OUTPUT_DIR, "frequency_table.csv")
    freq_df.to_csv(freq_path, index=False)
    print(f"  Saved {len(freq_df)} rows → {freq_path}")
    print(freq_df.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("PART 3: Statistical analysis (melanoma + miraclib + PBMC)...")
    stat_df = run_statistical_analysis(conn, freq_df)
    stat_path = os.path.join(OUTPUT_DIR, "statistical_results.csv")
    stat_df.to_csv(stat_path, index=False)
    print(stat_df.to_string(index=False))

    sig = stat_df[stat_df["significant"]]
    if len(sig):
        print(f"\n  Significant populations (p < 0.05): {', '.join(sig['population'].tolist())}")
    else:
        print("\n  No populations reached significance at p < 0.05.")

    plot_path = plot_boxplots(conn, freq_df, OUTPUT_DIR)
    print(f"  Boxplot saved → {plot_path}")

    print("\n" + "=" * 60)
    print("PART 4: Subset analysis (melanoma PBMC baseline miraclib)...")
    subset = run_subset_analysis(conn)
    print(f"  Samples per project:   {subset['samples_per_project']}")
    print(f"  Subjects by response:  {subset['subjects_by_response']}")
    print(f"  Subjects by sex:       {subset['subjects_by_sex']}")
    print(f"  Avg B cells (melanoma male responders, t=0): {subset['avg_bcell_male_resp']}")

    subset_path = os.path.join(OUTPUT_DIR, "subset_analysis.txt")
    with open(subset_path, "w") as f:
        f.write("Part 4: Melanoma PBMC Baseline (t=0) Miraclib Samples\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Samples per project:\n")
        for k, v in subset["samples_per_project"].items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nSubjects by response:\n")
        for k, v in subset["subjects_by_response"].items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nSubjects by sex:\n")
        for k, v in subset["subjects_by_sex"].items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nAvg B cells (melanoma male responders, t=0): {subset['avg_bcell_male_resp']}\n")
        f.write(f"  (based on {subset['n_male_resp_samples']} samples)\n")
    print(f"  Subset summary saved → {subset_path}")

    conn.close()
    print("\nAll outputs saved to ./outputs/")


if __name__ == "__main__":
    run_all()
