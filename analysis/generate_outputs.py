import json
import os
import pandas as pd
from typing import List, Dict

def aggregate_results(analysis_dir: str) -> pd.DataFrame:
    rows = []
    for filename in os.listdir(analysis_dir):
        if filename.endswith("_analysis.json"):
            path = os.path.join(analysis_dir, filename)
            variant_name = filename.replace("_dev_analysis.json", "").replace("_analysis.json", "")

            with open(path, 'r') as f:
                data = json.load(f)

            stats = data['stats']
            row = {
                "Variant": variant_name,
                "TP": stats['total_tp'],
                "FP": stats['total_fp'],
                "FN": stats['total_fn'],
                "Precision": stats['total_tp'] / (stats['total_tp'] + stats['total_fp']) if (stats['total_tp'] + stats['total_fp']) > 0 else 0,
                "Recall": stats['total_tp'] / (stats['total_tp'] + stats['total_fn']) if (stats['total_tp'] + stats['total_fn']) > 0 else 0,
            }
            row["F1"] = 2 * (row["Precision"] * row["Recall"]) / (row["Precision"] + row["Recall"]) if (row["Precision"] + row["Recall"]) > 0 else 0

            # Categories
            for cat, count in stats['categories'].items():
                row[f"Err_{cat}"] = count

            rows.append(row)

    return pd.DataFrame(rows).sort_values("F1", ascending=False)

def generate_markdown_table(df: pd.DataFrame) -> str:
    # Select key columns
    cols = ["Variant", "Precision", "Recall", "F1", "Err_missed_concept", "Err_over_extraction", "Err_value_mismatch", "Err_pure_hallucination"]
    # Ensure columns exist (fill with 0 if missing)
    for c in cols:
        if c not in df.columns: df[c] = 0

    sub_df = df[cols].copy()
    sub_df["Precision"] = sub_df["Precision"].map(lambda x: f"{x:.4f}")
    sub_df["Recall"] = sub_df["Recall"].map(lambda x: f"{x:.4f}")
    sub_df["F1"] = sub_df["F1"].map(lambda x: f"{x:.4f}")

    return sub_df.to_markdown(index=False)

def generate_latex_table(df: pd.DataFrame) -> str:
    # Similar to markdown but LaTeX format
    cols = ["Variant", "Precision", "Recall", "F1"]
    sub_df = df[cols].copy()
    return sub_df.to_latex(index=False, float_format="%.4f")

def main():
    analysis_dir = "outputs/results_ablations"
    df = aggregate_results(analysis_dir)

    # Save to CSV
    df.to_csv(os.path.join(analysis_dir, "aggregate_stats.csv"), index=False)

    # Generate report
    report = "# MEDIQA-SYNUR Ablation Results\n\n"
    report += "## Performance Metrics\n\n"
    report += generate_markdown_table(df)
    report += "\n\n## LaTeX Table Source\n\n"
    report += "```latex\n"
    report += generate_latex_table(df)
    report += "\n```\n"

    with open(os.path.join(analysis_dir, "summary_report.md"), 'w') as f:
        f.write(report)

    print(f"Summary report generated at {os.path.join(analysis_dir, 'summary_report.md')}")

if __name__ == "__main__":
    main()
