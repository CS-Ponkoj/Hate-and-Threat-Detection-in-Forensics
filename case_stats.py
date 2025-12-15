"""
Case statistics summary for multimodal forensic pipeline

Reads image_evidence.csv and reports:
- total images processed
- routing counts by case_label
- distribution by analysis_mode
"""

import pandas as pd


EVIDENCE_CSV = "image_evidence.csv"


def main():
    df = pd.read_csv(EVIDENCE_CSV)

    if df.empty:
        print("No evidence records found.")
        return

    total = len(df)

    print("\n=== Pipeline Statistics Summary ===\n")
    print(f"Total images processed: {total}\n")

    # -------------------------
    # Case label statistics
    # -------------------------
    print("Routing by case_label:")
    case_counts = df["case_label"].fillna("missing").value_counts()

    for label, count in case_counts.items():
        pct = (count / total) * 100
        print(f"  {label:20s} : {count:3d} ({pct:5.1f}%)")

    # -------------------------
    # Analysis mode statistics
    # -------------------------
    print("\nDistribution by analysis_mode:")
    mode_counts = df["analysis_mode"].fillna("missing").value_counts()

    for mode, count in mode_counts.items():
        pct = (count / total) * 100
        print(f"  {mode:20s} : {count:3d} ({pct:5.1f}%)")

    # -------------------------
    # Sanity checks
    # -------------------------
    incomplete = (df["case_label"] == "incomplete").sum()
    if incomplete > 0:
        print(f"\n⚠ Warning: {incomplete} image(s) marked as incomplete")

    print("\n=== End of Summary ===\n")


if __name__ == "__main__":
    main()
