import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Fixed paths for the story and output files.
# ---------------------------------------------------------------------------
OUTPUT_GROUP = "output_D"
STORY_ID = "St01_D"
COHORT_JSONL_FILENAME = "incremental_phonemic_cohorts_gpt_St01_D.jsonl"
PHONEME_SURPRISAL_CSV_FILENAME = "phoneme_surprisal_St01_D.csv"

MAX_WORDS_IN_GRID = 30
SAVE_INDIVIDUAL_WORD_PLOTS = True


def load_cohort_metadata(cohort_jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with open(cohort_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            prefix_phonemes = record.get("prefix_phonemes", [])
            if not prefix_phonemes:
                continue

            rows.append({
                "token_id": record["token_id"],
                "target_word": record["target_word"],
                "prefix_length": record["prefix_length"],
                "phoneme": prefix_phonemes[-1],
                "filtered_cohort_size": record["filtered_cohort_size"],
            })

    return pd.DataFrame(rows)


def load_plot_data(cohort_jsonl_path: Path, phoneme_surprisal_csv_path: Path) -> pd.DataFrame:
    metadata = load_cohort_metadata(cohort_jsonl_path)
    surprisal = pd.read_csv(phoneme_surprisal_csv_path)

    if len(metadata) != len(surprisal):
        raise ValueError(
            f"Row mismatch: {cohort_jsonl_path} has {len(metadata)} rows, "
            f"but {phoneme_surprisal_csv_path} has {len(surprisal)} rows."
        )

    data = metadata.copy()
    data["phoneme_surprisal"] = surprisal["phoneme_surprisal"].astype(float)
    data["phoneme_from_csv"] = surprisal["phoneme"].astype(str)

    mismatches = data[data["phoneme"].astype(str) != data["phoneme_from_csv"]]
    if not mismatches.empty:
        raise ValueError(
            "The CSV and JSONL are not aligned: at least one phoneme differs. "
            f"First mismatch: {mismatches.iloc[0].to_dict()}"
        )

    return data.drop(columns=["phoneme_from_csv"])


def plot_overlay(data: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    for _, word_df in data.groupby(["token_id", "target_word"], sort=True):
        word_df = word_df.sort_values("prefix_length")
        ax.plot(
            word_df["prefix_length"],
            word_df["phoneme_surprisal"],
            color="0.65",
            linewidth=1,
            alpha=0.45,
        )

    mean_by_position = (
        data.groupby("prefix_length", as_index=False)["phoneme_surprisal"]
        .mean()
        .sort_values("prefix_length")
    )
    ax.plot(
        mean_by_position["prefix_length"],
        mean_by_position["phoneme_surprisal"],
        color="#006d77",
        linewidth=2.5,
        marker="o",
        label="Mean across words",
    )

    ax.set_title("Phoneme Surprisal Trajectory Across Words")
    ax.set_xlabel("Phoneme position within word")
    ax.set_ylabel("Phoneme surprisal (-log2 probability)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_word_grid(data: pd.DataFrame, output_path: Path, max_words: int = MAX_WORDS_IN_GRID) -> None:
    grouped = list(data.groupby(["token_id", "target_word"], sort=True))[:max_words]
    if not grouped:
        return

    n_cols = 3
    n_rows = (len(grouped) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.8 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, ((token_id, target_word), word_df) in zip(axes_flat, grouped):
        word_df = word_df.sort_values("prefix_length")
        ax.plot(
            word_df["prefix_length"],
            word_df["phoneme_surprisal"],
            marker="o",
            color="#006d77",
            linewidth=2,
        )
        for _, row in word_df.iterrows():
            ax.annotate(
                row["phoneme"],
                (row["prefix_length"], row["phoneme_surprisal"]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )

        ax.set_title(f"{token_id}: {target_word}", fontsize=10)
        ax.set_xlabel("Position")
        ax.set_ylabel("Surprisal")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(grouped):]:
        ax.axis("off")

    fig.suptitle("Phoneme Surprisal by Word", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_individual_words(data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for (token_id, target_word), word_df in data.groupby(["token_id", "target_word"], sort=True):
        word_df = word_df.sort_values("prefix_length")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(
            word_df["prefix_length"],
            word_df["phoneme_surprisal"],
            marker="o",
            color="#006d77",
            linewidth=2,
        )
        for _, row in word_df.iterrows():
            ax.annotate(
                row["phoneme"],
                (row["prefix_length"], row["phoneme_surprisal"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=9,
            )

        ax.set_title(f"{token_id}: {target_word}")
        ax.set_xlabel("Phoneme position within word")
        ax.set_ylabel("Phoneme surprisal (-log2 probability)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        safe_word = "".join(ch if ch.isalnum() else "_" for ch in str(target_word))
        fig.savefig(output_dir / f"{token_id:04d}_{safe_word}.png", dpi=200)
        plt.close(fig)


def create_plots(cohort_jsonl_path: Path, phoneme_surprisal_csv_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_plot_data(cohort_jsonl_path, phoneme_surprisal_csv_path)

    enriched_csv = output_dir / "phoneme_surprisal_with_word_metadata.csv"
    data.to_csv(enriched_csv, index=False)

    plot_overlay(data, output_dir / "phoneme_surprisal_overlay.png")
    plot_word_grid(data, output_dir / "phoneme_surprisal_word_grid.png")

    if SAVE_INDIVIDUAL_WORD_PLOTS:
        plot_individual_words(data, output_dir / "individual_words")

    print(f"Saved enriched CSV to {enriched_csv}")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    story_dir = project_root / "output_nlp" / OUTPUT_GROUP / STORY_ID / "phoneme_level"

    cohort_jsonl_path = story_dir / COHORT_JSONL_FILENAME
    phoneme_surprisal_csv_path = story_dir / PHONEME_SURPRISAL_CSV_FILENAME
    output_dir = story_dir / "phoneme_surprisal_plots"

    create_plots(cohort_jsonl_path, phoneme_surprisal_csv_path, output_dir)
