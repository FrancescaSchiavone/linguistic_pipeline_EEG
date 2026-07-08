import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Fixed paths for the story and output files.
# ---------------------------------------------------------------------------
OUTPUT_GROUP = "output_C" #🦑
STORY_ID = "St05_C" #🦑
COHORT_JSONL_FILENAME = "incremental_phonemic_cohorts_gpt_St05_C.jsonl"#🦑
PHONEME_SURPRISAL_CSV_FILENAME = "phoneme_surprisal_St05_C.csv"#🦑

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


def assign_word_length_clusters(data: pd.DataFrame) -> pd.DataFrame:
    word_lengths = (
        data.groupby(["token_id", "target_word"], sort=False)["prefix_length"]
        .max()
        .rename("word_phoneme_length")
        .reset_index()
    )
    data = data.merge(word_lengths, on=["token_id", "target_word"])

    bins = [0, 3, 6, float("inf")]
    labels = ["1-3 phonemes", "4-6 phonemes", "7+ phonemes"]
    data["length_group"] = pd.cut(
        data["word_phoneme_length"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )
    data["length_group"] = data["length_group"].astype(str)

    return data


def _sanitize_filename(value: str) -> str:
    return value.replace("+", "plus").replace(" ", "_").replace("/", "_")


def plot_overlay_by_length_group(data: pd.DataFrame, output_dir: Path) -> None:
    if data.empty:
        return

    for length_group in sorted(data["length_group"].unique()):
        if length_group == "nan":
            continue

        group_data = data[data["length_group"] == length_group]
        if group_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 6))
        for _, word_df in group_data.groupby(["token_id", "target_word"], sort=True):
            word_df = word_df.sort_values("prefix_length")
            ax.plot(
                word_df["prefix_length"],
                word_df["phoneme_surprisal"],
                color="0.65",
                linewidth=1,
                alpha=0.35,
            )

        mean_by_position = (
            group_data.groupby("prefix_length", as_index=False)["phoneme_surprisal"]
            .mean()
            .sort_values("prefix_length")
        )
        ax.plot(
            mean_by_position["prefix_length"],
            mean_by_position["phoneme_surprisal"],
            color="#d62828",
            linewidth=2.5,
            marker="o",
            label="Mean across words",
        )

        ax.set_title(
            f"Phoneme Surprisal Trajectory for {length_group}"
        )
        ax.set_xlabel("Phoneme position within word")
        ax.set_ylabel("Phoneme surprisal (-log2 probability)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()

        output_path = output_dir / f"phoneme_surprisal_overlay_{_sanitize_filename(length_group)}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_mean_surprisal_for_length_groups(data: pd.DataFrame, output_path: Path) -> None:
    if data.empty:
        return

    mean_by_group = (
        data.groupby(["length_group", "prefix_length"], as_index=False)["phoneme_surprisal"]
        .mean()
    )
    pivot = mean_by_group.pivot(index="prefix_length", columns="length_group", values="phoneme_surprisal")

    fig, ax = plt.subplots(figsize=(11, 6))
    for length_group in sorted(pivot.columns, key=lambda x: (x != "1-3 phonemes", x != "4-6 phonemes", x != "7+ phonemes")):
        ax.plot(
            pivot.index,
            pivot[length_group],
            linewidth=2,
            marker="o",
            label=length_group,
        )

    ax.set_title("Mean Phoneme Surprisal by Word Length Cluster")
    ax.set_xlabel("Phoneme position within word")
    ax.set_ylabel("Phoneme surprisal (-log2 probability)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_surprisal_series(data: pd.DataFrame, output_path: Path) -> None:
    if data.empty:
        return

    x_positions = range(len(data))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        x_positions,
        data["phoneme_surprisal"],
        color="#4c78a8",
        linewidth=1.5,
    )

    ax.set_title("Phoneme surprisal across the story")
    ax.set_xlabel("")
    ax.set_ylabel("Phoneme surprisal (-log2 probability)")
    ax.set_xticks([])
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def create_plots(cohort_jsonl_path: Path, phoneme_surprisal_csv_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_plot_data(cohort_jsonl_path, phoneme_surprisal_csv_path)

    data = assign_word_length_clusters(data)

    enriched_csv = output_dir / "phoneme_surprisal_with_word_metadata.csv"
    data.to_csv(enriched_csv, index=False)

    plot_overlay(data, output_dir / "phoneme_surprisal_overlay.png")
    plot_overlay_by_length_group(data, output_dir)
    plot_mean_surprisal_for_length_groups(data, output_dir / "phoneme_surprisal_by_length_group.png")
    plot_surprisal_series(data, output_dir / "phoneme_surprisal_series.png")

    print(f"Saved enriched CSV to {enriched_csv}")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    story_dir = project_root / "output_nlp" / OUTPUT_GROUP / STORY_ID / "phoneme_level"

    cohort_jsonl_path = story_dir / COHORT_JSONL_FILENAME
    phoneme_surprisal_csv_path = story_dir / PHONEME_SURPRISAL_CSV_FILENAME
    output_dir = story_dir / "phoneme_surprisal_plots"

    create_plots(cohort_jsonl_path, phoneme_surprisal_csv_path, output_dir)
