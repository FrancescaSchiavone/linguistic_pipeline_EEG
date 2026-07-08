import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Configurable paths and runtime parameters.
# Edit this block when switching story or output group.
#   - STORY_PATH: Excel file containing story phoneme annotations.
#   - STORY_ID: story name, e.g. St01_D.
#   - OUTPUT_ROOT / OUTPUT_GROUP: base output folder.
#   - PHONITALIA_PATH: lexicon file path, either datasets/phonitalia.csv or .xlsx.
# ---------------------------------------------------------------------------
STORY_PATH = Path(r"data\phoneme_onset_C\St05_C.xlsx")  # Story Excel file. 🦑
STORY_ID = "St05_C"  # Story identifier used in file names.🦑
OUTPUT_ROOT = Path(r'output_nlp') 
OUTPUT_GROUP = Path(r'output_C')  # Output subgroup for this condition.🦑
PHONITALIA_PATH = Path(r"datasets\phonitalia.xlsx")
COHORTS_FILENAME_TEMPLATE = "incremental_phonemic_cohorts_{story_id}.jsonl"
MAX_TOKENS_FOR_TEST = None  # Use 50 for a quick test; None processes the full story. 🦑


def load_story(story_path: Path) -> pd.DataFrame:
    """
    Load and normalize the story phoneme annotation file.

    Args:
        story_path: Path to the Excel file containing at least TOKEN, ORT, and MAU columns.

    Returns:
        A DataFrame with rows that have MAU values, lowercase stripped ORT values,
        and stripped MAU phoneme labels.
    """
    # Read the story file and keep rows with phoneme labels.
    story_df = pd.read_excel(story_path)
    story_df = story_df[story_df["MAU"].notna()]
    # Normalize words and phoneme strings for matching.
    story_df["ORT"] = story_df["ORT"].astype(str).str.strip().str.lower()
    story_df["MAU"] = story_df["MAU"].astype(str).str.strip()

    return story_df


def keep_first_tokens(story_df: pd.DataFrame, max_tokens: Optional[int]) -> pd.DataFrame:
    """
    Optionally keep only the first story tokens for faster test runs.

    Args:
        story_df: Story DataFrame containing a TOKEN column.
        max_tokens: Maximum number of unique TOKEN values to keep, or None for all tokens.

    Returns:
        The original DataFrame when max_tokens is None, otherwise a filtered copy.
    """
    # Optionally restrict the story to the first N tokens.
    if max_tokens is None:
        return story_df

    token_ids = sorted(story_df["TOKEN"].dropna().unique())[:max_tokens]
    return story_df[story_df["TOKEN"].isin(token_ids)].copy()


def read_phonitalia(lexicon_path: Path) -> pd.DataFrame:
    """
    Read the PhonItalia lexicon from Excel or CSV.

    Args:
        lexicon_path: Path to the PhonItalia .xlsx, .xls, or .csv file.

    Returns:
        A DataFrame containing the lexicon entries.
    """
    # Prefer Excel when requested, with a CSV fallback if the file is locked.
    if lexicon_path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(lexicon_path)
        except PermissionError:
            csv_fallback = lexicon_path.with_suffix(".csv")
            if csv_fallback.exists():
                print(f"Cannot read {lexicon_path}; using {csv_fallback} instead.")
                return pd.read_csv(csv_fallback, sep=";")
            raise
    # Read CSV lexicons using the expected semicolon separator.
    return pd.read_csv(lexicon_path, sep=";")


def build_lexicon(lexicon_path: Path) -> List[Dict[str, List[str]]]:
    """
    Build a normalized word-to-phoneme lexicon from PhonItalia.

    Args:
        lexicon_path: Path to the PhonItalia lexicon file.

    Returns:
        A list of dictionaries with word and phonemes keys.
    """
    # Load the source lexicon and discard entries without phoneme strings.
    lex_df = read_phonitalia(lexicon_path)
    lex_df = lex_df[lex_df["SAMPA"].notna()]
    lexicon = []
    for _, row in lex_df.iterrows():
        # Normalize each word and split its SAMPA transcription into phonemes.
        word = str(row["WORD"]).strip().lower()
        phonemes = str(row["SAMPA"]).strip().split()
        lexicon.append({"word": word, "phonemes": phonemes})
    return lexicon


def build_incremental_phonemic_cohorts(story_df: pd.DataFrame, lexicon: List[Dict[str, List[str]]]) -> List[Dict]:
    """
    Build phonemic cohorts for every incremental target-word prefix.

    Args:
        story_df: Story DataFrame with TOKEN, ORT, and MAU columns.
        lexicon: List of lexicon entries with word and phonemes keys.

    Returns:
        A list of JSON-serializable records, one for each token-prefix pair.
    """
    # Collapse phoneme rows into one target word and phoneme list per token.
    grouped_story = (
        story_df.groupby("TOKEN")
        .agg({"MAU": list})
        .reset_index()
    )

    records: List[Dict] = []
    for _, row in grouped_story.iterrows():
        token_id = int(row["TOKEN"])
        target_phonemes = row["MAU"]

        # Resolve the target word from the lexicon using the whole phoneme sequence.
        target_word = next(
            (entry["word"] for entry in lexicon if entry["phonemes"] == target_phonemes),
            None,
        )
        if target_word is None:
            target_word = ""

        # Match each growing target prefix against all lexicon entries.
        for i in range(len(target_phonemes)):
            prefix = target_phonemes[: i + 1]
            if len(prefix) == len(target_phonemes):
                # At the final phoneme, keep only the word that matches the full phoneme sequence.
                cohort = [target_word] if target_word else []
            else:
                cohort = [
                    entry["word"]
                    for entry in lexicon
                    if entry["phonemes"][: len(prefix)] == prefix
                ]
            records.append({
                "token_id": token_id,
                "target_word": target_word,
                "prefix_phonemes": prefix,
                "prefix_length": len(prefix),
                "cohort_size": len(cohort),
                "cohort_words": cohort,
            })

    return records


def save_jsonl(records: List[Dict], output_path: Path) -> None:
    """
    Save records to a JSONL file.

    Args:
        records: JSON-serializable records to write.
        output_path: Destination JSONL path.

    Returns:
        None. The function writes one JSON object per line to output_path.
    """
    # Create the output folder before writing the JSONL file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


def get_output_base(output_root: Path, output_group: str, story_id: str) -> Path:
    """
    Build the output directory used for phoneme-level files.

    Args:
        output_root: Base output folder.
        output_group: Output group or condition folder.
        story_id: Story identifier used as a subfolder.

    Returns:
        Path to the story-specific phoneme-level output directory.
    """
    # Build the standard output folder for one story.
    return output_root / output_group / story_id / "phoneme_level"


def compute_all(
    story_path: Path = STORY_PATH,
    phonitalia_path: Path = PHONITALIA_PATH,
    story_id: str = STORY_ID,
    output_root: Path = OUTPUT_ROOT,
    output_group: str = OUTPUT_GROUP,
    max_tokens_for_test: Optional[int] = MAX_TOKENS_FOR_TEST,
) -> None:
    """
    Run the full phonemic cohort-building pipeline for one story.

    Args:
        story_path: Path to the story Excel file.
        phonitalia_path: Path to the PhonItalia lexicon file.
        story_id: Story identifier used to resolve output file names.
        output_root: Base output folder.
        output_group: Output group or condition folder.
        max_tokens_for_test: Optional number of initial story tokens to process for tests.

    Returns:
        None. The function writes the incremental phonemic cohorts JSONL file to disk.
    """
    # Load inputs and optionally shorten the story for a test run.
    story_df = load_story(story_path)
    story_df = keep_first_tokens(story_df, max_tokens_for_test)
    lexicon = build_lexicon(phonitalia_path)
    # Build one cohort record for every target-word phoneme prefix.
    cohort_records = build_incremental_phonemic_cohorts(story_df, lexicon)

    # Resolve the output path and write JSONL records.
    output_base = get_output_base(output_root, output_group, story_id)
    incremental_cohorts_path = output_base / COHORTS_FILENAME_TEMPLATE.format(story_id=story_id)

    save_jsonl(cohort_records, incremental_cohorts_path)
    print(f"Saved incremental cohorts to {incremental_cohorts_path}")


if __name__ == "__main__":
    compute_all()
