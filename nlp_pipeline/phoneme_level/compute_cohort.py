import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Configurable paths: change these only at the top of the file.
# CHANGE HERE (#🦑)
#   - STORY_PATH: percorso del file Excel della storia
#   - STORY_ID: nome della storia, es. St01_D
#   - OUTPUT_ROOT / OUTPUT_GROUP: cartella base output
#   - PHONITALIA_PATH: percorso al file datasets/phonitalia.csv/xlsx
# ---------------------------------------------------------------------------
STORY_PATH = Path(r"data\phoneme_onset_D\St02_D.xlsx") #🦑
STORY_ID = "St02_D" #🦑
OUTPUT_ROOT = Path(r'output_nlp') 
OUTPUT_GROUP = Path(r'output_D') #🦑
PHONITALIA_PATH = Path(r"datasets\phonitalia.xlsx")
COHORTS_FILENAME_TEMPLATE = "incremental_phonemic_cohorts_{story_id}.jsonl"
MAX_TOKENS_FOR_TEST = None  # usa 50 per un test veloce; None processa tutta la storia


def load_story(story_path: Path) -> pd.DataFrame:
    story_df = pd.read_excel(story_path)
    story_df = story_df[story_df["MAU"].notna()]
    story_df["ORT"] = story_df["ORT"].astype(str).str.strip().str.lower()
    story_df["MAU"] = story_df["MAU"].astype(str).str.strip()

    return story_df


def keep_first_tokens(story_df: pd.DataFrame, max_tokens: Optional[int]) -> pd.DataFrame:
    if max_tokens is None:
        return story_df

    token_ids = sorted(story_df["TOKEN"].dropna().unique())[:max_tokens]
    return story_df[story_df["TOKEN"].isin(token_ids)].copy()


def read_phonitalia(lexicon_path: Path) -> pd.DataFrame:
    if lexicon_path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(lexicon_path)
        except PermissionError:
            csv_fallback = lexicon_path.with_suffix(".csv")
            if csv_fallback.exists():
                print(f"Cannot read {lexicon_path}; using {csv_fallback} instead.")
                return pd.read_csv(csv_fallback, sep=";")
            raise
    return pd.read_csv(lexicon_path, sep=";")


def build_lexicon(lexicon_path: Path) -> List[Dict[str, List[str]]]:
    lex_df = read_phonitalia(lexicon_path)
    lex_df = lex_df[lex_df["SAMPA"].notna()]
    lexicon = []
    for _, row in lex_df.iterrows():
        word = str(row["WORD"]).strip().lower()
        phonemes = str(row["SAMPA"]).strip().split()
        lexicon.append({"word": word, "phonemes": phonemes})
    return lexicon


def build_incremental_phonemic_cohorts(story_df: pd.DataFrame, lexicon: List[Dict[str, List[str]]]) -> List[Dict]:
    grouped_story = (
        story_df.groupby("TOKEN")
        .agg({"ORT": "first", "MAU": list})
        .reset_index()
    )

    records: List[Dict] = []
    for _, row in grouped_story.iterrows():
        token_id = int(row["TOKEN"])
        target_word = row["ORT"]
        target_phonemes = row["MAU"]

        for i in range(len(target_phonemes)):
            prefix = target_phonemes[: i + 1]
            cohort = [entry["word"] for entry in lexicon if entry["phonemes"][: len(prefix)] == prefix]
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


def get_output_base(output_root: Path, output_group: str, story_id: str) -> Path:
    return output_root / output_group / story_id / "phoneme_level"


def compute_all(
    story_path: Path = STORY_PATH,
    phonitalia_path: Path = PHONITALIA_PATH,
    story_id: str = STORY_ID,
    output_root: Path = OUTPUT_ROOT,
    output_group: str = OUTPUT_GROUP,
    max_tokens_for_test: Optional[int] = MAX_TOKENS_FOR_TEST,
) -> None:
    story_df = load_story(story_path)
    story_df = keep_first_tokens(story_df, max_tokens_for_test)
    lexicon = build_lexicon(phonitalia_path)
    cohort_records = build_incremental_phonemic_cohorts(story_df, lexicon)

    output_base = get_output_base(output_root, output_group, story_id)
    incremental_cohorts_path = output_base / COHORTS_FILENAME_TEMPLATE.format(story_id=story_id)

    save_jsonl(cohort_records, incremental_cohorts_path)
    print(f"Saved incremental cohorts to {incremental_cohorts_path}")


if __name__ == "__main__":
    compute_all()
