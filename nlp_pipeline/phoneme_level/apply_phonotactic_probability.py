from pathlib import Path

import pandas as pd


def load_bigram_probabilities(bigram_matrix_csv):
    """Load a bigram probability matrix saved by phonotactic_probability.py."""
    matrix = pd.read_csv(bigram_matrix_csv, index_col=0)

    probabilities = {}
    for previous_phoneme in matrix.index:
        for current_phoneme in matrix.columns:
            probabilities[(str(previous_phoneme), str(current_phoneme))] = float(
                matrix.loc[previous_phoneme, current_phoneme]
            )

    return probabilities


def load_trigram_probabilities(trigram_probabilities_csv):
    """Load trigram probabilities saved by phonotactic_probability.py."""
    trigram_df = pd.read_csv(trigram_probabilities_csv)

    probabilities = {}
    for _, row in trigram_df.iterrows():
        probabilities[
            (
                str(row["ph1"]).strip(),
                str(row["ph2"]).strip(),
                str(row["ph3"]).strip(),
            )
        ] = float(row["probability"])

    return probabilities


def infer_story_output_file(input_file, output_filename=None):
    """
    Infer output_nlp/output_<story letter>/<story id>/... from an input story file.

    Example:
    data/phoneme_onset_D/St01_D.xlsx
    -> output_nlp/output_D/St01_D/St01_D_with_phonotactic_probability.xlsx
    """
    input_file = Path(input_file)
    project_root = Path(__file__).resolve().parents[2]

    story_id = input_file.stem
    story_group = story_id.rsplit("_", 1)[-1]

    if output_filename is None:
        output_filename = f"{story_id}_with_phonotactic_probability.xlsx"

    return (
        project_root
        / "output_nlp"
        / f"output_{story_group}"
        / story_id
        / output_filename
    )


def add_phonotactic_probability(
    input_file,
    bigram_matrix_csv,
    trigram_probabilities_csv,
    output_file=None,
    token_column="TOKEN",
    phoneme_column="MAU",
    output_column="phonotactic_probability",
    context_column="phonotactic_context",
):
    """
    Add one phonotactic probability value for each phoneme row.

    Logic:
    - first phoneme: empty probability, context = initial
    - second phoneme: bigram probability P(current | previous)
    - third phoneme onward: trigram probability P(current | previous two)
    - if the trigram is unavailable: back off to the bigram
    - if the bigram is unavailable too: 0.0
    """
    input_file = Path(input_file)
    if output_file is None:
        output_file = infer_story_output_file(input_file)
    else:
        output_file = Path(output_file)

    bigram_probabilities = load_bigram_probabilities(bigram_matrix_csv)
    trigram_probabilities = load_trigram_probabilities(trigram_probabilities_csv)

    if input_file.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)

    missing_columns = [
        column
        for column in (token_column, phoneme_column)
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {input_file}: {missing_columns}"
        )

    df[output_column] = pd.NA
    df[context_column] = pd.NA

    for _, token_rows in df.groupby(token_column, sort=False):
        previous_phonemes = []

        for row_index, row in token_rows.iterrows():
            current_phoneme = row[phoneme_column]

            if pd.isna(current_phoneme):
                previous_phonemes = []
                continue

            current_phoneme = str(current_phoneme).strip()

            if len(previous_phonemes) == 0:
                df.loc[row_index, context_column] = "initial"

            elif len(previous_phonemes) == 1:
                probability = bigram_probabilities.get(
                    (previous_phonemes[-1], current_phoneme),
                    0.0,
                )

                df.loc[row_index, output_column] = probability
                df.loc[row_index, context_column] = "bigram"

            else:
                trigram_key = (
                    previous_phonemes[-2],
                    previous_phonemes[-1],
                    current_phoneme,
                )
                bigram_key = (previous_phonemes[-1], current_phoneme)

                if trigram_key in trigram_probabilities:
                    probability = trigram_probabilities[trigram_key]
                    context = "trigram"
                else:
                    probability = bigram_probabilities.get(bigram_key, 0.0)
                    context = "bigram_fallback"

                df.loc[row_index, output_column] = probability
                df.loc[row_index, context_column] = context

            previous_phonemes.append(current_phoneme)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.suffix.lower() == ".xlsx":
        df.to_excel(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    # CHANGE HERE: input Excel/CSV with one phoneme per row.
    # The output path is inferred automatically from this path.
    # Example:
    # data/phoneme_onset_D/St01_D.xlsx
    # -> output_nlp/output_D/St01_D/St01_D_with_phonotactic_probability.xlsx
    input_file = project_root / "data" / "phoneme_onset_D" / "St01_D.xlsx"

    # Generated by nlp_pipeline/phoneme_level/phonotactic_probability.py.
    bigram_matrix_csv = project_root / "bigram_matrix.csv"
    trigram_probabilities_csv = project_root / "trigram_probabilities.csv"

    output_df = add_phonotactic_probability(
        input_file=input_file,
        bigram_matrix_csv=bigram_matrix_csv,
        trigram_probabilities_csv=trigram_probabilities_csv,
    )

    print(f"Saved {len(output_df)} rows.")
