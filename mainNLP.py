import argparse
import os
import json
import pandas as pd
from nlp_pipeline.processor import process_text_file
from nlp_pipeline.surprisal_entropy import calculate_surprisal_entropy
from nlp_pipeline.semantic_dissimilarity import calculate_semantic_dissimilarity

def process_folder(input_dir, output_dir):
    """
    Processes all .txt files in the specified input directory and saves outputs to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_dir, filename)
            print(f"Processing file: {filename}")
            process_text_file(filepath, output_dir)
            calculate_surprisal_entropy(filepath, output_dir)
            calculate_semantic_dissimilarity(filepath, output_dir)

def aggregate_summaries(output_dir):
    """
    Aggregates all summary JSON files into a single DataFrame and saves it to the output directory.
    """
    summary_rows = []
    for subfolder in os.listdir(output_dir):
        sub_path = os.path.join(output_dir, subfolder)
        if os.path.isdir(sub_path):
            for file in os.listdir(sub_path):
                if file.endswith("_summary.json"):
                    json_path = os.path.join(sub_path, file)
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["text_id"] = subfolder
                        summary_rows.append(data)

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        cols = ["text_id"] + [c for c in df_summary.columns if c != "text_id"]
        df_summary = df_summary[cols]

        percent_cols = ["num_function_words", "num_content_words",
                        "num_verbs", "num_adjectives", "num_nouns"]

        numeric_cols = [
            c for c in df_summary.columns
            if c not in ["text_id", "top_lemmas", "top_bigrams", "Zipf_freq"]
            and not c.endswith("_display") and not c.endswith("_pct")
        ]
        mean_row = df_summary[numeric_cols].mean(numeric_only=True)
        std_row  = df_summary[numeric_cols].std(numeric_only=True)

        combined = {}
        for col in numeric_cols:
            if col in mean_row.index:
                if col in percent_cols and f"{col}_pct" in df_summary.columns:
                    mean_val = df_summary[f"{col}_pct"].mean()
                    std_val  = df_summary[f"{col}_pct"].std()
                    combined[col] = f"{mean_val:.2f}% ± {std_val:.2f}%"
                else:
                    combined[col] = f"{mean_row[col]:.2f} ± {std_row[col]:.2f}"
        combined["text_id"] = "MEAN ± STD"

        for col in percent_cols:
            disp_col = col + "_display"
            if disp_col in df_summary.columns:
                df_summary[col] = df_summary[disp_col]

        empty_row = {col: "" for col in df_summary.columns}
        df_summary = pd.concat([df_summary, pd.DataFrame([empty_row, combined])], ignore_index=True)

        drop_cols = ["Zipf_std", "Zipf_freq"] + [f"{c}_display" for c in percent_cols] + [f"{c}_pct" for c in percent_cols]
        df_summary = df_summary.drop(columns=[c for c in drop_cols if c in df_summary.columns])

        summary_csv_path = os.path.join(output_dir, "summary_all_texts.csv")
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"Saved aggregated summary: {summary_csv_path}")
    else:
        print("No summary JSON files found.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the linguistic pipeline on a folder of text files.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input folder containing .txt files.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder for results.")

    args = parser.parse_args()
    process_folder(args.input, args.output)
    aggregate_summaries(args.output)

    