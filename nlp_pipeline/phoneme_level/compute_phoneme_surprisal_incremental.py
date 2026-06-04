import json
import csv
import math
from collections import defaultdict
from pathlib import Path

from sympy import python


def compute_phoneme_surprisal_incremental(input_jsonl, output_csv):
    """
    Computes phoneme surprisal from incremental cohort probabilities.
    
    For each word, processes each phoneme incrementally:
    - Probability = cohort_mass / filtered_cohort_size
    - Surprisal = -log2(probability)
    
    Args:
        input_jsonl: Path to wide_cohort_incremental_probs.jsonl
        output_csv: Path to output CSV file
    """
    
    # Group data by token_id and target_word
    word_data = defaultdict(list)
    
    print(f"Reading {input_jsonl}...")
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            key = (record['token_id'], record['target_word'])
            word_data[key].append(record)
    
    print(f"Processing {len(word_data)} unique words...")
    
    # List to store all phoneme rows
    phoneme_rows = []
    
    # Process each word
    for (token_id, target_word), records in word_data.items():
        # Sort records by prefix_length to ensure incrementality
        records.sort(key=lambda x: x['prefix_length'])
        
        # Process each phoneme
        for record in records:
            prefix_phonemes = record['prefix_phonemes']
            cohort_mass = record['cohort_mass']
            filtered_cohort_size = record['filtered_cohort_size']
            
            # FIRST PHONEME
            if record['prefix_length'] == 1:

                # probability of the initial phoneme
                probability = cohort_mass

            # SUBSEQUENT PHONEMES
            else:

                previous_record = records[record['prefix_length'] - 2]
                previous_cohort_mass = previous_record['cohort_mass']

                if previous_cohort_mass > 0:
                    probability = cohort_mass / previous_cohort_mass
                else:
                    probability = 0.0

            # Numerical safety
            probability = max(probability, 1e-12)

            # Surprisal
            surprisal = -math.log2(probability)



            # Get the current phoneme (last one in the prefix)
            if prefix_phonemes:
                current_phoneme = prefix_phonemes[-1]
            else:
                continue
            
            phoneme_rows.append({
                'word': target_word,
                'phoneme_surprisal': surprisal,
                'phoneme': current_phoneme,
            })
    
    # Write to CSV
    print(f"Writing {len(phoneme_rows)} phoneme rows to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'word', 'phoneme', 'phoneme_surprisal'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(phoneme_rows)
    
    print(f"Done! Output saved to {output_csv}")
    print(f"Total phonemes processed: {len(phoneme_rows)}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    # 🦑 CHANGE HERE: cambia output_D e St01_D per usare un'altra storia.
    # Esempio: output_group = "output_B", story_id = "St03_B"
    output_group = "output_D"
    story_id = "St01_D"

    # 🦑 CHANGE HERE: usa "cohort_incremental_probs.jsonl" per il file completo,
    # oppure "cohort_incremental_probs_first50tokens.jsonl" per la prova piccola.
    input_filename = "cohort_incremental_probs_first50tokens.jsonl"

    # 🦑 CHANGE HERE: cambia il nome dell'output se vuoi distinguere test/storia completa.
    output_filename = "phoneme_surprisal_incremental_first50tokens.csv"

    input_file = project_root / "output_nlp" / output_group / story_id / input_filename
    output_file = project_root / "output_nlp" / output_group / story_id / output_filename
    
    compute_phoneme_surprisal_incremental(str(input_file), str(output_file))
