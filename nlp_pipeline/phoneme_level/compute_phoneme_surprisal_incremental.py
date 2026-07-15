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
            probability = max(probability, 1e-6)

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

    # 🦑 CHANGE HERE: cambia output_C e St05_C per usare un'altra storia.
    # Esempio: output_group = "output_B", story_id = "St03_B"
    output_group = "output_D"
    story_id = "St10_D"

    # 🦑 CHANGE HERE: usa il prefisso del file senza estensione.
    # Il file completo per GPT dovrebbe essere:
    # output_nlp/output_C/St05_C/phoneme_level/incremental_phonemic_cohorts_gpt_St05_C.jsonl
    input_basename = "incremental_phonemic_cohorts_gpt"

    # 🦑 CHANGE HERE: cambia il nome dell'output se vuoi distinguere test/storia completa.
    # Il CSV sarà scritto come:
    # output_nlp/output_C/St05_C/phoneme_level/phoneme_surprisal_St05_C.csv
    output_basename = "phoneme_surprisal"

    input_file = project_root / "output_nlp" / output_group / story_id / "phoneme_level" / f"{input_basename}_{story_id}.jsonl"
    output_file = project_root / "output_nlp" / output_group / story_id / "phoneme_level" / f"{output_basename}_{story_id}.csv"

    compute_phoneme_surprisal_incremental(str(input_file), str(output_file))
