import os
import pandas as pd



#reading csv
df_surprisal = pd.read_csv(r'output_nlp\output_B\St07_B\surprisal_entropyN_St07_B.csv', encoding="utf-8") 
df_diss = pd.read_csv(r'output_nlp\output_B\St07_B\dissimilarity_St07_B.csv', encoding="utf-8")                          
df_features = pd.read_csv(r'output_nlp\output_B\St07_B\St07_B.csv', sep=";", encoding="utf-8")
df_word_onset = pd.read_csv(r'data\word_onset_B\word_onset_St07_B.csv', sep=";", encoding="utf-8")

#from object to float (transform ',' in '.')
df_features["Zipf_freq"] = df_features["Zipf_freq"].str.replace(',', '.').astype(float)

#index from 1
df_surprisal.index = range(1, len(df_surprisal) + 1) 
df_diss.index = range(1, len(df_diss)+1)
df_word_onset.index = range(1, len(df_word_onset)+1)
df_features.index = range (1, len(df_features)+1)

#creation new df
df_surpr = pd.concat([df_word_onset["BEGIN"], df_surprisal["surprisal"], df_features["type_of_words"], df_surprisal["word"]], axis=1)
df_entrop = pd.concat([df_word_onset["BEGIN"], df_surprisal["entropy"], df_features["type_of_words"], df_surprisal["word"]], axis=1)
df_semantics = pd.concat([df_word_onset["BEGIN"], df_diss["semantic_dissimilarity"], df_features["type_of_words"], df_diss["word"]], axis=1)
df_word_freq = pd.concat([df_word_onset["BEGIN"], df_features["Zipf_freq"], df_features["type_of_words"], df_features["tokens_no_punct"]], axis=1)
df_word_freq = df_word_freq.rename(columns={"tokens_no_punct": "word"})

# 'content' filter, drop 'type_of_words' column 
# dfs = [df_surpr, df_entrop, df_semantics, df_word_freq]

# for i, df in enumerate(dfs):
#     df = df[df["type_of_words"]=="content"]
#     df.drop("type_of_words", axis=1, inplace=True)

#     dfs[i] = df

# df_surpr, df_entrop, df_semantics, df_word_freq = dfs

#normalize data between 0 and 1 
# scaler = MinMaxScaler()
# df_surpr['surprisal'] = scaler.fit_transform(df_surpr[['surprisal']])
# df_semantics['semantic_dissimilarity'] = scaler.fit_transform(df_semantics[["semantic_dissimilarity"]])
# df_entrop["entropy"] = scaler.fit_transform(df_entrop[["entropy"]])
# df_word_freq["Zipf_freq"] = scaler.fit_transform(df_word_freq[["Zipf_freq"]])

#saveCSV
df_surpr.to_csv(r'data\word_onset+features_B\St07_B\Surprisal_St_07B.csv')
df_entrop.to_csv(r'data\word_onset+features_B\St07_B\Entropy_St_07B.csv')
df_semantics.to_csv(r'data\word_onset+features_B\St07_B\Dissimilarity_St_07B.csv')
df_word_freq.to_csv(r'data\word_onset+features_B\St07_B\WordFreq_St_07B.csv')


###
import os
import pandas as pd

BASE_OUTPUT_NLP = "output_nlp"
BASE_DATA = "data"
GROUPS = ["A", "B", "C", "D"]

for group in GROUPS:
    group_root = os.path.join(BASE_OUTPUT_NLP, f"output_{group}")
    if not os.path.isdir(group_root):
        print(f"Gruppo {group}: cartella non trovata -> {group_root}")
        continue

    # prende tutte le cartelle soggetto, es. St07_B, St01_B, ecc.
    st_folders = sorted([
        d for d in os.listdir(group_root)
        if os.path.isdir(os.path.join(group_root, d)) and d.startswith("St")
    ])

    print(f"\n=== Gruppo {group} | trovati {len(st_folders)} soggetti ===")

    for st_code in st_folders:
        print(f"--- {st_code} ---")

        surprisal_path = os.path.join(group_root, st_code, f"surprisal_entropyN_{st_code}.csv")
        diss_path      = os.path.join(group_root, st_code, f"dissimilarity_{st_code}.csv")
        features_path  = os.path.join(group_root, st_code, f"{st_code}.csv")
        onset_path     = os.path.join(BASE_DATA, f"word_onset_{group}", f"word_onset_{st_code}.csv")

        out_dir = os.path.join(BASE_DATA, f"word_onset+features_{group}", st_code)
        os.makedirs(out_dir, exist_ok=True)

        try:
            df_surprisal  = pd.read_csv(surprisal_path, encoding="utf-8")
            df_diss       = pd.read_csv(diss_path, encoding="utf-8")
            df_features   = pd.read_csv(features_path, sep=";", encoding="utf-8")
            df_word_onset = pd.read_csv(onset_path, sep=";", encoding="utf-8")
        except FileNotFoundError as e:
            print("   File mancante, skip:", e.filename)
            continue

        # ',' -> '.' per Zipf_freq
        df_features["Zipf_freq"] = df_features["Zipf_freq"].astype(str).str.replace(",", ".").astype(float)

        # index da 1
        df_surprisal.index  = range(1, len(df_surprisal) + 1)
        df_diss.index       = range(1, len(df_diss) + 1)
        df_word_onset.index = range(1, len(df_word_onset) + 1)
        df_features.index   = range(1, len(df_features) + 1)

        # nuovi df
        df_surpr = pd.concat(
            [df_word_onset["BEGIN"], df_surprisal["surprisal"], df_features["type_of_words"], df_surprisal["word"]],
            axis=1
        )
        df_entrop = pd.concat(
            [df_word_onset["BEGIN"], df_surprisal["entropy"], df_features["type_of_words"], df_surprisal["word"]],
            axis=1
        )
        df_semantics = pd.concat(
            [df_word_onset["BEGIN"], df_diss["semantic_dissimilarity"], df_features["type_of_words"], df_diss["word"]],
            axis=1
        )
        df_word_freq = pd.concat(
            [df_word_onset["BEGIN"], df_features["Zipf_freq"], df_features["type_of_words"], df_features["tokens_no_punct"]],
            axis=1
        ).rename(columns={"tokens_no_punct": "word"})

        # salva
        df_surpr.to_csv(os.path.join(out_dir, f"Surprisal_{st_code}.csv"), index=False)
        df_entrop.to_csv(os.path.join(out_dir, f"Entropy_{st_code}.csv"), index=False)
        df_semantics.to_csv(os.path.join(out_dir, f"Dissimilarity_{st_code}.csv"), index=False)
        df_word_freq.to_csv(os.path.join(out_dir, f"WordFreq_{st_code}.csv"), index=False)

        print("   OK ->", out_dir)
