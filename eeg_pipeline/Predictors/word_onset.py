import os
import pandas as pd
import glob

#AGGIUNTA COLONNA 'TOKEN'
cartella37 = "data\\alignment_37"  
for file in glob.glob(os.path.join(cartella37, "*.xlsx")):
    print(f"Processo: {file}")
    df = pd.read_excel(file)
    df["TOKEN"] = (df["ORT"] != df["ORT"].shift()).cumsum() - 1
    df.to_excel(file, index=False)

cartella710 = "data\\alignment_710" 
for file in glob.glob(os.path.join(cartella710, "*.xlsx")):
    print(f"Processo: {file}")
    df = pd.read_excel(file)
    df["TOKEN"] = (df["ORT"] != df["ORT"].shift()).cumsum() - 1
    df.to_excel(file, index=False)

cartella1015 = "data\\alignment_1015"
for file in glob.glob(os.path.join(cartella1015, "*.xlsx")):
    print(f"Processo: {file}")
    df = pd.read_excel(file)
    df["TOKEN"] = (df["ORT"] != df["ORT"].shift()).cumsum() - 1
    df.to_excel(file, index=False)

#WORD ONSET

filepath37 = "data\\phoneme_onset_37"

for file in glob.glob(os.path.join(filepath37, "*.xlsx")):
    print(f"Processo: {file}")
    df = pd.read_excel(file)

    df_valid = df[df["TOKEN"] != -1]
    df_first = df_valid.drop_duplicates(subset=["TOKEN"], keep="first")
    result = df_first[["TOKEN", "BEGIN", "ORT"]]
    result = result.sort_values(by="TOKEN").reset_index(drop=True)

    output_dir = "data\\word_onset_37"
    name_base = os.path.splitext(os.path.basename(file))[0]   
    csv_path = os.path.join(output_dir, f"word_onset_{name_base}.csv")
    result.to_csv(csv_path, index=False)

filepath710 = "data\\phoneme_onset_710"

for file in glob.glob(os.path.join(filepath710, "*.xlsx")):
    print(f"Processo: {file}")
    df = pd.read_excel(file)

    df_valid = df[df["TOKEN"] != -1]
    df_first = df_valid.drop_duplicates(subset=["TOKEN"], keep="first")
    result = df_first[["TOKEN", "BEGIN", "ORT"]]
    result = result.sort_values(by="TOKEN").reset_index(drop=True)

    output_dir = "data\\word_onset_710"
    name_base = os.path.splitext(os.path.basename(file))[0]   
    csv_path = os.path.join(output_dir, f"word_onset_{name_base}.csv")
    result.to_csv(csv_path, index=False)

filepath1015 = "data\\phoneme_onset_1015"

for file in glob.glob(os.path.join(filepath1015, "*.xlsx")):
    print(f"Processo: {file}")
    df = pd.read_excel(file)

    df_valid = df[df["TOKEN"] != -1]
    df_first = df_valid.drop_duplicates(subset=["TOKEN"], keep="first")
    result = df_first[["TOKEN", "BEGIN", "ORT"]]
    result = result.sort_values(by="TOKEN").reset_index(drop=True)

    output_dir = "data\\word_onset_1015"
    name_base = os.path.splitext(os.path.basename(file))[0]   
    csv_path = os.path.join(output_dir, f"word_onset_{name_base}.csv")
    result.to_csv(csv_path, index=False)

