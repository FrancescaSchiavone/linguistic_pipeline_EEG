import os
import pandas as pd
import glob


#FOLDER 
#AGGIUNTA COLONNA 'TOKEN'
# cartella37 = "data\\alignment_37"  
# for file in glob.glob(os.path.join(cartella37, "*.xlsx")):
#     print(f"Processo: {file}")
#     df = pd.read_excel(file)
#     df["TOKEN"] = (df["ORT"] != df["ORT"].shift()).cumsum() - 1
#     df.to_excel(file, index=False)

cartella37 = r'data\phoneme_onset_B' 
for file in glob.glob(os.path.join(cartella37, "*.xlsx")):
    print(f"Processo: {file}")
    df = pd.read_excel(file)
    df["TOKEN"] = (df["ORT"] != df["ORT"].shift()).cumsum() - 1
    df.to_excel(file, index=False)

cartella710 = r'data\phoneme_onset_C' 
for file in glob.glob(os.path.join(cartella710, "*.xlsx")):
    print(f"Processo: {file}")
    df = pd.read_excel(file)
    df["TOKEN"] = (df["ORT"] != df["ORT"].shift()).cumsum() - 1
    df.to_excel(file, index=False)

cartella1015 = "data\\phoneme_onset_D"
for file in glob.glob(os.path.join(cartella1015, "*.xlsx")):
    print(f"Processo: {file}")
    df = pd.read_excel(file)
    df["TOKEN"] = (df["ORT"] != df["ORT"].shift()).cumsum() - 1
    df.to_excel(file, index=False)

#VERSIONE SINGOLO FILE
import pandas as pd

file = r"C:\Users\schia\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\output_MAUS_Chiara_Finali_UsatiPerPaperFonemi\St04_C_CORRETTO.xlsx"
df = pd.read_excel(file)


# Crea il token ogni volta che cambia ORT
df["TOKEN"] = (df["ORT"] != df["ORT"].shift()).cumsum() - 1




df.to_excel(file, index=False)




#WORD ONSET

filepath37 = "data\\phoneme_onset_B"

for file in glob.glob(os.path.join(filepath37, "*.xlsx")):
    print(f"Processo: {file}")
    df = pd.read_excel(file)

    df_valid = df[df["TOKEN"] != -1]
    df_first = df_valid.drop_duplicates(subset=["TOKEN"], keep="first")
    result = df_first[["TOKEN", "BEGIN", "ORT"]]
    result = result.sort_values(by="TOKEN").reset_index(drop=True)

    output_dir = "data\\word_onset_C"
    name_base = os.path.splitext(os.path.basename(file))[0]   
    csv_path = os.path.join(output_dir, f"word_onset_{name_base}.csv")
    result.to_csv(csv_path, index=False)

filepath710 = "data\\phoneme_onset_C"

for file in glob.glob(os.path.join(filepath710, "*.xlsx")):
    print(f"Processo: {file}")
    df = pd.read_excel(file)

    df_valid = df[df["TOKEN"] != -1]
    df_first = df_valid.drop_duplicates(subset=["TOKEN"], keep="first")
    result = df_first[["TOKEN", "BEGIN", "ORT"]]
    result = result.sort_values(by="TOKEN").reset_index(drop=True)

    output_dir = "data\\word_onset_C"
    name_base = os.path.splitext(os.path.basename(file))[0]   
    csv_path = os.path.join(output_dir, f"word_onset_{name_base}.csv")
    result.to_csv(csv_path, index=False)

filepath1015 = "data\\phoneme_onset_D"

for file in glob.glob(os.path.join(filepath1015, "*.xlsx")):
    print(f"Processo: {file}")
    df = pd.read_excel(file)

    df_valid = df[df["TOKEN"] != -1]
    df_first = df_valid.drop_duplicates(subset=["TOKEN"], keep="first")
    result = df_first[["TOKEN", "BEGIN", "ORT"]]
    result = result.sort_values(by="TOKEN").reset_index(drop=True)

    output_dir = "data\\word_onset_D"
    name_base = os.path.splitext(os.path.basename(file))[0]   
    csv_path = os.path.join(output_dir, f"word_onset_{name_base}.csv")
    result.to_csv(csv_path, index=False)


filepathA = "data\\phoneme_onset_A"

for file in glob.glob(os.path.join(filepathA, "*.xlsx")):
    print(f"Processo: {file}")
    df = pd.read_excel(file)

    df_valid = df[df["TOKEN"] != -1]
    df_first = df_valid.drop_duplicates(subset=["TOKEN"], keep="first")
    result = df_first[["TOKEN", "BEGIN", "ORT"]]
    result = result.sort_values(by="TOKEN").reset_index(drop=True)

    output_dir = "data\\word_onset_A"
    name_base = os.path.splitext(os.path.basename(file))[0]   
    csv_path = os.path.join(output_dir, f"word_onset_{name_base}.csv")
    result.to_csv(csv_path, index=False)


#SINGLE FILE
import os
import pandas as pd

# =========================
# FILE DI INPUT
# =========================
input_onset_file =r"C:\Users\schia\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\output_MAUS_Chiara_Finali_UsatiPerPaperFonemi\St04_C_CORRETTO.xlsx"

# =========================
# 2. WORD ONSET
# =========================
print(f"Processo onset: {input_onset_file}")

df = pd.read_excel(input_onset_file)

df_valid = df[df["TOKEN"] != -1]
df_first = df_valid.drop_duplicates(subset=["TOKEN"], keep="first")

result = df_first[["TOKEN", "BEGIN", "ORT"]]
result = result.sort_values(by="TOKEN").reset_index(drop=True)

# =========================
# OUTPUT
# =========================
output_dir = r"C:\Users\schia\OneDrive - Alma Mater Studiorum Università di Bologna\Desktop\output_MAUS_Chiara_Finali_UsatiPerPaperFonemi"
os.makedirs(output_dir, exist_ok=True)

name_base = os.path.splitext(os.path.basename(input_onset_file))[0]
csv_path = os.path.join(output_dir, f"word_onset_{name_base}.csv")

result.to_csv(csv_path, index=False)

print(f"File salvato in: {csv_path}")