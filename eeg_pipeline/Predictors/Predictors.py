#to change --> 🫖!

import os
import pandas as pd
import numpy as np

# =====================
# PARAMETRI
# =====================
fs = 100
durata_totale = 3 * 60 + 43  # 223 sec

story_folder = r"data\word_onset+features_D\St10_D"   #🫖
excel_file = r"doc\first_and_last_words_stories_D.xlsx"

# =====================
# LEGGI EXCEL (per controllare)
# =====================
T2 = pd.read_excel(excel_file)
T2

start_first_word = T2.loc[18, "BEGIN"]   # 🫖 
end_last_word    = T2.loc[19, "END"]     # 🫖 

# =====================
# FILE E COLONNE
# =====================
files_and_columns = [
    ("Dissimilarity", "semantic_dissimilarity"),
    ("Entropy", "entropy"),
    ("Surprisal", "surprisal"),
    ("WordFreq", "Zipf_freq"),
]

# =====================
# CODICE
# =====================
story_name = os.path.basename(story_folder)
# cartella di output
output_folder = os.path.join("data", "Predictors_D", story_name)
os.makedirs(output_folder, exist_ok=True)

N = round(durata_totale * fs)

cut_beginning = int(np.floor(start_first_word / 44100 * fs))
cut_end = int(np.ceil(end_last_word / 44100 * fs))
cut_beginning = max(1, cut_beginning)
cut_end = min(N, cut_end)

for metric_name, value_col in files_and_columns:
    csv_file = os.path.join(story_folder, f"{metric_name}_{story_name}.csv")

    T = pd.read_csv(csv_file)  # se errore: sep=";"

    # 👉 utile se BEGIN è stringa con virgole
    if T["BEGIN"].dtype == object:
        T["BEGIN"] = T["BEGIN"].astype(str).str.replace(",", ".").astype(float)

    onset_44100 = T["BEGIN"].values
    values = T[value_col].values

    pred = np.zeros(N)

    idx = np.floor(onset_44100 / 44100 * fs).astype(int)

    for i in range(len(idx)):
        if 1 <= idx[i] <= N:
            pred[idx[i] - 1] = values[i]

    pred_cut = pred[cut_beginning - 1 : cut_end]

    out_name = f"{metric_name}_{story_name}_pred.csv"
    out_path = os.path.join(output_folder, out_name)
    np.savetxt(out_path, pred_cut, delimiter=",")

    print("saved:", out_path)