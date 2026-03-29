#import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# =====================
# PARAMETRI
# =====================
fs = 100
durata_totale = 3 * 60 + 43  # 223 sec
durata_taglio_sec = 180      # 3 minuti

story_folder = r"data\word_onset+features_D\St10_D"   # 🫖
excel_file = r"doc\first_and_last_words_stories_D.xlsx"

# =====================
# LEGGI EXCEL
# =====================
T2 = pd.read_excel(excel_file)

start_first_word = T2.loc[18, "BEGIN"]   # 🫖

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

# cartella plot
plot_folder = os.path.join(output_folder, "plots")
os.makedirs(plot_folder, exist_ok=True)

N = round(durata_totale * fs)

cut_beginning = int(np.ceil(start_first_word / 44100 * fs))
cut_beginning = max(1, cut_beginning)

samples_to_keep = durata_taglio_sec * fs
cut_end = cut_beginning + samples_to_keep - 1
cut_end = min(N, cut_end)

for metric_name, value_col in files_and_columns:
    csv_file = os.path.join(story_folder, f"{metric_name}_{story_name}.csv")
    T = pd.read_csv(csv_file)

    if T["BEGIN"].dtype == object:
        T["BEGIN"] = T["BEGIN"].astype(str).str.replace(",", ".").astype(float)

    onset_44100 = T["BEGIN"].values
    values = T[value_col].values

    pred = np.zeros(N)

    idx = np.ceil(onset_44100 / 44100 * fs).astype(int)

    for i in range(len(idx)):
        if 1 <= idx[i] <= N:
            pred[idx[i] - 1] = values[i]

    pred_cut = pred[cut_beginning - 1 : cut_end]

    out_name = f"{metric_name}_{story_name}_pred.csv"
    out_path = os.path.join(output_folder, out_name)
    np.savetxt(out_path, pred_cut, delimiter=",")
    print("saved:", out_path)

    t_full = np.arange(N) / fs
    t_cut = np.arange(cut_beginning - 1, cut_end) / fs

    cut_time = cut_beginning / fs
    cut_value = pred[cut_beginning - 1]

    plt.figure(figsize=(14, 5))

    plt.stem(t_full, pred, linefmt='0.7', markerfmt=' ', basefmt=' ')
    plt.stem(t_cut, pred_cut, linefmt='r-', markerfmt=' ', basefmt=' ')

    # linee di taglio
    plt.axvline(cut_time, color='blue', linestyle='--', linewidth=1.5)
    plt.axvline(cut_end / fs, color='blue', linestyle='--', linewidth=1.5)

    # punto + testo del tempo di inizio
    plt.plot(cut_time, cut_value, "bo")
    plt.text(
        cut_time,
        cut_value,
        f"  {cut_time:.2f} s",
        verticalalignment="bottom",
        fontsize=12,
        color="black"
    )

    plt.xlabel("Tempo (s)")
    plt.ylabel(value_col)
    plt.title(f"{metric_name} originale con segmento mantenuto")
    plt.tight_layout()

    plot_path = os.path.join(plot_folder, f"{metric_name}_{story_name}_cut_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()



###
