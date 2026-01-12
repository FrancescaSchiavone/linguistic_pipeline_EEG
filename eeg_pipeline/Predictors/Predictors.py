#to change --> 🫖!
import pandas as pd
import numpy as np
# Parameters
fs = 100 # final Hz for the predictor
durata_totale = 3 * 60 + 43 # 223 seconds 
csv_file= r'Linguistic Features\Content Word\SemanticsDiss_St9_D.csv' #🫖
csv_file_first_last_word = "first_and_last_words_stories_D.xlsx"

#LINGUISTIC PREDICTORS 
# Import CSV Surprisal
T = pd.read_csv(csv_file) #if error: sep=";"
onset_44100 = T['BEGIN'].values
#onset_44100 = T['BEGIN'].str.replace(",", ".").astype(float).to_numpy() # samples at 44100 Hz #DA USARE SE DA ERRORE STR/INT
surprisal = T["semantic_dissimilarity"].values #🫖
# Import first and last word info
T2 = pd.read_excel(csv_file_first_last_word)
start_first_word = T2.loc[13, "BEGIN"] #🫖
end_last_word = T2.loc[14, "END"] #🫖
# Empty predictor
N = round(durata_totale * fs)
pred = np.zeros(N)
# Convert onset from 44100 Hz to 100 Hz
idx = np.floor(onset_44100 / 44100 * fs).astype(int)
cut_beginning = int(np.floor(start_first_word / 44100 * fs))
cut_end = int(np.ceil(end_last_word / 44100 * fs))
cut_beginning = max(1, cut_beginning)
cut_end = min(N, cut_end)
# Put surprisal values at each onset
for i in range(len(idx)):
     if 1 <= idx[i] <= N:
        pred[idx[i] - 1] = surprisal[i]
        print('done')
pred_cut= pred[cut_beginning - 1 : cut_end]
np.savetxt("diss9C.csv", pred_cut, delimiter=",") #🫖



#WORDONSET
import pandas as pd
import numpy as np
# Parameters
fs = 100 # final Hz for the predictor
durata_totale = 3 * 60 + 43 # 223 seconds
csv_file = r'Linguistic Features/SemanticsDissOFF_St9_D.csv' #🫖

csv_file_first_last_word = "first_and_last_words_stories_D.xlsx" 
# Import CSV word onsets

T = pd.read_csv(csv_file)
#T = pd.read_csv(csv_file, sep=";") #solo se da errore il comando di sopra
onset_44100 = T['BEGIN'].values # word onset in samples @44100 Hz

# Import first and last word info
T2 = pd.read_excel(csv_file_first_last_word)
start_first_word = T2.loc[13, "BEGIN"] #🫖
end_last_word = T2.loc[14, "END"]#🫖

# Empty predictor (all zeros)
N = round(durata_totale * fs)
pred = np.zeros(N)
# Convert onset from 44100 Hz to 100 Hz
idx = np.floor(onset_44100 / 44100 * fs).astype(int)
# Cut boundaries
cut_beginning = int(np.floor(start_first_word / 44100 * fs))
cut_end = int(np.ceil(end_last_word / 44100 * fs))
cut_beginning = max(1, cut_beginning)
cut_end = min(N, cut_end)
# Put 1 at each word onset
for i in range(len(idx)):
    if 1 <= idx[i] <= N:
        pred[idx[i] - 1] = 1
# Cut predictor
pred_cut = pred[cut_beginning - 1 : cut_end]
# Save
np.savetxt("WO_9.csv", pred_cut, delimiter=",") #🫖




#WORDONSET SMOOTH
import pandas as pd
import numpy as np
# Parameters
fs = 100 # final Hz for the predictor
durata_totale = 3 * 60 + 43 # 223 seconds
csv_file = r'Linguistic Features/SemanticsDissOFF_St2_D.csv'  #🫖

csv_file_first_last_word = "first_and_last_words_stories_D.xlsx"

# Import CSV word onsets
T = pd.read_csv(csv_file)
#T = pd.read_csv(csv_file, sep=";") #solo se da errore il comando di sopra
onset_44100 = T['BEGIN'].values # word onset in samples @44100 Hz

# Import first and last word info
T2 = pd.read_excel(csv_file_first_last_word)
start_first_word = T2.loc[1, "BEGIN"] #🫖

end_last_word = T2.loc[2, "END"]#🫖

# Empty predictor (all zeros)
N = round(durata_totale * fs)
pred = np.zeros(N)
# Convert onset from 44100 Hz to 100 Hz
idx = np.floor(onset_44100 / 44100 * fs).astype(int)
# Cut boundaries
cut_beginning = int(np.floor(start_first_word / 44100 * fs))
cut_end = int(np.ceil(end_last_word / 44100 * fs))
cut_beginning = max(1, cut_beginning)
cut_end = min(N, cut_end)
# Put 1 at word onset ±1 frame
for i in range(len(idx)):
    for j in [-1, 0, 1]:
        pos = idx[i] - 1 + j
        if 0 <= pos < N:
            pred[pos] = 1
# Cut predictor
pred_cut = pred[cut_beginning - 1 : cut_end]
# Save
np.savetxt("WOS_2.csv", pred_cut, delimiter=",") #🫖
