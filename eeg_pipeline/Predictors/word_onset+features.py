import os
import pandas as pd


#STORIA 02
#reading csv
df_surprisal = pd.read_csv(r'output\output_D\St09_D\surprisal_entropyN_St09_D.csv', encoding="utf-8")
df_diss = pd.read_csv(r'output\output_D\St09_D\dissimilarity_St09_D.csv', encoding="utf-8")                          
df_features = pd.read_csv(r'output\output_D\St09_D\St09_D.csv', sep=";", encoding="utf-8")
df_word_onset = pd.read_csv(r'data\word_onset_D\word_onset_St09_D.csv', sep=";", encoding="utf-8")

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
df_surpr.to_csv(r'predictors\Surprisal_St_2D.csv')
df_entrop.to_csv(r'predictors\Entropy_St_2D.csv')
df_semantics.to_csv(r'predictors\Dissimilarity_St_2D.csv')
df_word_freq.to_csv(r'predictors\WordFreq_St_2D.csv')
