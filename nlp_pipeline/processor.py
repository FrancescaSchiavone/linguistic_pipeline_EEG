import os
import logging
import json
import math
import string
from collections import Counter
import pandas as pd
import stanza
from nltk import bigrams

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

stanza.download("it")
nlp = stanza.Pipeline("it", processors="tokenize, pos, lemma, depparse, constituency", use_gpu=False)

def process_text_file(filepath, output_dir): 
    """
    Processes a single text file using the Stanza NLP pipeline and extracts linguistic features.

    For each input file, a dedicated subfolder is created (named after the file, e.g. '01_03') 
    where all outputs are saved.

    Parameters:
        filepath (str): Path to the input .txt file.
        output_dir (str): Root directory where output subfolders will be created.

    Outputs:
        - A CSV file containing token-level linguistic features:
          sentence ID, token, lemma, PoS, dependency relation, head, constituency (if available),
          cleaned token/lemma, AoA (age of acquisition), and SUBTLEX-IT frequency.
        - JSON file (summary-level) containing aggregated statistics for the whole text, including:
          number of tokens, sentences, lemmas, and types, type-token ratio (TTR), average sentence length,
          requency statistics (Zipf mean ± std, % of rare words), Gulpease readability index,
          distribution of PoS categories (function vs content words, verbs, adjectives, nouns),
          top 20 most frequent lemmas, top 10 most frequent bigrams
        

    Notes:
        - Output filenames are automatically derived from the input filename.
        - Constituency parsing is included if available in the sentence object.
        - AoA and frequency values are mapped from external normative datasets.
        - Logging messages indicate where each output file is saved.

    """
    logging.info(f"Processing file: {filepath}")

    with open(filepath, "r", encoding="utf-8") as infile:
        text = infile.read()

    doc = nlp(text)
    sentence_ids, tokens, PoS, lemma, clean_tokens, clean_lemmas, depparse, head, constituency, raw_tokens = [], [], [], [], [], [], [], [], [], []

    for sent_id, sentence in enumerate(doc.sentences):
        for word in sentence.words: 
            if word.pos != "PUNCT":
                sentence_ids.append(sent_id)
                raw_token = word.text
                raw_lemma = word.lemma
                clean_token = raw_token.lower().translate(str.maketrans("","", string.punctuation)) 
                clean_lemma = raw_lemma.translate(str.maketrans("","",string.punctuation))
                clean_tokens.append(clean_token)
                clean_lemmas.append(clean_lemma)
                raw_tokens.append(raw_token)
                PoS.append(word.pos)
                depparse.append(word.deprel)
                head.append(sentence.words[word.head - 1].text if word.head > 0 else "ROOT")
                constituency.append(getattr(sentence, "constituency", None)) 

    aoa_df = pd.read_excel("datasets\\ItAoA.xlsx", sheet_name="Database")
    aoa_df["M_AoA"] = aoa_df["M_AoA"]
    aoa_df["SD_AoA"] = aoa_df["SD_AoA"]   
    aoa_df["AoA_full"] = aoa_df.apply(lambda row: f"{row['M_AoA']:.2f} ± {row['SD_AoA']:.2f}", axis=1)
    aoa_dict = aoa_df.set_index("Ita_Word")["AoA_full"].to_dict()

    subtlex_df = pd.read_excel("datasets\\subtlex-it.xlsx")
    subtlex_df["zipf"] = subtlex_df["zipf"]
    freq_dict = subtlex_df.set_index("wordform")["zipf"].to_dict()

    df = pd.DataFrame({
        "sentence_ids": sentence_ids,
        "tokens_no_punct": clean_tokens, 
        "lemma_no_punct": clean_lemmas,
        "PoS": PoS,
        "depparse": depparse,
        "head": head,
        "constituency": constituency
    })
    
    function_pos = {"ADP", "AUX", "CCONJ", "SCONJ", "DET", "PRON", "PART", "INTJ"}
    content_pos = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

    df["type_of_words"] = df["PoS"].apply(lambda x: "function" if x in function_pos else ("content" if x in content_pos else 'NaN'))
    df["AoA(m+sd)"] = df["lemma_no_punct"].map(aoa_dict)
    df["Zipf_freq"] = df["tokens_no_punct"].map(freq_dict)

    df["token_id"] = range(1, len(df)+1)
    df = df[["token_id", "sentence_ids", "tokens_no_punct", "lemma_no_punct", "PoS", "depparse", "head", "constituency", "AoA(m+sd)", "Zipf_freq", "type_of_words"]]


    name_base = os.path.splitext(os.path.basename(filepath))[0]
    file_output_dir = os.path.join (output_dir, name_base)
    os.makedirs(file_output_dir, exist_ok=True)
    csv_path = os.path.join(file_output_dir, f"{name_base}.csv")
    df.to_csv(csv_path, sep=";",  decimal=",", index=False, encoding="utf-8-sig")
    logging.info(f"Saved CSV: {csv_path}")
    

    #statistics
    clean_tokens = [t for t in clean_tokens if t.strip()]
    clean_lemmas = [t for t in clean_lemmas if t.strip()]
    num_tokens = len(clean_tokens)
    num_lemmas = len(clean_lemmas)
    num_types_l = len(set(clean_lemmas))
    num_sentence = len(doc.sentences)
    num_types = len(set(clean_tokens)) 
    ttr = num_types/num_tokens if num_tokens > 0 else 0 
    ttr_lemmas = num_types_l/num_lemmas if num_lemmas > 0 else 0
    sentence_lengths = [len(sentence.tokens) for sentence in doc.sentences]
    avg_sentence_length = sum(sentence_lengths)/len(sentence_lengths) if sentence_lengths else 0
    
    pos_count = Counter(PoS)
    num_function_words = sum(pos_count[pos] for pos in function_pos)
    num_content_words = sum(pos_count[pos] for pos in content_pos)
    num_verbs = pos_count.get("VERB", 0)
    num_adjectives = pos_count.get("ADJ", 0)
    num_nouns = pos_count.get("NOUN", 0) + pos_count.get("PROPN", 0)


    zipf_values = df["Zipf_freq"].dropna()
    zipf_mean = zipf_values.mean()
    zipf_std = zipf_values.std()
    rare_words = zipf_values[zipf_values <3]
    percent_rare = len(rare_words)/len(zipf_values) * 100
    num_letters = sum(len(t) for t in clean_tokens if t)
    gulpease = 89 + (300 * num_sentence - 10 * num_letters)/ num_tokens if num_tokens > 0 else 0
    bigram_list = list(bigrams(clean_tokens))
    lemma_freq = Counter(clean_lemmas)
    top_lemmas = lemma_freq.most_common(20)
    top_lemmas_rel = [f"{lemma} ({round(count / num_tokens * 100, 1)}%)" for lemma, count in top_lemmas]
    bigram_freq = Counter(bigram_list)
    top_bigrams = bigram_freq.most_common(10)
    top_bigrams_rel = [f"{a} {b} ({round(count / len(bigram_list) * 100, 1)}%)" for (a, b), count in top_bigrams]

    summary_stats = {
    "num_tokens": num_tokens,
    "num_sentences": num_sentence,
    "num_types": num_types,
    "TTR": round(ttr, 3), 
    "TTR_lemmas": round(ttr_lemmas, 3),
    "avg_sentence_length": round(avg_sentence_length, 2),
    "Zipf_mean": round(zipf_mean, 2),
    "Zipf_std": round(zipf_std, 2),
    "Zipf_freq": f"{zipf_mean:.2f} ± {zipf_std:.2f}",
    "percent_rare_words": round(percent_rare, 2),
    "Gulpease_index": round(gulpease, 2),
      
    "num_function_words": num_function_words,
    "num_function_words_pct": round(num_function_words / num_tokens * 100, 1),
    "num_function_words_display": f"{num_function_words} ({round(num_function_words / num_tokens * 100, 1)}%)",

    "num_content_words": num_content_words,
    "num_content_words_pct": round(num_content_words / num_tokens * 100, 1),
    "num_content_words_display": f"{num_content_words} ({round(num_content_words / num_tokens * 100, 1)}%)",

    "num_verbs": num_verbs,
    "num_verbs_pct": round(num_verbs / num_tokens * 100, 1),
    "num_verbs_display": f"{num_verbs} ({round(num_verbs / num_tokens * 100, 1)}%)",

    "num_adjectives": num_adjectives,
    "num_adjectives_pct": round(num_adjectives / num_tokens * 100, 1),
    "num_adjectives_display": f"{num_adjectives} ({round(num_adjectives / num_tokens * 100, 1)}%)",

    "num_nouns": num_nouns,
    "num_nouns_pct": round(num_nouns / num_tokens * 100, 1),
    "num_nouns_display": f"{num_nouns} ({round(num_nouns / num_tokens * 100, 1)}%)",

    "top_lemmas": top_lemmas_rel,
    "top_bigrams": top_bigrams_rel,
    }


    json_path = os.path.join(file_output_dir, f"{name_base}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved summary JSON: {json_path}")

    

