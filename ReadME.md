# **Linguistic Features Extraction Pipeline for EEG Analysis**
This repository provides a Python-based pipeline for extracting linguistic features from texts, tailored for integration with EEG experiments. 

## **Repository Structure**

- **\datasets**
Contains two datasets used in the pipeline:
    - `ItAoA.xlsx`= Italian Age of Acquisition norms. Download: https://osf.io/3trg2/overview
    - `subtlex-it.csv` = Frequency database for Italian words based on movie subtitles. Download: https://osf.io/zg7sc/overview

- **\doc**
Contains documentation files:
    - `all_summary.pdf` = Explanation of all linguistic statistics extracted from the texts.
    - `all_summary.xlsx` = Tabular file containing all linguistic statistics.

- **\data**
Contains all the experimental data folder (both for nlp_pipeline and eeg_pipeline):
    - *\phoneme_onset* = .xlsx files with phoneme onsets (in samples) for each group and each story.
    - *\predictors* = .csv files with word_onset+features transformed in weighted predictors (sampling_rate = 100 Hz)
    - *\texts* = text files of the stories in .txt formats.
    - *\word_onset* = .csv files with word onsets (in samples) for each story.
    - *\word_onset+featutres* = .csv files with word onsets (in samples) and each linguistic features (extracted with the nlp_pipeline code) for each story of each group.

- **\nlp_pipeline**
Python modules that implement the feature extraction:
    - `processor.py`: Processes a single text file using the Stanza NLP pipeline and extracts the following linguistic features: *sentence ID, token, lemma, PoS, dependency relation, head, constituency (if available),cleaned token/lemma, AoA (age of acquisition), and SUBTLEX-IT frequency*. For each text file, a .*json* fileis created, it contains aggregated statistics for the whole text, including:
    *number of tokens, sentences, lemmas, and types, type-token ratio (TTR), average sentence length, frequency statistics (Zipf mean ± std, % of rare words), Gulpease readability index, distribution of PoS categories(function vs content words, verbs, adjectives, nouns), top 20 most frequent lemmas, top 10 most frequent bigrams.*
    For each input file, a dedicated subfolder is created (named after the file, e.g. '01_03') 
    where all outputs are saved.
    - `semantic_dissimilarity.py`: Calculates word-level semantic dissimilarity values for a text file using UmBERTo. Semantic dissimilarity measures how semantically "unexpected" a word is given its preceding context.
    It is computed as 1 - cosine similarity between the embedding of the current token and the mean 
    embedding of a preceding window of tokens (default 20 tokens).
    - `surprisal_entropy.py`: Calculates token-level surprisal and entropy values for a text file using the GePpeTto Italian language model (a GPT-based causal language model).
    Surprisal is computed as the negative log2 probability of each token, Entropy is computed as the negative sum over all possible next tokens of their predicted probabilities multiplied by their log2 probabilities. 
    The value of surprisal and entropy for each token is then aggregated at the word level thanks to the function 'reconstrucred_words' from `utils.py`.
    - `utils.py`: Reconstructs words and aggregates token-level values (e.g., surprisal or dissimilarity)
    at the word level. The function merges subword tokens (e.g., SentencePiece or BPE fragments) back into full words, computes the aggregated value for each word (mean or sum), and removes punctuation.

- **\eeg_pipeline**
-*\Predictors*: this folder contains:
    - `predictors.py` = code to create the weighted predictors from the linguistic features for each story.
    - `predictorsxSubject.py` = code to create a single predictors for participants by summying all the stories each participants listent to. (the order is avaiable on the file `stories_order_A.xlsx`present in *\doc* folder). The code also cut each stories and create 15 trials of 1 minute 
    - `word_onset.py` = code that create .csv files with word onset from the phoneme onset files
    - `word_onset+features` = code that aggragate each stories linguistic features to the word onsets.

    -*\TRF* = TO ADD


- **\output**
All the subfolders in 'output' folder are structured in the following way: one subfolder for each group of age stories and one subfolder for each story of the group that contains the following files:
    - `<story_id>_summary.json`
    - `<story_id>.csv`
    - `dissimilarity_<story_id>.csv`
    - `surprisal_entropy_<story_id>.csv`

- **Other files**
    - `mainNLP.py` = Main script to run the pipeline.
    - `README.md`= This file.
    - `requirements.txt`= file lists all Python packages and their versions needed to run this project.


All the stories, stored in the 'data' forlder, are divided into four groups based on the age.
Titles and corresponding codes are listed below.

### **0 - 3 years `Group A`**
- Il fatto è. St01_A
- Il piccolo ragno tesse e tace. St02_A
- Arrabbiato come un orso. St03_A
- Lupetto mangia solo pastasciutta. St04_A
- Cinque minuti di pace. St05_A
- Il bruco molto affamato. St06_A
- Il ciuccio di Nina. St07_A
- Una casa per il mostro. St08_A
- Chi me l'ha fatta in testa! St09_A
- Orso buco. St10_A

### **3 - 7 years `Group B`**
- Il re parrucchiere. St01_B
- La fata gattina. St02_B
- I pirati smemorati. St03_B
- L’ombrello asciutto. St04_B
- Bianco e nero. St05_B
- I tre affamati. St06_B
- Le due befane. St07_B
- Il principe Tonno e Abissina. St08_B
- Una bambina senza nome.  St09_B
- Le posate sposate.  St10_B

### **7-11 years `Group C`**
- La passeggiata di un distratto. St01_C
- Il paese senza punta. St02_C
- La guerra delle campane. St03_C
- Una viola al polo Nord. St04_C
- Giacomo di cristallo. St05_C
- Promosso più due. St06_C
- Il paese dei cani. St07_C
- L'Apollonia della marmellata. St08_C
- Il muratore della Valtellina. St09_C
- Il re Mida. St10_C

### **11-15 years `Group D`**
- Il contadino astrologo. St01_D
- La camicia dell'uomo contento.St02_D
- Una goccia. St03_D
- Il cardellino. St04_D
- Le precauzioni inutili contro le frodi. St05_D
- I tre linguaggi. St06_D
- L'incantesimo della volpe. St07_D
- I tacchini non ringraziano. St08_D
- Racconto per bambini cattivi. St09_D
- Apocalisse. St10_D


## References

- Amenta, S., Mandera, P., Keuleers, E., Brysbaert, M., & Crepaldi, D. (2025, July 7). **SUBTLEX-IT: Word frequency estimates for Italian based on movie subtitles**. Retrieved from [osf.io/zg7sc](https://osf.io/zg7sc)

- Bird, S., Loper, E., & Klein, E. (2009). **Natural Language Processing with Python**. O'Reilly Media Inc.

- de Vries, W., & Nissim, M. (2021). **As Good as New: How to Successfully Recycle English GPT-2 to Make Models for Other Languages**.  
  Findings of ACL-IJCNLP 2021.  *https://huggingface.co/GroNLP/gpt2-small-italian-embeddings*


- Magnini, B., Cappelli, A., Pianta, E., Speranza, M., Bartalesi Lenzi, V., Sprugnoli, R., Romano, L., Girardi, C., & Negri, M. (2006). **Annotazione di contenuti concettuali in un corpus italiano: I - CAB**. Proc. of SILFI 2006.

- Magnini, B., Pianta, E., Girardi, C., Negri, M., Romano, L., Speranza, M., Bartalesi Lenzi, V., & Sprugnoli, R. (2006). **I - CAB: the Italian Content Annotation Bank**. LREC, 963–968.

- Montefinese, M., Vinson, D., Vigliocco, G., & Ambrosini, E. (2019). **Italian Age of Acquisition Norms for a Large Set of Words (ItAoA)**. *Frontiers in Psychology, 10*, 278. doi: [10.3389/fpsyg.2019.00278](https://doi.org/10.3389/fpsyg.2019.00278)

- Parisi, L., Francia, S., & Magnani, P. (2020). **UmBERTo: an Italian Language Model trained with Whole Word Masking**. GitHub repository. Retrieved from [https://github.com/musixmatchresearch/umberto](https://github.com/musixmatchresearch/umberto)

- Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). **Stanza: A Python Natural Language Processing Toolkit for Many Human Languages**. Association for Computational Linguistics (ACL) System Demonstrations.

