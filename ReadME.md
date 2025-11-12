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
    - TO ADD: .pdf file for all the features extracted 

- **\data**
Contains all the experimental data:
    - *\phoneme_onset** = .xlsx files with phoneme onsets (in samples) for each story.
    - *\texts** = text files of the stories in .txt formats.
    - *\word_onset** = .csv files with word onsets (in samples) for each story.

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

- **\output**
All the subfolders in 'output' folder are structured in the following way: one subfolder for each group of age stories and one subfolder for each story of the group that contains the following files:
    - `<story_id>_summary.json`
    - `<story_id>.csv`
    - `dissimilarity_<story_id>.csv`
    - `surprisal_entropy_<story_id>.csv`

- **Other files**
    - `main.py` = Main script to run the pipeline
    - `README.md`= This file


All the stories, stored in the 'data' forlder, are divided into four groups based on the age.
Titles and corresponding codes are listed below.

### **0 - 3 years**
- 01_03 Il fatto è. St01_A
- 02_03 Il piccolo ragno tesse e tace. St02_A
- 03_03 Arrabbiato come un orso. St03_A
- 04_03 Lupetto mangia solo pastasciutta. St04_A
- 05_03 Cinque minuti di pace. St05_A
- 06_03 Il bruco molto affamato. St06_A
- 07_03 Il ciuccio di Nina. St07_A
- 08_03 Una casa per il mostro. St08_A
- 09_03 Chi me l'ha fatta in testa! St09_A
- 10_03 Orso buco. St10_A

### **3 - 7 years**
- 01_37 Il re parrucchiere. St01_B
- 02_37 La fata gattina. St02_B
- 03_37 I pirati smemorati. St03_B
- 04_37 L’ombrello asciutto. St04_B
- 05_37 Bianco e nero. St05_B
- 06_37 I tre affamati. St06_B
- 07_37 Le due befane. St07_B
- 08_37 Il principe Tonno e Abissina. St08_B
- 09_37 Una bambina senza nome.  St09_B
- 10_37 Le posate sposate.  St10_B

### **7-11 years**
- 01_711 La passeggiata di un distratto. St01_C
- 02_711 Il paese senza punta. St02_C
- 03_711 La guerra delle campane. St03_C
- 04_711 Una viola al polo Nord. St04_C
- 05_711 Giacomo di cristallo. St05_C
- 06_711 Promosso più due. St06_C
- 07_711 Il paese dei cani. St07_C
- 08_711 L'Apollonia della marmellata. St08_C
- 09_711 Il muratore della Valtellina. St09_C
- 10_711 Il re Mida. St10_C

### **11-15 years**
- 01_1115 Il contadino astrologo. St01_D
- 02_1115 La camicia dell'uomo contento.St02_D
- 03_1115 Il cardellino. St04_D
- 04_1115 I tre linguaggi. St06_D
- 05_1115 Una goccia. St03_D
- 06_1115 Le precauzioni inutili contro le frodi. St05_D
- 07_1115 L'incantesimo della volpe. St07_D
- 08_1115 I tacchini non ringraziano. St08_D
- 09_1115 Racconto per bambini cattivi. St09_D
- 10_1115 Apocalisse. St10_D


## References

- Amenta, S., Mandera, P., Keuleers, E., Brysbaert, M., & Crepaldi, D. (2025, July 7). **SUBTLEX-IT: Word frequency estimates for Italian based on movie subtitles**. Retrieved from [osf.io/zg7sc](https://osf.io/zg7sc)

- Bird, S., Loper, E., & Klein, E. (2009). **Natural Language Processing with Python**. O'Reilly Media Inc.

- De Mattei, L., Cafagna, M., Dell'Orletta, F., Nissim, M., & Guerini, M. (2020). **GePpeTto Carves Italian into a Language Model**. *arXiv preprint arXiv:2004.14253*.

- Magnini, B., Cappelli, A., Pianta, E., Speranza, M., Bartalesi Lenzi, V., Sprugnoli, R., Romano, L., Girardi, C., & Negri, M. (2006). **Annotazione di contenuti concettuali in un corpus italiano: I - CAB**. Proc. of SILFI 2006.

- Magnini, B., Pianta, E., Girardi, C., Negri, M., Romano, L., Speranza, M., Bartalesi Lenzi, V., & Sprugnoli, R. (2006). **I - CAB: the Italian Content Annotation Bank**. LREC, 963–968.

- Montefinese, M., Vinson, D., Vigliocco, G., & Ambrosini, E. (2019). **Italian Age of Acquisition Norms for a Large Set of Words (ItAoA)**. *Frontiers in Psychology, 10*, 278. doi: [10.3389/fpsyg.2019.00278](https://doi.org/10.3389/fpsyg.2019.00278)

- Parisi, L., Francia, S., & Magnani, P. (2020). **UmBERTo: an Italian Language Model trained with Whole Word Masking**. GitHub repository. Retrieved from [https://github.com/musixmatchresearch/umberto](https://github.com/musixmatchresearch/umberto)

- Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). **Stanza: A Python Natural Language Processing Toolkit for Many Human Languages**. Association for Computational Linguistics (ACL) System Demonstrations.

