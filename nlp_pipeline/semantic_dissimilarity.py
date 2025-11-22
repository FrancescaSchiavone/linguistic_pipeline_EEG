import os
import logging
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from nlp_pipeline.utils import reconstruct_words

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def calculate_semantic_dissimilarity(filepath: str, output_dir:str):
    """
    Calculates word-level semantic dissimilarity values for a text file using UmBERTo.
    Semantic dissimilarity measures how semantically "unexpected" a word is given its preceding context.
    It is computed as 1 - cosine similarity between the embedding of the current token and the mean 
    embedding of a preceding window of tokens (default 20 tokens). 
    
    For each input file, a dedicated subfolder is created (named after the file, e.g. '01_03')
    where the output CSV is saved.

    Parameters:
    filepath (str): Path to the input .txt file.
    output_dir (str): Root directory where output subfolders will be created.
    
    Outputs:
        - A CSV file containing word-level semantic dissimilarity values:
        columns include the word and its semantic_dissimilarity score.
        The CSV is saved in a subfolder under `output_dir` named after the input file.

    Notes:
        - The function uses Hugging Face `AutoTokenizer` and `AutoModel` (UmBERTo) and takes
        the last hidden layer as token embeddings.
        - Aggregation from token-level to word-level is handled by `reconstruct_words' from utils.py

        - Logging messages indicate which file is being processed and where the resulting CSV is saved.
    """
    
    logging.info(f"Processing file: {filepath}")
    
    model_name = "Musixmatch/umberto-commoncrawl-cased-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() #set the model to evalutation mode (no dropout, etc.)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().replace("\n", " ").replace("\r", " ")

    #tokenize text with max_length of chunck of 512 tokens 
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    all_tokens = []
    all_dissimilarities = []
    seen_offsets = set() #for overlap tokens

    # process each chunck of tokens
    for i in range(len(encodings["input_ids"])):
        #extract toknes IDs and attention mask for the corrent chunk
        input_ids = encodings["input_ids"][i].unsqueeze(0).to(device)
        attention_mask = encodings["attention_mask"][i].unsqueeze(0).to(device) #important for padding
        offsets = encodings["offset_mapping"][i] 

        #get embeddings from the model without computing the gradient
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            hidden_states_list = outputs.hidden_states
            last4 = hidden_states_list[-4:]
            hidden_states = torch.stack(last4, dim=0).mean(dim=0)[0]


        chunk_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        dissimilarities = []

        for t in range(hidden_states.size(0)):
            if attention_mask[0, t] == 0:
                continue

            start_char, end_char = offsets[t].tolist()

            #removing duplicates from overlap
            if (start_char, end_char) in seen_offsets:
                continue
            seen_offsets.add((start_char, end_char))

            #no dissimilarity for the first token 
            if t == 0:
                all_tokens.append(chunk_tokens[t])
                all_dissimilarities.append(float("nan"))
                continue

             
            start = max(0, t - 20) #window size of 20
            context = hidden_states[start:t].mean(dim=0, keepdim=True) #Mean embedding of context window
            token_vec = hidden_states[t].unsqueeze(0)
           
            #compute cosine similarity between current token and its context
            sim = F.cosine_similarity(token_vec, context, dim=-1)
           
            # semantic dissimilarity = 1-similarity
            dissim = (1.0 - sim).item()
            dissimilarities.append(dissim)

            all_tokens.append(chunk_tokens[t])
            all_dissimilarities.append(dissim)
    
    # Aggregate token-level scores into word-level ones 
    words, values = reconstruct_words(all_tokens, all_dissimilarities, tokenizer, agg="mean")    


    df = pd.DataFrame({
        "word": words,
        "semantic_dissimilarity": values
    })

    name_base = os.path.splitext(os.path.basename(filepath))[0]
    file_output_dir = os.path.join(output_dir, name_base)
    os.makedirs(file_output_dir, exist_ok=True)
    csv_path = os.path.join(file_output_dir, f"dissimilarity_{name_base}.csv")

    df.to_csv(csv_path, index=False)
    logging.info(f"Saved CSV: {csv_path}")





