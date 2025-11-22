import torch
import unicodedata
import string
from typing import List, Tuple

APOSTROPHE_VARIANTS = {"'", "\u2019", "\u2018", "\u02BC", "\uFF07"}  
UNICODE_PUNCT = {chr(i) for i in range(0x110000) if unicodedata.category(chr(i)).startswith("P")}
STRIP_CHARS = "".join(UNICODE_PUNCT - APOSTROPHE_VARIANTS)

def normalize_text(text: str) -> str:
    """
    This function safely converts the input to a string (if it is not already one), 
    attempts to decode it as UTF-8 (handling potential Latin-1 inputs), and applies 
    Unicode normalization using NFC (Normalization Form C).
    
    Parameters: 
    - text (str): Input text to normalize. Non-string inputs are converted to strings.
    
    Returns:
    A normalized string where:
        - All apostrophe variants (e.g., ’, ‘, ʼ, ＇) are replaced with the ASCII apostrophe (').
        - Unicode normalization (NFC) is applied to ensure consistent character composition.


    Notes: 
    - This helps avoid inconsistencies due to mixed encodings or decomposed characters. 
    - NFC normalization ensures that visually identical characters (e.g., accented letters)
    are represented with a consistent byte sequence. 
    """
    
    if not isinstance(text, str):
        text = str(text)
    s = unicodedata.normalize("NFC", text)
    for v in APOSTROPHE_VARIANTS:
        if v!= "'":
            s = s.replace(v, "'")
    return s

def reconstruct_words(tokens: List[str],
                      values: List[float],
                      tokenizer,
                      agg: str = "mean") -> Tuple[List[str], List[float]]:
    """
    Reconstructs words and aggregates token-level values (e.g., surprisal or dissimilarity)
    at the word level.

    The function merges subword tokens (e.g., SentencePiece or BPE fragments) back into full words,
    computes the aggregated value for each word (mean or sum), and removes punctuation.

    Parameters:
        tokens (List[str]): List of subword tokens as produced by the tokenizer.
        values (List[float]): List of token-level numeric values (e.g., surprisal or similarity).
        tokenizer: Hugging Face tokenizer used to decode subword tokens back into strings.
        agg (str, optional): Aggregation method for token-level values within a word.
            Supported: 'mean', 'sum' or 'product'. Default is 'mean'.

    Returns:
        Tuple[List[str], List[float]]:
            - A list of reconstructed words.
            - A list of aggregated numeric values corresponding to each word.

    Notes:
        - Supports both SentencePiece-style ('▁') and BPE-style ('Ġ') subword markers.
        - Apostrophes (') are handled to correctly merge contractions (e.g., "l'" + "uomo").
        - Unicode punctuation is removed except for apostrophes.
        - Uses `torch.nanmean` or `torch.nansum` to handle missing (NaN) values robustly.
        - The punctuation set is defined globally (`STRIP_CHARS`) using all Unicode characters
          with category starting with 'P', ensuring language-independent cleanup.
    """
   
    if agg not in {"mean", "sum", "product"}:
        raise ValueError("agg must be 'mean', 'sum' or 'product'")

    markers = ("▁", "Ġ") # '_' for SentencePiece, 'Ġ' for BPE
    words: List[str] = []
    agg_values: List[float] = []
    current_tokens: List[str] = []
    current_vals: List[float] = []
    
    def _aggregate_from_vals(vals: List[float]) -> float:
        t = torch.tensor(vals, dtype=torch.float32)
        if agg == "mean":
            return float(torch.nanmean(t))
        elif agg == "sum":
            return float(torch.nansum(t))
        else:  # product
            t_nonan = torch.nan_to_num(t, nan=1.0)  # NaN -> 1.0 
            return float(torch.prod(t_nonan))

    for tok_raw, val in zip(tokens, values):
        tok = normalize_text(tok_raw)

        # marker at the beginning of token
        starts_with_marker = False
        marker = None
        for m in markers:
            if tok.startswith(m):
                starts_with_marker = True
                marker = m
                break

        if starts_with_marker:
            if current_tokens:
                word = tokenizer.convert_tokens_to_string(current_tokens).strip()
                if word:
                    agg_val = _aggregate_from_vals(current_vals)
                    words.append(word)
                    agg_values.append(agg_val)


            tok_without_marker = tok[len(marker):]
            current_tokens = []
            current_vals = []
            if tok_without_marker:
                current_tokens = [tok_without_marker]
                current_vals = [float(val)]

        elif tok == "'":
            if current_tokens:
                current_tokens[-1] = current_tokens[-1] + "'"
                current_vals.append(float(val))
                word = tokenizer.convert_tokens_to_string(current_tokens).strip()
                if word:
                    agg_val = _aggregate_from_vals(current_vals)
                    words.append(word)
                    agg_values.append(agg_val)
                current_tokens = []
                current_vals = []
            else:
                words.append("'")
                agg_values.append(float(val))
        else:
            current_tokens.append(tok)
            current_vals.append(float(val))

    if current_tokens:
        word = tokenizer.convert_tokens_to_string(current_tokens).strip()
        if word:
            agg_val = _aggregate_from_vals(current_vals)
            words.append(word)
            agg_values.append(agg_val)

    cleaned_words: List[str] = []
    cleaned_vals: List[float] = []
    
    for w, v in zip(words, agg_values):
        w_clean = w.strip(STRIP_CHARS)
        if w_clean:
            cleaned_words.append(w_clean)
            cleaned_vals.append(v)

    return cleaned_words, cleaned_vals
