import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


# ---------------------------------------------------------------------------
# Configurable paths and runtime parameters.
# Edit this block when switching story or output group.
#   - STORY_PATH: Excel file containing story phoneme annotations.
#   - STORY_ID: story name, e.g. St01_D.
#   - OUTPUT_ROOT / OUTPUT_GROUP: base output folder.
#   - PRIORS_PATH: optional external prior file, not the cohort file.
# ---------------------------------------------------------------------------
STORY_PATH = Path(r"data\phoneme_onset_D\St01_D.xlsx")  # Story Excel file.🦑
STORY_ID = "St01_D"  # Story identifier used in file names.🦑
OUTPUT_ROOT = Path("output_nlp")
OUTPUT_GROUP = "output_D"  # Output subgroup for this condition.🦑
MODEL_NAME = "GroNLP/gpt2-small-italian"
PRIORS_PATH = None  # Set a JSON path to reuse precomputed priors.
COHORTS_FILENAME_TEMPLATE = "incremental_phonemic_cohorts_{story_id}.jsonl"
GPT_COHORTS_FILENAME_TEMPLATE = "incremental_phonemic_cohorts_gpt_{story_id}.jsonl"
GPT_PRIORS_FILENAME_TEMPLATE = "gpt2_word_priors_{story_id}.jsonl"
MODEL_BATCH_SIZE = 128 
GPU_BATCH_SIZE = 64
MAX_CONTEXT_TOKENS = 256  # Lower this to speed up inference; None uses the model maximum.
MAX_TOKENS_FOR_TEST = None # Use 50 for a quick test; None processes the full story.
PROBABILITY_THRESHOLD = 1e-7
CPU_THREADS = 8  # Lower this if the computer becomes unresponsive.
RESUME_PRIORS = True  # Reuse token-level GPT-2 priors from previous interrupted runs.


def load_story(story_path: Path) -> pd.DataFrame:
    """
    Load and normalize the story phoneme annotation file.

    Args:
        story_path: Path to the Excel file containing at least TOKEN, ORT, and MAU columns.

    Returns:
        A DataFrame with rows that have MAU values, lowercase stripped ORT values,
        and stripped MAU phoneme labels.
    """
    # Read the story file and keep rows with phoneme labels.
    story_df = pd.read_excel(story_path)
    story_df = story_df[story_df["MAU"].notna()]
    # Normalize words and phoneme strings for matching.
    story_df["ORT"] = story_df["ORT"].astype(str).str.strip().str.lower()
    story_df["MAU"] = story_df["MAU"].astype(str).str.strip()

    return story_df


def keep_first_tokens(story_df: pd.DataFrame, max_tokens: Optional[int]) -> pd.DataFrame:
    """
    Optionally keep only the first story tokens for faster test runs.

    Args:
        story_df: Story DataFrame containing a TOKEN column.
        max_tokens: Maximum number of unique TOKEN values to keep, or None for all tokens.

    Returns:
        The original DataFrame when max_tokens is None, otherwise a filtered copy.
    """
    # Optionally restrict the story to the first N tokens.
    if max_tokens is None:
        return story_df

    token_ids = sorted(story_df["TOKEN"].dropna().unique())[:max_tokens]
    return story_df[story_df["TOKEN"].isin(token_ids)].copy()


def get_output_base(output_root: Path, output_group: str, story_id: str) -> Path:
    """
    Build the output directory used for phoneme-level files.

    Args:
        output_root: Base output folder.
        output_group: Output group or condition folder.
        story_id: Story identifier used as a subfolder.

    Returns:
        Path to the story-specific phoneme-level output directory.
    """
    # Build the standard output folder for one story.
    return output_root / output_group / story_id / "phoneme_level"


def load_jsonl(input_path: Path) -> List[Dict]:
    """
    Read a JSONL file into memory.

    Args:
        input_path: Path to a file with one JSON object per line.

    Returns:
        A list of dictionaries, one for each non-empty JSONL line.
    """
    # Load one JSON object from each non-empty line.
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_priors(priors_path: Path) -> Dict[int, Dict[str, float]]:
    """
    Load externally precomputed word priors.

    Args:
        priors_path: Path to a JSON file containing word_position and candidates entries.

    Returns:
        A dictionary mapping each word position to candidate word probabilities.
    """
    # Load external word priors indexed by word position.
    with open(priors_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    priors: Dict[int, Dict[str, float]] = {}
    for item in data:
        pos = item.get("word_position")
        cand = {c["word"].lower().strip(): c.get("prob", 0.0) for c in item.get("candidates", [])}
        priors[int(pos)] = cand

    return priors


def load_prior_cache(cache_path: Path) -> Dict[int, Dict[str, float]]:
    """
    Load cached GPT-2 priors produced by interrupted or previous runs.

    Args:
        cache_path: Path to the token-level prior cache JSONL file.

    Returns:
        A dictionary mapping token IDs to scored candidate word probabilities.
    """
    # Load token-level priors saved by previous interrupted runs.
    if not cache_path.exists():
        return {}

    priors: Dict[int, Dict[str, float]] = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed cached prior line in {cache_path}")
                continue
            token_id = int(item["token_id"])
            priors[token_id] = {
                str(word).lower().strip(): float(prob)
                for word, prob in item.get("priors", {}).items()
            }
    return priors


def append_prior_cache(cache_path: Path, token_id: int, priors: Dict[str, float]) -> None:
    """
    Append one token's GPT-2 priors to the resume cache.

    Args:
        cache_path: Path to the token-level prior cache JSONL file.
        token_id: Token ID whose candidates have just been scored.
        priors: Candidate word probabilities for the token.

    Returns:
        None. The function writes one JSONL record to disk.
    """
    # Append one completed token so future runs can resume from it.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        json.dump({"token_id": token_id, "priors": priors}, f, ensure_ascii=False)
        f.write("\n")


def get_story_words(story_df: pd.DataFrame) -> List[Dict]:
    """
    Convert phoneme-level story rows into one word-level record per token.

    Args:
        story_df: Story DataFrame containing TOKEN and ORT columns.

    Returns:
        A list of dictionaries with token_id and normalized word keys.
    """
    # Collapse phoneme rows into one word entry per token.
    grouped_story = (
        story_df.groupby("TOKEN")
        .agg({"ORT": "first"})
        .reset_index()
        .sort_values("TOKEN")
    )

    return [
        {"token_id": int(row["TOKEN"]), "word": str(row["ORT"]).lower().strip()}
        for _, row in grouped_story.iterrows()
    ]


def iter_batches(items: List[str], batch_size: int) -> Iterable[List[str]]:
    """
    Split a list into consecutive batches.

    Args:
        items: Items to split.
        batch_size: Maximum number of items in each batch.

    Yields:
        Consecutive list slices with up to batch_size items.
    """
    # Yield consecutive list chunks.
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def configure_torch_threads(cpu_threads: Optional[int] = CPU_THREADS) -> None:
    """
    Configure PyTorch CPU thread usage.

    Args:
        cpu_threads: Number of intra-op CPU threads to use, or None to leave defaults.

    Returns:
        None. The function updates PyTorch runtime thread settings when possible.
    """
    # Limit CPU thread usage during inference.
    if cpu_threads is None:
        return

    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
    except RuntimeError:
        pass


ENCODE_CACHE = {}

def encode_next_word(tokenizer, word: str, has_context: bool) -> List[int]:
    """
    Tokenize a candidate next word with GPT-2 spacing rules.

    Args:
        tokenizer: Hugging Face tokenizer used by the GPT-2 model.
        word: Candidate word to encode.
        has_context: Whether the word follows previous textual context.

    Returns:
        A list of token IDs representing the candidate word.
    """
    # Cache tokenized candidates because they are reused often.
    key = (word, has_context)

    if key in ENCODE_CACHE:
        return ENCODE_CACHE[key]

    # GPT-2 expects a leading space when a word follows context.
    text = f" {word}" if has_context else word
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Fall back to the bare word if the spaced form fails.
    if not token_ids and has_context:
        token_ids = tokenizer.encode(word, add_special_tokens=False)

    ENCODE_CACHE[key] = token_ids
    return token_ids



def expand_past_key_values(past_key_values, batch_size: int):
    """
    Expand cached prefix states to match a candidate batch size.

    Args:
        past_key_values: Cached transformer key/value states returned by the model.
        batch_size: Number of candidate continuations that will reuse the cache.

    Returns:
        Expanded cache in the format expected by the installed transformers version.
    """
    # Convert the cache to legacy format when required.
    legacy_cache = (
        past_key_values.to_legacy_cache()
        if hasattr(past_key_values, "to_legacy_cache")
        else past_key_values
    )
    # Repeat cached prefix states for every item in the batch.
    expanded_cache = tuple(
        tuple(
            None if past_state is None
            else past_state.expand(batch_size, *past_state.shape[1:]).contiguous()
            for past_state in layer
        )
        for layer in legacy_cache
    )
    # Return DynamicCache when supported by this transformers version.
    if DynamicCache is not None:
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(expanded_cache)

        cache = DynamicCache()
        for layer_idx, layer in enumerate(expanded_cache):
            key_states, value_states = layer[:2]
            if key_states is not None and value_states is not None:
                cache.update(key_states, value_states, layer_idx)
        return cache
    return expanded_cache


def score_next_words_with_gpt2(
    context_words: List[str],
    candidate_words: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = MODEL_BATCH_SIZE,
    min_probability: float = PROBABILITY_THRESHOLD,
    max_context_tokens: Optional[int] = MAX_CONTEXT_TOKENS,
    always_keep_words: Optional[set] = None,
) -> Dict[str, float]:
    """
    Score each candidate as the next word after context_words.

    Multi-subtoken words are scored as the product of their conditional subtoken
    probabilities, i.e. P(w | context) = prod_i P(subtoken_i | context, subtokens_<i).

    Args:
        context_words: Words that precede the candidate word in the story.
        candidate_words: Candidate cohort words to score as possible next words.
        tokenizer: Hugging Face tokenizer paired with the causal language model.
        model: Hugging Face causal language model used to compute probabilities.
        device: Torch device where tensors and model are placed.
        batch_size: Maximum number of candidate continuations scored together.
        min_probability: First-token probability threshold used to skip very unlikely words.
        max_context_tokens: Maximum number of GPT-2 context tokens to keep, or None for full context.
        always_keep_words: Words that should bypass the first-token probability filter.

    Returns:
        A dictionary mapping each retained candidate word to its GPT-2 next-word probability.
    """
    # Stop early when there are no candidates to score.
    if not candidate_words:
        return {}

    # Keep context tokenization bounded for long stories.
    if max_context_tokens is not None and len(context_words) > max_context_tokens:
        context_words = context_words[-max_context_tokens:]

    # Encode the left context used to predict the next word.
    context_text = " ".join(w for w in context_words if w)
    context_ids = tokenizer.encode(context_text, add_special_tokens=False) if context_text else []
    has_context = bool(context_ids)
    prefix_ids = context_ids[:]

    # Use BOS/EOS as a minimal prefix for the first word.
    if not prefix_ids:
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        if bos_id is not None:
            prefix_ids = [bos_id]

    if not prefix_ids:
        raise ValueError("Il tokenizer non ha un token BOS/EOS utilizzabile per la prima parola.")

    # Trim long contexts so prefix plus candidate fits the model window.
    max_positions = getattr(model.config, "n_positions", tokenizer.model_max_length)
    max_candidate_len = max(
        len(encode_next_word(tokenizer, word, has_context=has_context)) for word in candidate_words
    )
    max_prefix_len = max(1, max_positions - max_candidate_len - 1)
    if max_context_tokens is not None:
        max_prefix_len = min(max_prefix_len, max_context_tokens)
    if len(prefix_ids) > max_prefix_len:
        prefix_ids = prefix_ids[-max_prefix_len:]

    unique_words = sorted({str(w).lower().strip() for w in candidate_words if str(w).strip()})
    always_keep_words = always_keep_words or set()
    # Pre-tokenize all unique candidates once.
    encoded_by_word = {
        word: encode_next_word(tokenizer, word, has_context=has_context)
        for word in unique_words
    }
    scores: Dict[str, float] = {}

    # Run the prefix once and reuse its cached hidden states.
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    prefix_attention = torch.ones_like(prefix_tensor, device=device)

    model.eval()
    with torch.inference_mode():
        prefix_outputs = model(
            input_ids=prefix_tensor,
            attention_mask=prefix_attention,
            use_cache=True,
        )
        first_token_log_probs = torch.log_softmax(prefix_outputs.logits[0, len(prefix_ids) - 1], dim=-1)
        first_token_probs = first_token_log_probs.exp()

        # Filter unlikely words using their first token probability.
        words_to_score = []
        for word in unique_words:
            cand_ids = encoded_by_word[word]
            if not cand_ids:
                scores[word] = 0.0
                continue

            first_log_prob = float(first_token_log_probs[cand_ids[0]].item())
            first_prob = float(first_token_probs[cand_ids[0]].item())
            if first_prob < min_probability and word not in always_keep_words:
                continue

            if len(cand_ids) == 1:
                scores[word] = math.exp(first_log_prob)
            else:
                words_to_score.append(word)

        print(f"  kept {len(words_to_score) + len(scores)} / {len(unique_words)} candidates after first-token filter")

        # Group multi-token candidates by continuation length for batching.
        words_by_continuation_len: Dict[int, List[str]] = {}
        for word in words_to_score:
            continuation_len = len(encoded_by_word[word]) - 1
            words_by_continuation_len.setdefault(continuation_len, []).append(word)

        # Keep GPU batches smaller to reduce memory pressure.
        effective_batch_size = min(batch_size, GPU_BATCH_SIZE) if device.type == "cuda" else batch_size
        for continuation_len, words in words_by_continuation_len.items():
            for batch_words in iter_batches(words, effective_batch_size):
                batch_size_actual = len(batch_words)
                continuation_ids = [
                    encoded_by_word[word][:-1]
                    for word in batch_words
                ]
                # Reuse the prefix cache and score only candidate continuations.
                input_ids = torch.tensor(continuation_ids, dtype=torch.long, device=device)
                attention_mask = torch.ones(
                    (batch_size_actual, len(prefix_ids) + continuation_len),
                    dtype=torch.long,
                    device=device,
                )
                past_key_values = expand_past_key_values(prefix_outputs.past_key_values, batch_size_actual)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                ).logits
                log_probs = torch.log_softmax(logits, dim=-1)

                for row_idx, word in enumerate(batch_words):
                    cand_ids = encoded_by_word[word]
                    token_log_prob = float(first_token_log_probs[cand_ids[0]].item())
                    for cand_pos in range(1, len(cand_ids)):
                        token_log_prob += float(log_probs[row_idx, cand_pos - 1, cand_ids[cand_pos]].item())
                    scores[word] = math.exp(token_log_prob)

    return scores


def compute_gpt2_priors_for_cohorts(
    story_df: pd.DataFrame,
    records: List[Dict],
    model_name: str = MODEL_NAME,
    batch_size: int = MODEL_BATCH_SIZE,
    threshold: float = PROBABILITY_THRESHOLD,
    prior_cache_path: Optional[Path] = None,
    resume_priors: bool = RESUME_PRIORS,
    max_context_tokens: Optional[int] = MAX_CONTEXT_TOKENS,
) -> Dict[int, Dict[str, float]]:
    """
    Compute or resume GPT-2 next-word priors for all token-level cohort candidates.

    Args:
        story_df: Story DataFrame with one row per phoneme and TOKEN/ORT columns.
        records: Incremental phonemic cohort records loaded from JSONL.
        model_name: Hugging Face model name or local model path.
        batch_size: Maximum number of candidate continuations scored together.
        threshold: First-token probability threshold passed to the scorer.
        prior_cache_path: Optional JSONL cache path for saving and resuming token priors.
        resume_priors: Whether to reuse cached priors already present on disk.
        max_context_tokens: Maximum number of GPT-2 context tokens to keep, or None for full context.

    Returns:
        A dictionary mapping each token ID to candidate word probabilities.
    """
    # Build context words for each token in the story.
    story_words = get_story_words(story_df)
    words_by_token = {item["token_id"]: item["word"] for item in story_words}
    context_by_token: Dict[int, List[str]] = {}
    previous_words: List[str] = []

    for item in story_words:
        token_id = item["token_id"]
        context_by_token[token_id] = previous_words[:]
        previous_words.append(item["word"])

    # Collect all cohort candidates and always include the target word.
    candidates_by_token: Dict[int, set] = {}
    for rec in records:
        token_id = int(rec["token_id"])
        candidates_by_token.setdefault(token_id, set()).update(
            str(w).lower().strip() for w in rec.get("cohort_words", []) if str(w).strip()
        )
        target = words_by_token.get(token_id, str(rec.get("target_word", "")).lower().strip())
        if target:
            candidates_by_token[token_id].add(target)

    # Reuse already completed token priors before loading the model.
    priors: Dict[int, Dict[str, float]] = {}
    if prior_cache_path is not None and resume_priors:
        priors.update(load_prior_cache(prior_cache_path))
        if priors:
            print(f"Loaded cached GPT-2 priors for {len(priors)} tokens from {prior_cache_path}")

    token_ids_to_score = [
        token_id for token_id in sorted(candidates_by_token)
        if token_id not in priors
    ]
    if not token_ids_to_score:
        print("All GPT-2 priors were already cached; skipping model scoring.")
        return {token_id: priors[token_id] for token_id in sorted(candidates_by_token)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    # Load the tokenizer and language model only when new tokens need scoring.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device.type == "cpu":
        configure_torch_threads()
        print(f"Using CPU with {torch.get_num_threads()} PyTorch threads")
    else:
        print(f"Using {device}")
    model.to(device)
        
    # Score candidate priors token by token.
    total = len(token_ids_to_score)
    for idx, token_id in enumerate(token_ids_to_score, start=1):
        candidates = sorted(candidates_by_token[token_id])
        print(
            f"Scoring GPT-2 next-word probabilities for token {token_id} "
            f"({idx}/{total}, {len(candidates)} candidates)"
        )
        priors[token_id] = score_next_words_with_gpt2(
            context_words=context_by_token.get(token_id, []),
            candidate_words=candidates,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            min_probability=threshold,
            max_context_tokens=max_context_tokens,
            always_keep_words={words_by_token.get(token_id, "")},
        )
        if prior_cache_path is not None:
            append_prior_cache(prior_cache_path, token_id, priors[token_id])
    return priors


def compute_incremental_prefix_probs(
    records: List[Dict],
    priors: Dict[int, Dict[str, float]],
    threshold: float = PROBABILITY_THRESHOLD,
    include_target: bool = True,
    out_path: Path = Path("cohort_incremental_probs.jsonl"),
) -> None:
    """
    Convert token-level word priors into incremental phoneme-prefix probabilities.

    Args:
        records: Incremental phonemic cohort records loaded from JSONL.
        priors: GPT-2 word probabilities indexed by token ID.
        threshold: Minimum probability required for a candidate to remain in the filtered cohort.
        include_target: Whether to force the target word into the filtered cohort when valid.
        out_path: Destination JSONL path for the incremental probability records.

    Returns:
        None. The function writes one JSONL record per token-prefix row to out_path.
    """
    # Create the output folder before writing the JSONL file.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    previous_mass_by_token: Dict[int, float] = {}
    # Process records in story order and increasing prefix length.
    sorted_records = sorted(
        records,
        key=lambda rec: (int(rec.get("token_id")), int(rec.get("prefix_length", 0))),
    )
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in sorted_records:
            token_id = rec.get("token_id")
            token_key = int(token_id)
            target = str(rec.get("target_word", "")).lower().strip()
            cohort = [w.lower().strip() for w in rec.get("cohort_words", [])]
            prefix = rec.get("prefix_phonemes")
            prefix_length = rec.get("prefix_length")

            cand_dict = priors.get(token_key, {})
            cohort_set = set(cohort)
            # Keep only cohort words with GPT-2 probability above threshold.
            filtered = [(w, p) for w, p in cand_dict.items() if w in cohort_set and p >= threshold]
            target_prob = cand_dict.get(target, 0.0)
            # Reinsert the target word if requested, regardless of probability threshold.
            if (
                include_target
                and target in cohort_set
                and not any(w == target for w, _ in filtered)
            ):
                filtered.append((target, target_prob))
            filtered.sort(key=lambda x: x[1], reverse=True)

            # Cohort mass is the summed probability of surviving candidates.
            cohort_mass = sum(p for _, p in filtered)
            previous_cohort_mass = previous_mass_by_token.get(token_key)
            # The first phoneme has no prefix-0 cohort in the input file.
            phoneme_prob = (
                cohort_mass / previous_cohort_mass
                if previous_cohort_mass and previous_cohort_mass > 0.0
                else None
            )
            previous_mass_by_token[token_key] = cohort_mass
            # Store aggregate measures and the filtered cohort itself.
            out_record = {
                "token_id": token_id,
                "target_word": target,
                "prefix_phonemes": prefix,
                "prefix_length": prefix_length,
                "cohort_size": len(cohort),
                "filtered_cohort_size": len(filtered),
                "cohort_mass": cohort_mass,
                "previous_cohort_mass": previous_cohort_mass,
                "phoneme_prob": phoneme_prob,
                "target_prob": target_prob,
                "filtered_cohort": filtered,
            }
            json.dump(out_record, f, ensure_ascii=False)
            f.write("\n")


def compute_all(
    story_path: Path = STORY_PATH,
    priors_path: Optional[Path] = PRIORS_PATH,
    model_name: str = MODEL_NAME,
    story_id: str = STORY_ID,
    output_root: Path = OUTPUT_ROOT,
    output_group: str = OUTPUT_GROUP,
    threshold: float = PROBABILITY_THRESHOLD,
    include_target: bool = True,
    max_tokens_for_test: Optional[int] = MAX_TOKENS_FOR_TEST,
    max_context_tokens: Optional[int] = MAX_CONTEXT_TOKENS,
    resume_priors: bool = RESUME_PRIORS,
) -> None:
    """
    Run the full GPT-2 cohort probability pipeline for one story.

    Args:
        story_path: Path to the story Excel file.
        priors_path: Optional path to external precomputed priors.
        model_name: Hugging Face model name or local model path.
        story_id: Story identifier used to resolve input and output file names.
        output_root: Base output folder.
        output_group: Output group or condition folder.
        threshold: Probability threshold used for scoring and filtering candidates.
        include_target: Whether to keep the target word when it passes filtering rules.
        max_tokens_for_test: Optional number of initial story tokens to process for tests.
        max_context_tokens: Maximum number of GPT-2 context tokens to keep, or None for full context.
        resume_priors: Whether to resume GPT-2 scoring from the token-level prior cache.

    Returns:
        None. The function writes the GPT-2 cohort probability JSONL file to disk.
    """
    # Load and optionally shorten the story.
    story_df = load_story(story_path)
    story_df = keep_first_tokens(story_df, max_tokens_for_test)

    # Resolve input cohort path and output probability path.
    output_base = get_output_base(output_root, output_group, story_id)
    incremental_cohorts_path = output_base / COHORTS_FILENAME_TEMPLATE.format(story_id=story_id)
    cohort_probs_path = output_base / GPT_COHORTS_FILENAME_TEMPLATE.format(story_id=story_id)
    prior_cache_path = output_base / GPT_PRIORS_FILENAME_TEMPLATE.format(story_id=story_id)

    cohort_records = load_jsonl(incremental_cohorts_path)
    print(f"Loaded incremental cohorts from {incremental_cohorts_path}")

    # Either reuse external priors or compute them with GPT-2.
    if priors_path is not None:
        priors = load_priors(priors_path)
    else:
        priors = compute_gpt2_priors_for_cohorts(
            story_df=story_df,
            records=cohort_records,
            model_name=model_name,
            batch_size=MODEL_BATCH_SIZE,
            threshold=threshold,
            prior_cache_path=prior_cache_path,
            resume_priors=resume_priors,
            max_context_tokens=max_context_tokens,
        )

    # Convert token priors into incremental phoneme-prefix probabilities.
    compute_incremental_prefix_probs(
        cohort_records,
        priors,
        threshold=threshold,
        include_target=include_target,
        out_path=cohort_probs_path,
    )
    print(f"Saved incremental prefix probability records to {cohort_probs_path}")


if __name__ == "__main__":
    compute_all()
