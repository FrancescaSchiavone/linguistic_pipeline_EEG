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
# Configurable paths: change these only at the top of the file.
# CHANGE HERE (#🦑)
#   - STORY_PATH: percorso del file Excel della storia
#   - STORY_ID: nome della storia, es. St01_D
#   - OUTPUT_ROOT / OUTPUT_GROUP: cartella base output
#   - PRIORS_PATH: percorso al file dei prior esterno (questo NON e il file delle coorti)
# ---------------------------------------------------------------------------
STORY_PATH = Path(r"data\phoneme_onset_D\St01_D.xlsx") #🦑
STORY_ID = "St01_D" #🦑
OUTPUT_ROOT = Path("output_nlp")
OUTPUT_GROUP = "output_D" #🦑
MODEL_NAME = "GroNLP/gpt2-small-italian"
PRIORS_PATH = None  # es. Path("provettatopk1000/candidate_words.json") per riusare prior gia pronti
COHORTS_FILENAME_TEMPLATE = "incremental_phonemic_cohorts_{story_id}.jsonl"
GPT_COHORTS_FILENAME_TEMPLATE = "incremental_phonemic_cohorts_gpt_{story_id}.jsonl"
MODEL_BATCH_SIZE = 128
GPU_BATCH_SIZE = 16
MAX_TOKENS_FOR_TEST = None # usa 50 per un test veloce; None processa tutta la storia #🦑
PROBABILITY_THRESHOLD = 1e-8
CPU_THREADS = 8  # metti un numero piu basso se il PC diventa poco responsivo


def load_story(story_path: Path) -> pd.DataFrame:
    story_df = pd.read_excel(story_path)
    story_df = story_df[story_df["MAU"].notna()]
    story_df["ORT"] = story_df["ORT"].astype(str).str.strip().str.lower()
    story_df["MAU"] = story_df["MAU"].astype(str).str.strip()

    return story_df


def keep_first_tokens(story_df: pd.DataFrame, max_tokens: Optional[int]) -> pd.DataFrame:
    if max_tokens is None:
        return story_df

    token_ids = sorted(story_df["TOKEN"].dropna().unique())[:max_tokens]
    return story_df[story_df["TOKEN"].isin(token_ids)].copy()


def get_output_base(output_root: Path, output_group: str, story_id: str) -> Path:
    return output_root / output_group / story_id / "phoneme_level"


def load_jsonl(input_path: Path) -> List[Dict]:
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_priors(priors_path: Path) -> Dict[int, Dict[str, float]]:
    with open(priors_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    priors: Dict[int, Dict[str, float]] = {}
    for item in data:
        pos = item.get("word_position")
        cand = {c["word"].lower().strip(): c.get("prob", 0.0) for c in item.get("candidates", [])}
        priors[int(pos)] = cand

    return priors


def get_story_words(story_df: pd.DataFrame) -> List[Dict]:
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
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def configure_torch_threads(cpu_threads: Optional[int] = CPU_THREADS) -> None:
    if cpu_threads is None:
        return

    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
    except RuntimeError:
        pass


def encode_next_word(tokenizer, word: str, has_context: bool) -> List[int]:
    text = f" {word}" if has_context else word
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if not token_ids and has_context:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
    return token_ids


def expand_past_key_values(past_key_values, batch_size: int):
    legacy_cache = (
        past_key_values.to_legacy_cache()
        if hasattr(past_key_values, "to_legacy_cache")
        else past_key_values
    )
    expanded_cache = tuple(
        tuple(
            None if past_state is None
            else past_state.expand(batch_size, *past_state.shape[1:]).contiguous()
            for past_state in layer
        )
        for layer in legacy_cache
    )
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
    always_keep_words: Optional[set] = None,
) -> Dict[str, float]:
    """
    Score each candidate as the next word after context_words.

    Multi-subtoken words are scored as the product of their conditional subtoken
    probabilities, i.e. P(w | context) = prod_i P(subtoken_i | context, subtokens_<i).
    """
    if not candidate_words:
        return {}

    context_text = " ".join(w for w in context_words if w)
    context_ids = tokenizer.encode(context_text, add_special_tokens=False) if context_text else []
    has_context = bool(context_ids)
    prefix_ids = context_ids[:]

    if not prefix_ids:
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        if bos_id is not None:
            prefix_ids = [bos_id]

    if not prefix_ids:
        raise ValueError("Il tokenizer non ha un token BOS/EOS utilizzabile per la prima parola.")

    max_positions = getattr(model.config, "n_positions", tokenizer.model_max_length)
    max_candidate_len = max(
        len(encode_next_word(tokenizer, word, has_context=has_context)) for word in candidate_words
    )
    max_prefix_len = max(1, max_positions - max_candidate_len - 1)
    if len(prefix_ids) > max_prefix_len:
        prefix_ids = prefix_ids[-max_prefix_len:]

    unique_words = sorted({str(w).lower().strip() for w in candidate_words if str(w).strip()})
    always_keep_words = always_keep_words or set()
    encoded_by_word = {
        word: encode_next_word(tokenizer, word, has_context=has_context)
        for word in unique_words
    }
    scores: Dict[str, float] = {}

    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    prefix_attention = torch.ones_like(prefix_tensor, device=device)

    model.eval()
    with torch.no_grad():
        prefix_outputs = model(
            input_ids=prefix_tensor,
            attention_mask=prefix_attention,
            use_cache=True,
        )
        first_token_log_probs = torch.log_softmax(prefix_outputs.logits[0, len(prefix_ids) - 1], dim=-1)
        first_token_probs = first_token_log_probs.exp()

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

        words_by_continuation_len: Dict[int, List[str]] = {}
        for word in words_to_score:
            continuation_len = len(encoded_by_word[word]) - 1
            words_by_continuation_len.setdefault(continuation_len, []).append(word)

        effective_batch_size = min(batch_size, GPU_BATCH_SIZE) if device.type == "cuda" else batch_size
        for continuation_len, words in words_by_continuation_len.items():
            for batch_words in iter_batches(words, effective_batch_size):
                batch_size_actual = len(batch_words)
                continuation_ids = [
                    encoded_by_word[word][:-1]
                    for word in batch_words
                ]
                if device.type == "cuda":
                    sequences = [prefix_ids + encoded_by_word[word] for word in batch_words]
                    input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
                    attention_mask = torch.ones_like(input_ids, device=device)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    log_probs = torch.log_softmax(logits, dim=-1)

                    prefix_len = len(prefix_ids)
                    for row_idx, word in enumerate(batch_words):
                        cand_ids = encoded_by_word[word]
                        token_log_prob = 0.0
                        for cand_pos, token_id in enumerate(cand_ids):
                            pred_pos = prefix_len + cand_pos - 1
                            token_log_prob += float(log_probs[row_idx, pred_pos, token_id].item())
                        scores[word] = math.exp(token_log_prob)
                    continue

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
) -> Dict[int, Dict[str, float]]:
    story_words = get_story_words(story_df)
    words_by_token = {item["token_id"]: item["word"] for item in story_words}
    context_by_token: Dict[int, List[str]] = {}
    previous_words: List[str] = []

    for item in story_words:
        token_id = item["token_id"]
        context_by_token[token_id] = previous_words[:]
        previous_words.append(item["word"])

    candidates_by_token: Dict[int, set] = {}
    for rec in records:
        token_id = int(rec["token_id"])
        candidates_by_token.setdefault(token_id, set()).update(
            str(w).lower().strip() for w in rec.get("cohort_words", []) if str(w).strip()
        )
        target = words_by_token.get(token_id, str(rec.get("target_word", "")).lower().strip())
        if target:
            candidates_by_token[token_id].add(target)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        configure_torch_threads()
        print(f"Using CPU with {torch.get_num_threads()} PyTorch threads")
    else:
        print(f"Using {device}")
    model.to(device)

    priors: Dict[int, Dict[str, float]] = {}
    total = len(candidates_by_token)
    for idx, token_id in enumerate(sorted(candidates_by_token), start=1):
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
            always_keep_words={words_by_token.get(token_id, "")},
        )
    return priors


def compute_incremental_prefix_probs(
    records: List[Dict],
    priors: Dict[int, Dict[str, float]],
    threshold: float = PROBABILITY_THRESHOLD,
    include_target: bool = True,
    out_path: Path = Path("cohort_incremental_probs.jsonl"),
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    previous_mass_by_token: Dict[int, float] = {}
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
            filtered = [(w, p) for w, p in cand_dict.items() if w in cohort_set and p >= threshold]
            target_prob = cand_dict.get(target, 0.0)
            if (
                include_target
                and target in cohort_set
                and target_prob >= threshold
                and not any(w == target for w, _ in filtered)
            ):
                filtered.append((target, target_prob))
            filtered.sort(key=lambda x: x[1], reverse=True)

            cohort_mass = sum(p for _, p in filtered)
            previous_cohort_mass = previous_mass_by_token.get(token_key)
            # The first phoneme has no prefix-0 cohort in the input file.
            phoneme_prob = (
                cohort_mass / previous_cohort_mass
                if previous_cohort_mass and previous_cohort_mass > 0.0
                else None
            )
            previous_mass_by_token[token_key] = cohort_mass
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
) -> None:
    story_df = load_story(story_path)
    story_df = keep_first_tokens(story_df, max_tokens_for_test)

    output_base = get_output_base(output_root, output_group, story_id)
    incremental_cohorts_path = output_base / COHORTS_FILENAME_TEMPLATE.format(story_id=story_id)
    cohort_probs_path = output_base / GPT_COHORTS_FILENAME_TEMPLATE.format(story_id=story_id)

    cohort_records = load_jsonl(incremental_cohorts_path)
    print(f"Loaded incremental cohorts from {incremental_cohorts_path}")

    if priors_path is not None:
        priors = load_priors(priors_path)
    else:
        priors = compute_gpt2_priors_for_cohorts(
            story_df=story_df,
            records=cohort_records,
            model_name=model_name,
            batch_size=MODEL_BATCH_SIZE,
            threshold=threshold,
        )

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
