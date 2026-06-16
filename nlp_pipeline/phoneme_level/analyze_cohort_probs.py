from pathlib import Path
import statistics as stats
import csv
import matplotlib.pyplot as plt

INPUT_PATH = Path("output_nlp/output_D/St01_D/phoneme_level/phoneme_surprisal_St01_D.csv")
OUTPUT_DIR = Path("output_nlp/output_D/St01_D/phoneme_level/analysis")


def load_results(path: Path):
    records = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                surprisal = float(row.get("phoneme_surprisal", "0"))
            except ValueError:
                surprisal = 0.0

            try:
                token_id = int(row.get("token_id", 0))
            except ValueError:
                token_id = 0

            records.append({
                "token_id": token_id,
                "word": row.get("word", ""),
                "phoneme": row.get("phoneme", ""),
                "phoneme_surprisal": surprisal,
            })
    return records


def summarize(results):
    surprisal_values = [r["phoneme_surprisal"] for r in results]
    unique_words = {r["word"] for r in results if r["word"]}

    stats_summary = {
        "n_phonemes": len(results),
        "n_words": len(unique_words),
        "surprisal_mean": stats.mean(surprisal_values) if surprisal_values else 0,
        "surprisal_median": stats.median(surprisal_values) if surprisal_values else 0,
        "surprisal_std": stats.pstdev(surprisal_values) if len(surprisal_values) > 1 else 0,
        "surprisal_min": min(surprisal_values) if surprisal_values else 0,
        "surprisal_max": max(surprisal_values) if surprisal_values else 0,
    }

    return results, stats_summary


def save_summary_csv(summary, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def make_plots(tokens, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    surprisal_values = [t["phoneme_surprisal"] for t in tokens]
    words = [t["word"] for t in tokens]

    plt.figure(figsize=(8, 4))
    plt.hist(surprisal_values, bins=40, color="#4c78a8", edgecolor="black")
    plt.xlabel("Phoneme surprisal")
    plt.ylabel("Count")
    plt.title("Distribution of phoneme surprisal")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_phoneme_surprisal.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.boxplot(surprisal_values, vert=False)
    plt.xlabel("Phoneme surprisal")
    plt.title("Boxplot of phoneme surprisal")
    plt.tight_layout()
    plt.savefig(out_dir / "box_phoneme_surprisal.png")
    plt.close()

    avg_surprisal_by_word = {}
    count_by_word = {}
    for t in tokens:
        word = t["word"] or "<unknown>"
        avg_surprisal_by_word.setdefault(word, 0.0)
        count_by_word.setdefault(word, 0)
        avg_surprisal_by_word[word] += t["phoneme_surprisal"]
        count_by_word[word] += 1

    avg_surprisal_by_word = {
        word: avg_surprisal_by_word[word] / count_by_word[word]
        for word in avg_surprisal_by_word
        if count_by_word[word] > 0
    }

    top_words = sorted(avg_surprisal_by_word.items(), key=lambda x: x[1], reverse=True)[:20]
    if top_words:
        words, avg_values = zip(*top_words)
        plt.figure(figsize=(10, 5))
        plt.bar(words, avg_values, color="#ff7f0e")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Average phoneme surprisal")
        plt.title("Top 20 words by average phoneme surprisal")
        plt.tight_layout()
        plt.savefig(out_dir / "top20_avg_surprisal_by_word.png")
        plt.close()


def main():
    path = INPUT_PATH
    out_dir = OUTPUT_DIR
    summary_csv = out_dir / "summary.csv"

    results = load_results(path)
    tokens, summary = summarize(results)

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    save_summary_csv(summary, summary_csv)
    make_plots(tokens, out_dir)
    print(f"Saved summary to {summary_csv} and plots to {out_dir}")


if __name__ == "__main__":
    main()
