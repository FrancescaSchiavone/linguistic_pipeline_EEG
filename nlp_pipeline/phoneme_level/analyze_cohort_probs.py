import argparse
import json
from pathlib import Path
import statistics as stats
import csv
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze cohort probability output and generate CSV/plots.")
    parser.add_argument("--input", type=str, default="cohort_firstletter_probs.json", help="Input JSON file to analyze")
    parser.add_argument("--output-dir", type=str, default="analysis", help="Directory to save CSV and plots")
    return parser.parse_args()


def load_results(path: Path):
    if path.suffix.lower() == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                records.append(json.loads(text))
        return records

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(results):
    tokens = []
    cohort_sizes = []
    sum_masses = []
    max_probs = []
    mean_probs = []
    target_probs = []

    for i, v in enumerate(results):
        token_id = int(v.get("token_id", i))
        cohort = v.get("filtered_cohort", [])
        probs = [p for (_, p) in cohort]
        cohort_sizes.append(len(cohort))
        sum_masses.append(sum(probs))
        max_probs.append(max(probs) if probs else 0.0)
        mean_probs.append(stats.mean(probs) if probs else 0.0)
        # find target prob
        targ = next((p for w, p in cohort if w == v.get("target_word")), 0.0)
        target_probs.append(targ)
        tokens.append({
            "token_id": token_id,
            "target_word": v.get("target_word"),
            "cohort_size": len(cohort),
            "sum_mass": sum(probs),
            "max_prob": (max(probs) if probs else 0.0),
            "mean_prob": (stats.mean(probs) if probs else 0.0),
            "target_prob": targ,
        })

    stats_summary = {
        "n_tokens": len(tokens),
        "cohort_size_mean": stats.mean(cohort_sizes) if cohort_sizes else 0,
        "cohort_size_median": stats.median(cohort_sizes) if cohort_sizes else 0,
        "cohort_size_std": stats.pstdev(cohort_sizes) if cohort_sizes else 0,
        "sum_mass_mean": stats.mean(sum_masses) if sum_masses else 0,
        "sum_mass_median": stats.median(sum_masses) if sum_masses else 0,
        "max_prob_mean": stats.mean(max_probs) if max_probs else 0,
        "target_prob_mean": stats.mean(target_probs) if target_probs else 0,
    }

    return tokens, stats_summary


def save_csv(tokens, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["token_id", "target_word", "cohort_size", "sum_mass", "max_prob", "mean_prob", "target_prob"])
        writer.writeheader()
        for t in sorted(tokens, key=lambda x: x["token_id"]):
            writer.writerow(t)


def make_plots(tokens, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort_sizes = [t["cohort_size"] for t in tokens]
    sum_masses = [t["sum_mass"] for t in tokens]
    max_probs = [t["max_prob"] for t in tokens]
    target_probs = [t["target_prob"] for t in tokens]

    plt.figure(figsize=(8,4))
    plt.hist(cohort_sizes, bins=30)
    plt.xlabel("Cohort size (filtered)")
    plt.ylabel("Count")
    plt.title("Distribution of filtered cohort sizes")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_cohort_sizes.png")
    plt.close()

    plt.figure(figsize=(8,4))
    plt.hist(sum_masses, bins=30)
    plt.xlabel("Cohort probability mass")
    plt.ylabel("Count")
    plt.title("Distribution of cohort probability mass")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_sum_mass.png")
    plt.close()

    plt.figure(figsize=(8,4))
    plt.hist(max_probs, bins=30)
    plt.xlabel("Max probability in cohort")
    plt.ylabel("Count")
    plt.title("Distribution of max candidate probability")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_max_prob.png")
    plt.close()

    plt.figure(figsize=(8,4))
    plt.boxplot([max_probs, target_probs], labels=["max_prob", "target_prob"])
    plt.title("Boxplot: max vs target probabilities")
    plt.tight_layout()
    plt.savefig(out_dir / "box_max_vs_target.png")
    plt.close()

    # scatter cohort_size vs sum_mass
    plt.figure(figsize=(6,6))
    plt.scatter(cohort_sizes, sum_masses, alpha=0.7)
    plt.xlabel("Cohort size")
    plt.ylabel("Sum mass")
    plt.title("Cohort size vs probability mass")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_size_mass.png")
    plt.close()

    # top tokens by sum_mass
    top = sorted(tokens, key=lambda x: x["sum_mass"], reverse=True)[:20]
    words = [t["target_word"] for t in top]
    masses = [t["sum_mass"] for t in top]
    plt.figure(figsize=(10,4))
    plt.bar(words, masses)
    plt.xticks(rotation=45, ha='right')
    plt.title("Top 20 tokens by cohort mass")
    plt.tight_layout()
    plt.savefig(out_dir / "top20_sum_mass.png")
    plt.close()


def main():
    args = parse_args()
    path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_csv = out_dir / "cohort_stats.csv"
    out_plots = out_dir / "plots"

    results = load_results(path)
    tokens, summary = summarize(results)

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    save_csv(tokens, out_csv)
    make_plots(tokens, out_plots)
    print(f"Saved CSV to {out_csv} and plots to {out_plots}")


if __name__ == "__main__":
    main()
