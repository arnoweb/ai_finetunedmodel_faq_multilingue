import json
import sys
from collections import defaultdict
from typing import Dict, List, Sequence

from sentence_transformers import SentenceTransformer, util

# Models to compare
MODEL_PATHS = {
    "Base multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Fine-tuned (AutoTrain)": "arnoweb/model-faq-sentence-autotrain",
}

# FAQ sources per language, mirroring the Streamlit app
FAQ_DATA_PATHS = {
    "en": "data/faq_source_en.jsonl",
    "fr": "data/faq_source_fr.jsonl",
}

# Evaluation data (must contain fields: query, answer, lang)
EVAL_DATA_PATH = "data/faq_evaluation.jsonl"

# Report recall up to these cutoffs
TOP_KS: Sequence[int] = (1, 3, 5)


def load_faq(path: str):
    questions, answers = [], []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                questions.append(item["query"])
                answers.append(item["answer"])
    return questions, answers


def load_eval(path: str):
    items = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def evaluate_model(
    model_name: str,
    model_path: str,
    eval_items: List[Dict],
    faq_by_lang: Dict[str, Dict[str, List[str]]],
    top_ks: Sequence[int],
):
    model = SentenceTransformer(model_path)
    metrics = {
        "overall": defaultdict(float),
        "lang": defaultdict(lambda: defaultdict(float)),
        "missing_answers": defaultdict(int),
    }

    counts = defaultdict(int)
    lang_counts = defaultdict(int)

    for lang, faq_data in faq_by_lang.items():
        answers = faq_data["answers"]
        answer_embeddings = model.encode(answers, convert_to_tensor=True)

        for item in [e for e in eval_items if e.get("lang") == lang]:
            counts["total"] += 1
            lang_counts[lang] += 1

            query = item["query"]
            true_answer = item["answer"]

            candidate_indices = [i for i, a in enumerate(answers) if a == true_answer]
            if not candidate_indices:
                metrics["missing_answers"][lang] += 1
                continue

            query_emb = model.encode(query, convert_to_tensor=True)
            sims = util.pytorch_cos_sim(query_emb, answer_embeddings)[0]
            sorted_indices = sims.argsort(descending=True).tolist()

            best_rank = None
            best_score = None
            for rank, idx in enumerate(sorted_indices):
                if idx in candidate_indices:
                    best_rank = rank
                    best_score = sims[idx].item()
                    break

            if best_rank is None:
                metrics["missing_answers"][lang] += 1
                continue

            # Aggregations
            metrics["overall"]["mrr"] += 1 / (best_rank + 1)
            metrics["overall"]["sim_sum"] += best_score
            for k in top_ks:
                if best_rank < k:
                    metrics["overall"][f"recall@{k}"] += 1

            metrics["lang"][lang]["mrr"] += 1 / (best_rank + 1)
            metrics["lang"][lang]["sim_sum"] += best_score
            for k in top_ks:
                if best_rank < k:
                    metrics["lang"][lang][f"recall@{k}"] += 1

    # Finalize averages
    results = {"model": model_name, "overall": {}, "lang": {}, "missing": metrics["missing_answers"]}

    for key, value in metrics["overall"].items():
        if key.startswith("recall@"):
            results["overall"][key] = value / counts["total"] if counts["total"] else 0.0
        elif key == "mrr":
            results["overall"]["mrr"] = value / counts["total"] if counts["total"] else 0.0
        elif key == "sim_sum":
            results["overall"]["mean_true_sim"] = value / counts["total"] if counts["total"] else 0.0

    for lang, agg in metrics["lang"].items():
        results["lang"][lang] = {}
        for key, value in agg.items():
            denom = lang_counts[lang]
            if key.startswith("recall@"):
                results["lang"][lang][key] = value / denom if denom else 0.0
            elif key == "mrr":
                results["lang"][lang]["mrr"] = value / denom if denom else 0.0
            elif key == "sim_sum":
                results["lang"][lang]["mean_true_sim"] = value / denom if denom else 0.0

    return results


def validate_eval_answers(eval_items: List[Dict], faq_by_lang: Dict[str, Dict[str, List[str]]], max_examples: int = 5):
    """Ensure every eval answer exists in the retrieval corpus; otherwise fail fast with details."""
    missing = defaultdict(list)
    faq_answer_sets = {lang: set(data["answers"]) for lang, data in faq_by_lang.items()}
    for item in eval_items:
        lang = item.get("lang")
        answer = item.get("answer")
        if lang not in faq_answer_sets or answer not in faq_answer_sets[lang]:
            missing[lang].append(answer)

    if missing:
        print("ERROR: Some evaluation answers are not present in the FAQ candidates. Metrics would be zero.\n")
        for lang, answers in missing.items():
            print(f"Lang '{lang}': missing {len(answers)} answers.")
            for a in answers[:max_examples]:
                print(f"  - {a}")
            if len(answers) > max_examples:
                print(f"  ... and {len(answers) - max_examples} more.")
        print("\nFix by aligning eval answers with the FAQ files (data/faq_source_*.jsonl) or updating the eval set.")
        sys.exit(1)


def main():
    eval_items = load_eval(EVAL_DATA_PATH)
    faq_by_lang = {}
    for lang, path in FAQ_DATA_PATHS.items():
        questions, answers = load_faq(path)
        faq_by_lang[lang] = {"questions": questions, "answers": answers}

    validate_eval_answers(eval_items, faq_by_lang)

    print(f"Loaded {len(eval_items)} eval queries.")
    for lang, data in faq_by_lang.items():
        print(f"Lang '{lang}': {len(data['answers'])} candidate answers.")

    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n=== Evaluating: {model_name} ===")
        results = evaluate_model(model_name, model_path, eval_items, faq_by_lang, TOP_KS)
        overall = results["overall"]
        print(
            f"MRR: {overall.get('mrr', 0):.3f} | "
            + " | ".join(
                f"Recall@{k}: {overall.get(f'recall@{k}', 0):.3f}" for k in TOP_KS
            )
            + f" | Mean true sim: {overall.get('mean_true_sim', 0):.3f}"
        )
        for lang, metrics in results["lang"].items():
            print(
                f"  [{lang}] MRR: {metrics.get('mrr', 0):.3f} | "
                + " | ".join(
                    f"Recall@{k}: {metrics.get(f'recall@{k}', 0):.3f}"
                    for k in TOP_KS
                )
                + f" | Mean true sim: {metrics.get('mean_true_sim', 0):.3f}"
            )
        if results["missing"]:
            print(f"Missing answers (not found in FAQ): {dict(results['missing'])}")


if __name__ == "__main__":
    main()
