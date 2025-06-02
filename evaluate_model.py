import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# Paths (update as needed)
model_path = "model-faq-sentence-autotrain"  # Path to your model
eval_path = "models/evaluations"  # Path to your model
# Use a .jsonl file for evaluation data
# Each line should be a JSON object with 'query' and 'answer' fields
eval_data_path = "data/faq_evaluation.jsonl"

# Load model
model = SentenceTransformer(model_path)

# Load evaluation data from .jsonl
eval_data = []
with open(eval_data_path, "r") as f:
    for line in f:
        if line.strip():
            eval_data.append(json.loads(line))

# Prepare lists of queries and answers
queries = [item["query"] for item in eval_data]
answers = [item["answer"] for item in eval_data]

# Encode all answers once
answer_embeddings = model.encode(answers, convert_to_tensor=True)

# Evaluate
correct = 0
similarities = []
for i, query in enumerate(queries):
    query_embedding = model.encode(query, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(query_embedding, answer_embeddings)[0]
    similarities.append(sims[i].item())  # Similarity with correct answer
    top_idx = sims.argmax().item()
    if top_idx == i:
        correct += 1

# After the evaluation loop
low_similarity_threshold = 0.5  # You can adjust this value

print("\nQueries with low similarity to their correct answer:")
for i, sim in enumerate(similarities):
    if sim < low_similarity_threshold:
        print(f"\nQuery: {queries[i]}")
        print(f"Correct Answer: {answers[i]}")
        print(f"Similarity: {sim:.2f}")

precision_at_1 = correct / len(queries)
print(f"Precision@1: {precision_at_1:.3f}")

# Plot similarity distribution
plt.figure(figsize=(8, 5))
plt.hist(similarities, bins=20, color='skyblue')
plt.title(f"Similarity Distribution (mean={np.mean(similarities):.2f})")
plt.xlabel("Cosine Similarity (query vs. correct answer)")
plt.ylabel("Count")
plt.tight_layout()
plot_path = os.path.join(eval_path, "model-faq-sentence-autotrain_similarity_distribution.png")
plt.savefig(plot_path)
plt.close()
print(f"Similarity distribution plot saved to: {plot_path}") 