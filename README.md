[tags: python-3.11, streamlit-1.51, autotrain, huggingface-hub, local-llm]

# FAQ Retrieval Apps (Base vs Fine-tuned)

Two Streamlit apps to serve and compare an FAQ semantic search workflow:
- `faq_streamlit.py`: main RAG-style app (top-3 retrieval + optional LLM answer).
- `faq_compare_streamlit.py`: side-by-side comparison of a base multilingual encoder vs your fine-tuned AutoTrain model.

Models used
- Base: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Fine-tuned: `arnoweb/model-faq-sentence-autotrain` (Hugging Face Hub)

Data used
- Live FAQ content: `data/faq_source_en.jsonl`, `data/faq_source_fr.jsonl`
- Eval set: `data/faq_evaluation.jsonl`

## Prerequisites
- Python 3.11 (recommended)
- pip/venv
- Hugging Face Hub access if the fine-tuned repo is private (`huggingface-cli login` or `HUGGINGFACE_HUB_TOKEN`)
- Optional: a local LLM server exposing an OpenAI-compatible `/v1/chat/completions` endpoint at `http://localhost:1234` (e.g., LM Studio, llama.cpp, Ollama)

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# If needed: huggingface-cli login
```

## Run the main FAQ RAG app
```bash
source .venv/bin/activate
streamlit run faq_streamlit.py --server.port 8501 --server.address 0.0.0.0
```
- Retrieval: SentenceTransformer embeddings over the FAQ JSONL for the selected language.
- Generation (optional): calls `query_local_llm` to your local LLM endpoint.

### Local LLM connection (LM Studio example)
- In LM Studio, start a model and enable the OpenAI-compatible server (default `http://localhost:1234/v1/chat/completions`).
- The app already calls that URL. If you change the port/host, update `query_local_llm` in `faq_streamlit.py`.
- No API key is read from env by default; add headers in `query_local_llm` if your server requires one.

## Run the comparison app (base vs fine-tuned)
```bash
source .venv/bin/activate
streamlit run faq_compare_streamlit.py --server.port 8502 --server.address 0.0.0.0
```
- Shows top-K results from both models side by side on the same FAQ data.
- Adjust K and a similarity threshold to flag low-confidence matches.

## Evaluate models on the held-out set
```bash
source .venv/bin/activate
python evaluate_models_compare.py
```
- Reports MRR, Recall@1/3/5, and mean similarity overall and per language.
- Warns if any eval answers are missing from the FAQ files.

### Visualize training metrics with TensorBoard
If you have training logs in `model-faq-sentence-autotrain/runs`, you can inspect them with:
```bash
tensorboard --logdir model-faq-sentence-autotrain/runs
```
Open the local URL printed by TensorBoard to view losses/metrics over time.
- Train loss vs eval/validation loss: train loss measures fit on the training batches; eval loss measures generalization on held-out data. Watch them over epochs.
- Epochs and steps: one epoch = one full pass over the training set; steps are the mini-batch updates inside each epoch.
- Interpreting curves:
  - If both losses decrease smoothly (less noise) across steps/epochs, the model is learning useful structure.
  - If eval loss bottoms out then rises while train keeps dropping toward zero, that signals overfitting.
  - Extremely small losses on train and eval can also indicate overfitting or a trivial task; prefer a modest gap and stable eval loss.

## Evaluation Metrics

### MRR (Mean Reciprocal Rank)
Measures how highly the correct answer is ranked.  
A score closer to 1 means the model places the correct answer near the top of the results list.

### Recall@K
Indicates whether the correct answer appears within the top K returned results.  
- **Recall@1**: the correct answer is the first result  
- **Recall@3 / Recall@5**: the correct answer is within the top 3 or top 5  
Higher values mean the model retrieves relevant answers more reliably.

### Mean True Similarity
Represents the average similarity (e.g., cosine similarity) between each query and its correct answer.  
Higher scores indicate a better semantic understanding between questions and their corresponding answers.

---

## Example Results

| Model                     | MRR   | Recall@1 | Recall@3 | Recall@5 | Mean True Sim | Explanation |
|---------------------------|-------|----------|----------|----------|----------------|-------------|
| Base multilingual (global) | 0.844 | 0.750    | 0.950    | 0.950    | 0.544          | Baseline performance before fine-tuning. |
| Fine-tuned (global)        | 0.883 | 0.800    | 0.950    | 0.950    | 0.635          | Better ranking and stronger semantic similarity after fine-tuning. |
| Base multilingual [en]    | 0.900 | 0.800    | 1.000    | 1.000    | 0.558          | Solid performance in English before fine-tuning. |
| Fine-tuned [en]           | 0.950 | 0.900    | 1.000    | 1.000    | 0.656          | Significant improvement in English ranking and similarity. |
| Base multilingual [fr]    | 0.789 | 0.700    | 0.900    | 0.900    | 0.531          | Lower baseline scores in French compared to English. |
| Fine-tuned [fr]           | 0.817 | 0.700    | 0.900    | 0.900    | 0.614          | Better semantic similarity, stable ranking; French still improves. |

## Notes
- If your fine-tuned HF repo is private, ensure `huggingface-cli login` (or set `HUGGINGFACE_HUB_TOKEN`).
- To swap models, edit `MODEL_PATHS` in `faq_compare_streamlit.py` and `model_path` in `faq_streamlit.py`.
