import json
from typing import Dict, List, Tuple

import streamlit as st
from sentence_transformers import SentenceTransformer, util

# FAQ sources per language
FAQ_DATA_PATHS = {
    "English": "data/faq_source_en.jsonl",
    "Français": "data/faq_source_fr.jsonl",
}

# Models to compare
MODEL_PATHS = {
    "Base multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Fine-tuned (AutoTrain)": "arnoweb/model-faq-sentence-autotrain",
}


@st.cache_resource(show_spinner=False)
def load_model(path: str) -> SentenceTransformer:
    return SentenceTransformer(path)


@st.cache_data(show_spinner=False)
def load_faq_data(faq_data_path: str) -> Tuple[List[str], List[str]]:
    faq_items = []
    with open(faq_data_path, "r") as f:
        for line in f:
            if line.strip():
                faq_items.append(json.loads(line))
    faq_questions = [item["query"] for item in faq_items]
    faq_answers = [item["answer"] for item in faq_items]
    return faq_questions, faq_answers


@st.cache_data(show_spinner=False)
def compute_embeddings(model_name: str, model_path: str, texts: List[str]):
    # Use model_path to ensure the cache key is hashable; load_model is already cached.
    model = load_model(model_path)
    return model.encode(texts, convert_to_tensor=True)


def run_search(
    model: SentenceTransformer,
    query: str,
    answers: List[str],
    embeddings,
    top_k: int,
) -> Dict:
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_k_indices = similarities.topk(top_k).indices.tolist()
    return {
        "indices": top_k_indices,
        "similarities": similarities,
    }


st.title("FAQ Retrieval Comparison (Base vs Fine-tuned)")
st.write(
    "Compare a pre-trained multilingual encoder with your fine-tuned AutoTrain model on the same FAQ data."
)

language = st.radio("Language / Langue", ["English", "Français"], horizontal=True)
top_k = st.slider("Top K", min_value=1, max_value=5, value=3, step=1)
similarity_threshold = st.slider(
    "Similarity threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01,
    help="If the best match is below this value, the model is considered uncertain.",
)

faq_data_path = FAQ_DATA_PATHS[language]
faq_questions, faq_answers = load_faq_data(faq_data_path)

# Prepare models and embeddings
models = {name: load_model(path) for name, path in MODEL_PATHS.items()}
embeddings_by_model = {
    name: compute_embeddings(name, MODEL_PATHS[name], faq_answers)
    for name in models
}

user_query = st.text_input(
    "Search the FAQ (e.g., “track my order”):"
    if language == "English"
    else "Recherchez dans la FAQ (ex. « suivre ma commande ») :",
    placeholder="Refund policy, shipping cost, change email…"
    if language == "English"
    else "Politique de remboursement, frais de livraison, changer d'email…",
)

if user_query:
    cols = st.columns(len(models))
    for col, (model_name, model_obj) in zip(cols, models.items()):
        with col:
            st.subheader(model_name)
            search_results = run_search(
                model_obj,
                user_query,
                faq_answers,
                embeddings_by_model[model_name],
                top_k=top_k,
            )

            top_indices = search_results["indices"]
            similarities = search_results["similarities"]

            if not top_indices:
                st.write("No results.")
                continue

            best_score = similarities[top_indices[0]].item()
            if best_score < similarity_threshold:
                st.warning(
                    "No confident match (score below threshold)."
                    if language == "English"
                    else "Pas de correspondance sûre (score sous le seuil)."
                )
            for rank, idx in enumerate(top_indices, start=1):
                st.markdown(f"**Rank {rank}**")
                st.markdown(f"**Q:** {faq_questions[idx]}")
                st.markdown(f"**A:** {faq_answers[idx]}")
                st.markdown(f"Similarity: {similarities[idx].item():.2f}")
                st.markdown("---")
else:
    st.info(
        "Search the FAQ (e.g., “refund shipping”, “change password”) to compare rankings."
        if language == "English"
        else "Cherchez dans la FAQ (ex. « frais de retour », « changer le mot de passe ») pour comparer les résultats."
    )
