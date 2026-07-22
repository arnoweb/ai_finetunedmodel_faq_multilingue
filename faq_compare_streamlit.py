import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import streamlit.components.v1 as components
from streamlit_searchbox import st_searchbox
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv()

DOCS_DIR = Path(__file__).resolve().parent / "docs"
try:
    _secret_hf_key = st.secrets.get("HF_API_KEY", "")
except Exception:
    _secret_hf_key = ""
HF_API_KEY = os.getenv("HF_API_KEY") or _secret_hf_key
os.environ.setdefault("HF_TOKEN", HF_API_KEY or "")

# FAQ sources per language
FAQ_DATA_PATHS = {
    "English": "data/faq_source_en.jsonl",
    "Français": "data/faq_source_fr.jsonl",
}

# Models to compare — same base encoder, before and after fine-tuning.
MODEL_PATHS = {
    "Pre-fine-tuning": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "Fine-tuned (AutoTrain)": "arnoweb/model-faq-sentence-autotrain",
}

st.markdown(
    """
    <style>
    .hero-blob { position: fixed; z-index: -1; border-radius: 50%; filter: blur(70px); opacity: 0.25; pointer-events: none; }
    .blob-a { width: 320px; height: 320px; background: #4F46E5; top: -80px; left: -60px; }
    .blob-b { width: 280px; height: 280px; background: #22D3EE; top: -40px; right: -60px; }

    h1 { color: #4F46E5 !important; font-weight: 800 !important; letter-spacing: -0.02em; }

    .trust-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.4rem 0 1rem; }
    .trust-pill {
        display: inline-flex; align-items: center; gap: 0.35rem;
        background: #F0FDF4; color: #14532D; border: 1px solid #4ADE80;
        border-radius: 100px; padding: 0.3rem 0.75rem; font-size: 0.82rem; font-weight: 600;
    }

    .result-tight {
        background: #FFFFFF; border: 1px solid #DDE1E8; border-radius: 12px;
        padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; box-shadow: 0 1px 2px rgba(15,23,42,0.06);
    }
    .result-tight p { margin: 0 0 4px 0; line-height: 1.25; }
    .result-tight .sim {
        display: inline-block; margin-top: 6px; color: #14532D; background: #F0FDF4;
        border: 1px solid #4ADE80; border-radius: 100px; padding: 2px 10px;
        font-size: 0.78rem; font-weight: 600;
    }

    div[data-testid="stTextInput"] input {
        border: 2px solid #4F46E5 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1.05rem !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.08);
    }
    div[data-testid="stTextInput"] input:focus {
        box-shadow: 0 0 0 3px rgba(79,70,229,0.18) !important;
    }
    div[data-testid="stTextInput"] label { font-weight: 600 !important; }
    hr { margin: 2px 0; }
    </style>
    <div class="hero-blob blob-a"></div>
    <div class="hero-blob blob-b"></div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(path: str) -> SentenceTransformer:
    return SentenceTransformer(path)


@st.cache_data(show_spinner=False, ttl=300)
def load_faq_data(faq_data_path: str) -> Tuple[List[str], List[str]]:
    faq_items = []
    with open(faq_data_path, "r") as f:
        for line in f:
            if line.strip():
                faq_items.append(json.loads(line))
    faq_questions = [item["query"] for item in faq_items]
    faq_answers = [item["answer"] for item in faq_items]
    return faq_questions, faq_answers


@st.cache_data(show_spinner=False, ttl=300)
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


st.title("E-commerce FAQ Retrieval Comparison (Base vs Fine-tuned)")
st.caption("Comparez concrètement l'effet du fine-tuning sur la qualité de la recherche FAQ.")
st.markdown(
    "**Pre-fine-tuning:** `paraphrase-multilingual-mpnet-base-v2`, unmodified — the exact checkpoint "
    "the fine-tuned model started from, never trained on this FAQ.  \n"
    "**Fine-tuned:** `arnoweb/model-faq-sentence-autotrain` — that same checkpoint, fine-tuned on this "
    "FAQ's own questions and answers."
)
st.write(
    "Type a question and compare both sides: same encoder, before and after fine-tuning. Any difference "
    "you see is the isolated effect of fine-tuning itself — not a confound from a different base model."
)

with st.expander("What do Top K and Similarity threshold control?"):
    st.markdown(
        "**Top K** — how many ranked results are shown per model.\n"
        "- Higher → a more forgiving test: checks whether the correct answer appears *anywhere* in the list, even if not ranked first.\n"
        "- Lower → a stricter test, closer to real usage: most users only read the first result or two.\n\n"
        "**Similarity threshold** — the minimum score required before a result is trusted.\n"
        "- Higher → fewer answers shown, but each is more likely to be genuinely relevant (more refusals, less risk of a wrong match).\n"
        "- Lower → more answers shown, but some may be weak or irrelevant matches (fewer refusals, higher risk of a bad match slipping through).\n\n"
        "**Combining both**\n\n"
        "| | High threshold | Low threshold |\n"
        "|---|---|---|\n"
        "| **High Top K** | Strict but thorough: reveals whether the correct answer exists among many candidates, but only shows it if confidently ranked | Most permissive: shows many candidates even when uncertain — good for auditing near-misses and noise |\n"
        "| **Low Top K** | Strictest setup, closest to a production chatbot that only answers when sure — expect more refusals | Always shows its single best guess, right or wrong — reveals raw ranking quality with no safety net |\n"
    )

language = st.radio("Language / Langue", ["Français", "English"], horizontal=True)

st.markdown(
    f"""
    <div class="trust-pills">
      <span class="trust-pill">✓ {"No data sent to a third party for search" if language == "English" else "Aucune donnée envoyée à un tiers pour la recherche"}</span>
      <span class="trust-pill">✓ {"Refuses to answer rather than invent" if language == "English" else "Refuse de répondre plutôt que d'inventer"}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

top_k = st.slider(
    "Top K",
    min_value=1,
    max_value=5,
    value=3,
    step=1,
    help="How many ranked results to show per model.",
)
similarity_threshold = st.slider(
    "Similarity threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01,
    help="The minimum similarity score required before a result is trusted. Below it, the model reports no confident match instead of guessing.",
)

faq_data_path = FAQ_DATA_PATHS[language]
faq_questions, faq_answers = load_faq_data(faq_data_path)

if not HF_API_KEY:
    st.error(
        "Missing HF_API_KEY: the fine-tuned model repo is private and no Hugging Face "
        "token was found (checked env var HF_API_KEY and Streamlit secrets). On Streamlit "
        "Cloud, add HF_API_KEY under this app's Settings → Secrets, then save to trigger a reboot."
    )
    st.stop()

# Prepare models and embeddings
models = {name: load_model(path) for name, path in MODEL_PATHS.items()}
embeddings_by_model = {
    name: compute_embeddings(name, MODEL_PATHS[name], faq_answers)
    for name in models
}

def _capture_query(searchterm: str):
    st.session_state["compare_last_query"] = searchterm
    return []  # live-typing capture only; results are shown as columns below, not a dropdown


st_searchbox(
    _capture_query,
    placeholder="Refund policy, shipping cost, change email…"
    if language == "English"
    else "Politique de remboursement, frais de livraison, changer d'email…",
    label="Search the FAQ (e.g., “track my order”):"
    if language == "English"
    else "Recherchez dans la FAQ (ex. « suivre ma commande ») :",
    key=f"compare_searchbox_{language}",
)

user_query = st.session_state.get("compare_last_query", "")

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
                st.markdown(
                    f"""
                    <div class="result-tight">
                        <p><strong>{'Rank' if language == 'English' else 'Rang'} {rank}:</strong> {faq_questions[idx]}</p>
                        <p>{faq_answers[idx]}</p>
                        <p class="sim">Similarity: {similarities[idx].item():.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
else:
    st.info(
        "Search the FAQ (e.g., “refund shipping”, “change password”) to compare rankings."
        if language == "English"
        else "Cherchez dans la FAQ (ex. « frais de retour », « changer le mot de passe ») pour comparer les résultats."
    )

st.divider()
st.markdown(
    "Want the full search + generated-answer experience? "
    "[Try the main FAQ RAG search app](https://arnoweb-rag-llm-faq-finetuned-huggingface.streamlit.app/)."
)

with st.expander("Business value & use cases"):
    bv_language = st.radio("Language / Langue", ["Français", "English"], horizontal=True, key="bv_lang_compare")
    bv_file = "business-value-en.html" if bv_language == "English" else "business-value.html"
    components.html((DOCS_DIR / bv_file).read_text(encoding="utf-8"), height=6000, scrolling=True)

with st.expander("Technical architecture"):
    components.html((DOCS_DIR / "architecture.html").read_text(encoding="utf-8"), height=6000, scrolling=True)
