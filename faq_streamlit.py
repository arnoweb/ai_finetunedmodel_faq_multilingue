import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import os
import requests
from dotenv import load_dotenv

load_dotenv()

st.markdown(
    """
    <style>
    .result-tight p { margin: 0 0 4px 0; line-height: 1.25; }
    .result-tight .sim { color: #6b7280; font-size: 0.85rem; margin: 0; }
    .result-tight hr { margin: 2px 0; }
    hr { margin: 2px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"

model_path = "arnoweb/model-faq-sentence-autotrain"

# File paths for each language
faq_data_paths = {
    "English": "data/faq_source_en.jsonl",
    "Français": "data/faq_source_fr.jsonl"
}

@st.cache_resource
def load_model():
    return SentenceTransformer(model_path)

@st.cache_data
def load_faq_data(faq_data_path):
    faq_items = []
    with open(faq_data_path, "r") as f:
        for line in f:
            if line.strip():
                faq_items.append(json.loads(line))
    faq_questions = [item["query"] for item in faq_items]
    faq_answers = [item["answer"] for item in faq_items]
    return faq_questions, faq_answers

@st.cache_data
def compute_answer_embeddings(_model, faq_answers):
    return _model.encode(faq_answers, convert_to_tensor=True)

def build_rag_prompt(user_question, retrieved_faqs, retrieved_questions, language="fr"):
    if language == "Français":
        intro = "Voici des extraits de notre FAQ :\n"
        docs = "\n\n".join([
            f"Document {i+1} :\nQuestion : {q}\nRéponse : {a}" for i, (q, a) in enumerate(zip(retrieved_questions, retrieved_faqs))
        ])
        question = f"\n\nQuestion utilisateur : {user_question}\n"
        instruction = (
            "En t'appuyant strictement sur les documents ci-dessus, rédige une réponse claire et détaillée (3 à 5 phrases) à la question de l'utilisateur. "
            "Si l'information n'est pas présente dans les documents, réponds simplement : 'Je ne sais pas.' Ne fais pas d'hypothèses et n'invente pas de réponses."
        )
    else:
        intro = "Here are some excerpts from our FAQ:\n"
        docs = "\n\n".join([
            f"Document {i+1}:\nQuestion: {q}\nAnswer: {a}" for i, (q, a) in enumerate(zip(retrieved_questions, retrieved_faqs))
        ])
        question = f"\n\nUser question: {user_question}\n"
        instruction = (
            "Based strictly on the above documents, write a clear, moderately detailed answer (3–5 sentences) to the user's question. "
            "If the information is not present in the documents, simply answer: 'I don't know.' Do not make assumptions or invent answers."
        )
    return f"{intro}{docs}{question}{instruction}"


def query_local_llm(prompt, model="qwen/qwen3-vl-4b"):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 512
    }
    response = requests.post("http://localhost:1234/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def query_remote_llm(prompt, model: str = HF_MODEL_DEFAULT, temperature: float = 0.2, max_tokens: int = 800):
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY is not set in the environment.")

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Hugging Face API error: {response.text}") from exc

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response format from Hugging Face: {data}") from exc



st.title("E-commerce FAQ RAG Search")

# Language switcher
language = st.radio("Select language / Choisissez la langue:", ["English", "Français"])

faq_data_path = faq_data_paths[language]
model = load_model()
faq_questions, faq_answers = load_faq_data(faq_data_path)
answer_embeddings = compute_answer_embeddings(model, faq_answers)

st.write(
    "Enter your question and see the top 3 answers from the FAQ model."
    if language == "English"
    else "Posez votre question et ayez les 3 meilleurs résultats :"
)

placeholder_text = (
    "e.g., reset password, change shipping address, refund policy"
    if language == "English"
    else "ex. réinitialiser mot de passe, changer adresse de livraison, politique de remboursement"
)

user_query = st.text_input(
    "Ask your question:" if language == "English" else "Posez votre question :",
    placeholder=placeholder_text,
)

if user_query:
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, answer_embeddings)[0]
    top_k = 3
    top_k_indices = similarities.topk(top_k).indices.tolist()
    top_score = similarities[top_k_indices[0]].item()
    SIMILARITY_THRESHOLD = 0.1  # You can adjust this value

    if top_score < SIMILARITY_THRESHOLD:
        st.warning("No relevant answer found. Please try rephrasing your question." if language == "English" else "Aucune réponse pertinente trouvée. Veuillez reformuler votre question.")
    else:
        st.subheader("Top 3 Answers:" if language == "English" else "Top 3 réponses :")
        retrieved_faqs = [faq_answers[idx] for idx in top_k_indices]
        retrieved_questions = [faq_questions[idx] for idx in top_k_indices]
        for rank, idx in enumerate(top_k_indices, 1):
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
            if rank < top_k:
                st.markdown("<hr>", unsafe_allow_html=True)

        # RAG section
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        if st.button("Générer une réponse avec le LLM" if language == "Français" else "Generate answer with LLM", key="llm_button", help=None):
            prompt = build_rag_prompt(user_query, retrieved_faqs, retrieved_questions, language=language)
            with st.spinner("Call of the LLM..."):
                try:
                    llm_response = query_remote_llm(prompt)
                except Exception as exc:
                    st.error(f"LLM call failed: {exc}")
                    llm_response = None
            if llm_response:
                low_conf = llm_response.strip().lower()
                if ("i don't know" in low_conf) or ("je ne sais pas" in low_conf):
                    st.warning(
                        "Aucune réponse trouvée. Réessayez en reformulant votre question."
                        if language == "Français"
                        else "No answer found. Try rephrasing your question."
                    )
                else:
                    st.markdown("### Réponse générée :" if language == "Français" else "### Generated answer:")
                    st.write(llm_response)

st.markdown(
    'Made by <a href="https://www.linkedin.com/in/bretonarnaud/" target="_blank">Arnaud BRETON</a>',
    unsafe_allow_html=True,
)
