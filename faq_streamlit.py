import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import requests

model_path = "model-faq-sentence-autotrain"

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
            "En t'appuyant strictement sur les documents ci-dessus, rédige une réponse claire et concise à la question de l'utilisateur. "
            "Si l'information n'est pas présente dans les documents, réponds simplement : 'Je ne sais pas.' Ne fais pas d'hypothèses et n'invente pas de réponses."
        )
    else:
        intro = "Here are some excerpts from our FAQ:\n"
        docs = "\n\n".join([
            f"Document {i+1}:\nQuestion: {q}\nAnswer: {a}" for i, (q, a) in enumerate(zip(retrieved_questions, retrieved_faqs))
        ])
        question = f"\n\nUser question: {user_question}\n"
        instruction = (
            "Based strictly on the above documents, write a clear and concise answer to the user's question. "
            "If the information is not present in the documents, simply answer: 'I don't know.' Do not make assumptions or invent answers."
        )
    return f"{intro}{docs}{question}{instruction}"


def query_local_llm(prompt, model="deepseek-r1-distill-qwen-7b"):
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

st.title("FAQ Top-3 Answer Finder")

# Language switcher
language = st.radio("Select language / Choisissez la langue:", ["English", "Français"])

faq_data_path = faq_data_paths[language]
model = load_model()
faq_questions, faq_answers = load_faq_data(faq_data_path)
answer_embeddings = compute_answer_embeddings(model, faq_answers)

st.write(
    "Enter your question and see the top 3 answers from the FAQ model." if language == "English" else "Posez votre question et ayez les 3 meilleurs résultats :")

user_query = st.text_input("Ask your question:" if language == "English" else "Posez votre question :")

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
            st.markdown(f"**Rank {rank}:**" if language == "English" else f"**Rang {rank} :**")
            st.markdown(f"**Question:** {faq_questions[idx]}")
            st.markdown(f"**Answer:** {faq_answers[idx]}")
            st.markdown(f"**Similarity:** {similarities[idx].item():.2f}")
            st.markdown("---")

        # RAG section
        if st.button("Générer une réponse avec le LLM" if language == "Français" else "Generate answer with LLM"):
            prompt = build_rag_prompt(user_query, retrieved_faqs, retrieved_questions, language=language)
            with st.spinner("Appel au LLM local..."):
                llm_response = query_local_llm(prompt)
            st.markdown("### Réponse générée :" if language == "Français" else "### Generated answer:")
            st.write(llm_response)