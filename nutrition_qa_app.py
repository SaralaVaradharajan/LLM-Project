import os
import faiss
import json
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from io import StringIO

#function

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to("cpu")
    return embed_model, gen_tokenizer, gen_model

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_chunks(chunks, embed_model):
    return embed_model.encode(chunks, convert_to_numpy=True)

def search_faiss_index(query_embedding, faiss_index, text_chunks, top_k=5):
    D, I = faiss_index.search(np.array([query_embedding]), top_k)
    return [text_chunks[i] for i in I[0]]

def generate_answer(context_chunks, question, tokenizer, model):
    context = "\n".join(context_chunks)
    prompt = f"Answer the question in detail based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}"

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=250, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Streamlit UI


st.set_page_config(page_title=" Nutrition & Diet Q&A", layout="centered")
st.title(" Nutrition & Diet: Q&A Assistant")
uploaded_file = st.file_uploader(" Upload a `.txt` document on Nutrition & Diet", type="txt")


query = st.text_input("Ask a question about the uploaded document")

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    chunks = chunk_text(raw_text)

    # Load models
    embed_model, tokenizer, model = load_models()

    # Embed and create FAISS index
    embeddings = embed_chunks(chunks, embed_model)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    if query:
        query_embedding = embed_model.encode([query])[0]
        top_chunks = search_faiss_index(query_embedding, index, chunks)
        answer = generate_answer(top_chunks, query, tokenizer, model)

        st.markdown(f"### ✅ Answer:")
        st.write(answer)

        # Optionally save
        with open("generated_answer.txt", "w") as f:
            f.write(f"Question: {query}\nAnswer: {answer}")
        st.download_button("⬇️ Download Answer", data=open("generated_answer.txt", "rb").read(),
                           file_name="answer.txt", mime="text/plain")

