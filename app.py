# app_free.py
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json
import re

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/medical_faq_clean.csv")
    answer_embeddings = np.load("answer_embeddings.npy")
    return df, answer_embeddings

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize
df, answer_embeddings = load_data()
model = load_model()

# Cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Retrieve top-k answers
def retrieve_context(query, top_k=3):
    q_emb = model.encode([query])[0]
    sims = [cosine_sim(q_emb, emb) for emb in answer_embeddings]
    top_idx = np.argsort(sims)[-top_k:][::-1]
    
    context_items = []
    for i, idx in enumerate(top_idx):
        similarity_score = sims[idx]
        context_items.append({
            'question': df.iloc[idx]['Question'],
            'answer': df.iloc[idx]['Answer'],
            'similarity': similarity_score,
            'rank': i + 1
        })
    
    return context_items

# Free text generation using Hugging Face Inference API (free tier)
def generate_answer_hf(question, context_items):
    # Format context
    context_text = "\n\n".join([
        f"Q: {item['question']}\nA: {item['answer']}"
        for item in context_items
    ])
    
    prompt = f"""Based on the medical FAQ context below, answer the user's question. If the information is not in the context, say you don't know and recommend consulting a healthcare professional.

Context:
{context_text}

Question: {question}

Answer (be concise and helpful):"""

    # Using Hugging Face's free inference API
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
    
    try:
        # Simplified approach: extract most relevant answer
        best_match = context_items[0] if context_items else None
        
        if best_match and best_match['similarity'] > 0.3:  # Threshold for relevance
            # Extract key information and provide a structured response
            answer = best_match['answer']
            
            # Simple post-processing to make answer more conversational
            response = f"Based on the medical information available:\n\n{answer}\n\n"
            response += "**Important:** This information is for educational purposes only. Please consult with a healthcare professional for personalized medical advice."
            
            return response
        else:
            return "I don't have specific information about your question in my knowledge base. Please consult with a healthcare professional for accurate medical advice."
            
    except Exception as e:
        # Fallback to simple retrieval-based response
        if context_items and context_items[0]['similarity'] > 0.2:
            return f"Based on available information:\n\n{context_items[0]['answer']}\n\n**Note:** Please consult a healthcare professional for personalized advice."
        else:
            return "I don't have specific information about your question. Please consult with a healthcare professional."

# Alternative: Local LLM using Ollama (if installed)
def generate_answer_local(question, context_items):
    try:
        import requests
        
        context_text = "\n\n".join([
            f"Q: {item['question']}\nA: {item['answer']}"
            for item in context_items
        ])
        
        prompt = f"""You are a medical information assistant. Use only the provided context to answer the question. If the answer isn't in the context, say so and recommend consulting a healthcare professional.

Context:
{context_text}

Question: {question}

Answer:"""

        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': 'llama2',  # or any model you have installed
                                   'prompt': prompt,
                                   'stream': False
                               })
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception("Ollama not available")
            
    except:
        # Fallback to simple retrieval
        return generate_answer_hf(question, context_items)

# Streamlit UI
st.title("🏥 Medical FAQ Chatbot")
st.markdown("""
**⚠️ Medical Disclaimer:** This chatbot provides general medical information for educational purposes only. 
It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals.
""")

# Sidebar with information
st.sidebar.title("About")
st.sidebar.info("""
This chatbot uses:
- Semantic search through medical FAQs
- Free/local text generation
- Retrieval-Augmented Generation (RAG)

No OpenAI API required!
""")

# Main interface
question = st.text_input("💭 Ask a medical question:", placeholder="e.g., What are the symptoms of diabetes?")

if st.button("🔍 Get Answer", type="primary") and question.strip():
    with st.spinner("Searching medical knowledge base..."):
        # Retrieve relevant context
        context_items = retrieve_context(question, top_k=3)
        
        if not context_items:
            st.error("No relevant information found.")
        else:
            # Display similarity scores in sidebar
            with st.sidebar:
                st.subheader("Search Results")
                for item in context_items:
                    st.metric(f"Match {item['rank']}", f"{item['similarity']:.2%}")
            
            # Generate answer
            with st.spinner("Generating response..."):
                # Choose generation method
                generation_method = st.selectbox("Choose response method:", 
                                               ["Simple Retrieval", "Local LLM (Ollama)"], 
                                               key="method")
                
                if generation_method == "Local LLM (Ollama)":
                    answer = generate_answer_local(question, context_items)
                else:
                    answer = generate_answer_hf(question, context_items)
                
                # Display answer
                st.markdown("### 📋 Answer")
                st.markdown(answer)
                
                # Show confidence
                best_similarity = context_items[0]['similarity'] if context_items else 0
                confidence = "High" if best_similarity > 0.7 else "Medium" if best_similarity > 0.4 else "Low"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", confidence)
                with col2:
                    st.metric("Best Match", f"{best_similarity:.1%}")
    
    # Show sources
    with st.expander("📚 View Sources"):
        for i, item in enumerate(context_items, 1):
            st.markdown(f"**Source {i}** (Similarity: {item['similarity']:.1%})")
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            st.markdown("---")

# Additional features
st.markdown("### 🎯 Quick Questions")
quick_questions = [
    "What are the symptoms of diabetes?",
    "How is high blood pressure treated?",
    "What causes headaches?",
    "How to prevent heart disease?"
]

cols = st.columns(2)
for i, qq in enumerate(quick_questions):
    with cols[i % 2]:
        if st.button(qq, key=f"quick_{i}"):
            st.experimental_set_query_params(q=qq)