import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
DATA_DIR = "./data/processed"
MODELS_DIR = "./models"
CSV_PATH = f"{DATA_DIR}/movies_cleaned.csv"
EMBEDDINGS_PATH = f"{MODELS_DIR}/embeddings.npy"
FAISS_PATH = f"{MODELS_DIR}/movie_index.faiss"

default_poster = "https://i.ibb.co/pHxqDX6/2ebfe3fcf82a4c6ccac494de2306a357.jpg"  # Placeholder image for missing posters

# Load files
st.set_page_config(
    page_title="IntelliFlix: Semantic Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🎬IntelliFlix: Semantic Movie Recommender")
st.markdown(
    "Find similar movies based on plot descriptions using Sentence Transformers + FAISS."
)


@st.cache_resource
def load_model_and_data():
    model = SentenceTransformer("all-MiniLM-L12-v2")
    df = pd.read_csv(CSV_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    # Load FAISS index
    index = faiss.read_index(FAISS_PATH)

    return model, df, embeddings, index


model, df, embeddings, index = load_model_and_data()
device = "cuda" if torch.cuda.is_available() else "cpu"


# Search function
def semantic_search(query, k=15):
    query_vec = model.encode([query], convert_to_tensor=True, device=device)
    query_np = query_vec.cpu().numpy().astype("float32")

    D, I = index.search(query_np, k)
    similarities = cosine_similarity(query_np, embeddings[I[0]])[0]

    results = df.iloc[I[0]].copy()
    results["similarity"] = similarities
    results["poster_url"] = results["poster_path"].apply(
        lambda path: (
            f"https://image.tmdb.org/t/p/w500{path}" if pd.notnull(path) else None
        )
    )
    return results.sort_values(by="similarity", ascending=False).reset_index(drop=True)


# Input box
query = st.text_input(
    "🔍 Enter a movie plot or description:",
    "An adventure of explorers lost in space for a wormhole and tries to survive on a distant planet.",
).strip()


if query:
    with st.spinner("Finding similar movies..."):
        results = semantic_search(query)

    st.subheader(f"🔝 Top {len(results)} similar movies:")

for _, row in results.iterrows():
    col1, col2 = st.columns([1, 4])

    with col1:
        st.image(row["poster_url"] or default_poster, width=150)

    with col2:
        st.markdown(
            f"### {row['title']} ({row['release_date'][:4] if pd.notnull(row['release_date']) else 'N/A'})"
        )
        st.markdown(f"**Similarity:** `{row['similarity']:.4f}`")
        st.markdown(f"**Overview:** {row['overview']}")

    st.markdown("---")
st.markdown("Built with ❤️ by [Arghya Ghosh](https://arghya.dev) using FAISS + Sentence Transformers")
