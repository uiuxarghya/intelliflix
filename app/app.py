import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download

# Configure environment paths to avoid permission issues
os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['STREAMLIT_SERVER_ROOT'] = '/tmp/streamlit'

REPO_ID = "uiuxarghya/intelliflix-store"
default_poster = "https://i.ibb.co/pHxqDX6/2ebfe3fcf82a4c6ccac494de2306a357.jpg"

# Set up Streamlit page configuration
st.set_page_config(
    page_title="IntelliFlix: Semantic Movie Recommender",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/uiuxarghya/intelliflix/issues',
        'Report a bug': 'https://github.com/uiuxarghya/intelliflix/issues',
        'About': (
            "IntelliFlix is a powerful movie recommender built with Sentence Transformers "
            "and FAISS. It helps you find similar movies based on plot descriptions. Built by "
            "[Arghya Ghosh](https://arghya.dev). Try out the app and discover your next favorite movie!"
        )
    }
)

# Create necessary directories if they don't exist
os.makedirs('/tmp/huggingface', exist_ok=True)
os.makedirs('/tmp/streamlit', exist_ok=True)

st.title("🎬 IntelliFlix: Semantic Movie Recommender")
st.markdown("Find similar movies based on plot descriptions using Sentence Transformers + FAISS.")

@st.cache_resource(show_spinner=False)
def load_model_and_data():
    try:
        # Load model with explicit cache directory
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L12-v2",
            cache_folder='/tmp/huggingface'
        )

        # Load files from Hugging Face Hub
        with st.spinner("Loading movie data..."):
            csv_path = hf_hub_download(
                repo_id=REPO_ID,
                filename="data/tmdb_movies_dataset_processed.csv",
                repo_type="dataset",
                cache_dir='/tmp/huggingface'
            )
            df = pd.read_csv(csv_path)

        with st.spinner("Loading embeddings..."):
            embeddings_path = hf_hub_download(
                repo_id=REPO_ID,
                filename="embeddings/movie_ovierview_embeddings.npy",
                repo_type="dataset",
                cache_dir='/tmp/huggingface'
            )
            embeddings = np.load(embeddings_path)

        with st.spinner("Loading FAISS index..."):
            faiss_path = hf_hub_download(
                repo_id=REPO_ID,
                filename="indexes/movie_overview_index.faiss",
                repo_type="dataset",
                cache_dir='/tmp/huggingface'
            )
            index = faiss.read_index(faiss_path)

        return model, df, embeddings, index

    except Exception as e:
        st.error(f"Error loading model or data: {str(e)}")
        st.stop()

model, df, embeddings, index = load_model_and_data()
device = "cuda" if torch.cuda.is_available() else "cpu"

def semantic_search(query, k=15):
    try:
        query_vec = model.encode([query], convert_to_tensor=True, device=device)
        query_np = query_vec.cpu().numpy().astype("float32")

        D, I = index.search(query_np, k)
        similarities = cosine_similarity(query_np, embeddings[I[0]])[0]

        results = df.iloc[I[0]].copy()
        results["similarity"] = similarities
        results["poster_url"] = results["poster_path"].apply(
            lambda path: f"https://image.tmdb.org/t/p/w500{path}" if pd.notnull(path) else default_poster
        )
        return results.sort_values(by="similarity", ascending=False).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return pd.DataFrame()

# Main UI
query = st.text_input(
    "🔍 Enter a movie plot or description:",
    "An adventure of explorers lost in space for a wormhole and tries to survive on a distant planet.",
    help="Describe a movie plot or theme to find similar movies"
).strip()

if query:
    with st.spinner("Finding similar movies..."):
        results = semantic_search(query)

    if not results.empty:
        st.subheader(f"🔝 Top {len(results)} similar movies:")

        cols = st.columns(3)
        for idx, (_, row) in enumerate(results.iterrows()):
            with cols[idx % 3]:
                st.image(
                    row["poster_url"],
                    width=200,
                    caption=f"{row['title']} ({row['release_date'][:4]})"
                )
                with st.expander(f"Similarity: {row['similarity']:.2f}"):
                    st.write(row['overview'])
    else:
        st.warning("No results found. Try a different query.")

# Footer
st.markdown("---")
st.markdown("""
    **Built with** ❤️ **by [Arghya Ghosh](https://arghya.dev)**
    *Technologies used: FAISS + Sentence Transformers + Streamlit*
""")