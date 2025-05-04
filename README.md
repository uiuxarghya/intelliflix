# 🎬 IntelliFlix – Semantic Movie Recommender App

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?logo=streamlit)](https://streamlit.io/)
[![Hugging Face Spaces](https://img.shields.io/badge/Hosted%20on-Hugging%20Face%20Spaces-yellow?logo=huggingface)](https://huggingface.co/spaces/uiuxarghya/intelliflix)
[![HF Dataset](https://img.shields.io/badge/Dataset-uiuxarghya%2Fintelliflix--store-green?logo=huggingface)](https://huggingface.co/datasets/uiuxarghya/intelliflix-store)
[![Last Commit](https://img.shields.io/github/last-commit/uiuxarghya/intelliflix)](https://github.com/uiuxarghya/intelliflix/commits/main)
[![GitHub stars](https://img.shields.io/github/stars/uiuxarghya/intelliflix?style=social)](https://github.com/uiuxarghya/intelliflix/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/uiuxarghya/intelliflix?style=social)](https://github.com/uiuxarghya/intelliflix/network/members)

**IntelliFlix** is a semantic movie recommender app powered by **Sentence Transformers** and **FAISS**, with a clean UI built in **Streamlit**. It finds similar movies based on plot descriptions using powerful NLP embeddings and fast vector search.

## 🚀 Features

- 🧠 Semantic search based on movie plot summaries
- ⚡ Fast and scalable search via **FAISS**
- ✨ Embeddings and index hosted on Hugging Face Datasets
- 🖥️ UI hosted on **Hugging Face Spaces**
- 🔍 Search any movie plot snippet to get relevant recommendations
- 📊 Visualize results with a clean and interactive UI

## 🌐 Live Demo

👉 Try it now: [**IntelliFlix App**](https://uiuxarghya-intelliflix.hf.space)

## 🛠️ Tech Stack

- 🐍 **Python** – Core programming language
- ⛵ **Streamlit** – UI framework for interactive web apps
- 💬 **Sentence Transformers** (`all-MiniLM-L12-v2`) – For semantic embeddings
- ⚡ **FAISS** – Fast similarity search and indexing
- 🤗 **Hugging Face Datasets & Spaces** – For storing and deploying models/data
- 📊 **pandas**, **NumPy** – Data manipulation and analysis
- 🧠 **scikit-learn** – Model training and evaluation

## 📂 Project Structure

```
📁 intelliflix
│
├── app/ → [hosted]                  # Streamlit app files (on HF Spaces)
│   ├── app.py                       # Streamlit app script
│   ├── requirements.txt             # Streamlit app specific dependencies
│   ├── README.md                    # App-specific README for Spaces
│   └── .gitattributes               # Required for HF Spaces
│
├── requirements.txt                 # Root dependencies for local dev / CI
├── README.md                        # Main project README
│
├── data/ → [stored remotely]        # Movie metadata (on HF Datasets)
├── embeddings/ → [stored remotely]  # SentenceTransformer embeddings (on HF Datasets)
├── indexes/ → [stored remotely]     # FAISS index files (on HF Datasets)
│
├── notebooks/                       # Jupyter notebooks for exploration/training
    ├── 01-movies_exploration.ipynb
    └── 02-semantic_movie_recommender.ipynb
```

## 💾 Data & Model Hosting

- 🧠 **Embeddings & FAISS index** are stored in:
  👉 [uiuxarghya/intelliflix-store ](https://huggingface.co/datasets/uiuxarghya/intelliflix-store)

- 🛰️ Streamlit app runs on:
  👉 [uiuxarghya/intelliflix (HF Space)](https://huggingface.co/spaces/uiuxarghya/intelliflix)

## 🧪 Run Locally

1. **Clone the repo**

   ```bash
   git clone https://github.com/uiuxarghya/intelliflix.git
   cd intelliflix
   ```

2. **(Optional) Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

_Note: The app will automatically download embeddings and FAISS index from Hugging Face on first run._

## ✅ Roadmap

- [x] 🎥 Trailer/poster preview via TMDB API
- [ ] 🗂️ Genre or actor-based filtering
- [ ] 💬 Natural language query support (e.g., "movies like Inception but romantic")
- [ ] 🔍 Search by title, genre, or actor
- [ ] 📅 Release year filtering
- [ ] 📅 Release date sorting
- [ ] 🔄 Feedback loop to improve recommendations
- [ ] 📊 Show explanation of similarity scoring

## 🙌 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face Datasets & Spaces](https://huggingface.co/)
- [TMDB](https://www.themoviedb.org/)

## License

[![AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue?logo=gnu)](https://opensource.org/licenses/AGPL-3.0)

This project is licensed under the AGPL-3.0 License. See the [LICENSE](LICENSE) file for details.

## 🧑‍💻 Author

**Arghya Ghosh**
[🌐 arghya.dev](https://arghya.dev) • [🐙 GitHub](https://github.com/uiuxarghya) • [🔗 LinkedIn](https://linkedin.com/in/uiuxarghya)

_Built with ❤️ for movie lovers and AI enthusiasts._
