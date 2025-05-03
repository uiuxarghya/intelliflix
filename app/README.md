---
title: Intelliflix
emoji: 🎬
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.45.0
app_file: app.py
tags:
  - streamlit
pinned: true
license: agpl-3.0
short_description: A semantic movie recommendation system.
---

# 🎬 Intelliflix – Semantic Movie Recommender

Intelliflix is a semantic movie recommendation system powered by TMDb metadata and sentence-transformer embeddings.

💡 Simply enter a movie name or plot description, and Intelliflix will return semantically similar movies — not just based on genre, but meaning.

## 🔍 How It Works

- Uses TMDb metadata (title, overview, genres)
- Generates sentence embeddings using `all-MiniLM-L6-v2`
- Runs semantic search via FAISS over precomputed vectors
- Recommends movies with similar plot meanings

## 📁 Dataset

This Space uses the [Intelliflix Store Dataset](https://huggingface.co/datasets/uiuxarghya/intelliflix-store), which includes:

- Movie metadata (`data/`)
- Sentence embeddings (`embeddings/`)
- FAISS vector indexes (`indexes/`)

## 🚀 Run Locally

```bash
  git clone https://huggingface.co/spaces/uiuxarghya/intelliflix
  cd intelliflix
  pip install -r requirements.txt
  streamlit run app.py
```

## 🛡 License

This project is licensed under the **AGPL-3.0**.

## 📫 Author

Built by [Arghya Ghosh](https://arghya.dev) · [GitHub](https://github.com/uiuxarghya)
