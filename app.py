# AI helped generate this code

import pandas as pd
from pathlib import Path

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer


# Keep the model loaded when Streamlit reruns the page
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


model = load_model()

# Open the saved database next to app.py
project_folder = Path(__file__).resolve().parent
client = chromadb.PersistentClient(path=str(project_folder / "chromadb"))
collection = client.get_or_create_collection("episode_transcript")


def build_database():
    csv_path = Path.home() / "Downloads" / "transcript.csv"
    transcript = pd.read_csv(csv_path)

    # Remove the TurboScribe notice in the first row.
    transcript = transcript.iloc[1:].copy()
    transcript = transcript.sort_values("start")

    # Give each line a 30-second window number.
    transcript["window"] = transcript["start"] // 30_000

    ids = []
    texts = []
    timestamps = []

    for window in transcript["window"].unique():
        # Select the lines in this window.
        lines = transcript[transcript["window"] == window]

        # Combine those lines into one searchable passage.
        passage = " ".join(lines["text"].str.strip())
        texts.append(passage)
        ids.append(str(window))
        timestamps.append({
            "start": int(lines["start"].min()),
            "end": int(lines["end"].max()),
        })

    embeddings = model.encode(texts)
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=timestamps,
    )

    print(f"Saved {collection.count()} passages to chromadb/")

def format_timestamp(milliseconds):
    total_seconds = milliseconds // 1_000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


def semantic_search(query):
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(3, collection.count()),
        include=["documents", "metadatas"],
    )

    matches = []
    for text, times in zip(results["documents"][0], results["metadatas"][0]):
        start = format_timestamp(times["start"])
        end = format_timestamp(times["end"])
        matches.append((start, end, text))

    return matches


# Build the database only if there are no saved passages
if collection.count() == 0:
    build_database()

# Streamlit page
st.title("Austin & Ally Scene Finder")
st.write("Describe a moment from the episode to find matching dialogue.")

query = st.text_input("Describe a scene")

if st.button("Search"):
    if query.strip():
        st.subheader("Top matches")
        for start, end, text in semantic_search(query.strip()):
            st.write(start + " to " + end)
            st.write(text)
    else:
        st.write("Enter a description of the scene you want to find.")