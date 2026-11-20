# AI helped generate this code

# Austin & Ally Scene Finder

AIPI 510 · Week 13 Semantic Search Activity

Describe a scene from the *Austin & Ally* episode “Duets and Destiny” in your own words. The Streamlit app returns up to three matching dialogue passages with start and end timestamps.

## Run locally

Use Python 3.12. From the repository root, run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`. The first launch needs an internet connection to download the embedding model.

The included `chromadb/` folder contains the saved database. Keep the whole folder, including `chroma.sqlite3` and the index files. You can run the app and search the saved passages without the source CSV.

## Dataset and chunking

The source is a TurboScribe CSV export of the Disney Channel episode. It contains 516 rows with three columns: `start`, `end`, and `text`. The start and end times are measured in milliseconds.

1. Load the CSV with pandas, remove the first row containing the TurboScribe notice, and sort by `start`.
2. Assign each line to a 30-second window using `start // 30_000`. Join the dialogue within each window into one passage. Save the earliest start and latest end as the passage’s timestamps.
3. Embed the resulting 49 passages with `sentence-transformers/all-MiniLM-L6-v2`. Save the passage text, embeddings, and timestamps in ChromaDB’s `episode_transcript` collection.
4. At search time, embed the query with the same model. ChromaDB retrieves the closest passages, and the app displays their timestamps in minutes and seconds.

## Rebuild the database

1. Save the original CSV export as `~/Downloads/transcript.csv`.
2. Stop the app with `Ctrl+C`. In a local checkout, rename `chromadb/` to `chromadb-backup/` to preserve the saved database.
3. Run `python -m streamlit run app.py` and open the local URL. When the app finds an empty collection, it rebuilds the database. With the original export, the terminal should print `Saved 49 passages to chromadb/`.

The original CSV is kept separately. Rebuilding the same passages requires that export because a fresh transcription may split or timestamp the dialogue differently.

## Verify the app

Search for **“two people are excited to see each other again.”** Check whether the opening reunion appears among the results and whether the timestamps lead to the matching dialogue. Then submit an empty search to check that the app asks for a scene description.

## Deploy on Streamlit Community Cloud

Use the `510-week13` GitHub repository, branch `main`, and entry point `app.py`. Select Python 3.12 and make sure the full `chromadb/` folder is committed. After deployment, repeat the test query on the live app.