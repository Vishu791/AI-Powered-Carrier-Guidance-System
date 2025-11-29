from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CACHE_DIR = os.path.join(MODEL_DIR, ".hf_cache")


def load_sentence_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the transformer with clearer error reporting for offline runs."""
    try:
        return SentenceTransformer(model_name, cache_folder=CACHE_DIR)
    except Exception as err:
        print(
            "⚠️ Unable to download the SentenceTransformer model.\n"
            "Please ensure you have an active internet connection or preload the model into "
            f"{CACHE_DIR}.\nDetailed error:"
        )
        print(err)
        sys.exit(1)


def load_career_details():
    details_path = os.path.join(MODEL_DIR, "career_details.json")
    if not os.path.exists(details_path):
        raise FileNotFoundError(
            f"career_details.json not found at {details_path}. Train the model first."
        )
    with open(details_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    model = load_sentence_model()
    careers = load_career_details()

    names = list(careers.keys())
    texts = []

    for name, data in careers.items():
        description = data.get("description", "")
        skills = ", ".join(data.get("skills", []))
        text = f"{name}. {description} Required skills: {skills}."
        texts.append(text)

    print("👉 Generating embeddings...")
    embeds = model.encode(texts, show_progress_bar=True)

    np.save(os.path.join(MODEL_DIR, "career_vectors.npy"), embeds)
    with open(os.path.join(MODEL_DIR, "career_names.json"), "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)

    print("✅ Career embeddings saved.")


if __name__ == "__main__":
    main()
