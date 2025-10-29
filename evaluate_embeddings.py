# DESCRIPTION:
# This script performs the "evaluation" phase of the pipeline.
# It loads the Word2Vec and FastText models trained by 'train_embeddings.py'
# and runs a series of quantitative and qualitative tests to compare them.
# The output of this script is the primary evidence for the README.md report.
#
# EXECUTION:
# This script MUST be run *after* 'train_embeddings.py' has
# successfully created the models in the 'embeddings/' directory.

# =================================================================
# STAGE 1: IMPORTS AND LOGGING SETUP
# =================================================================

import pandas as pd
from pathlib import Path
import logging
import re
from gensim.models import Word2Vec, FastText

import numpy as np
from numpy import dot, float32 as REAL
from numpy.linalg import norm

# --- Logging Configuration ---
# Set logging to ERROR only. We don't need gensim's INFO logs
# while just loading and querying the models.
logging.basicConfig(level=logging.ERROR)

# =================================================================
# STAGE 2: HELPER FUNCTIONS FOR EVALUATION
# =================================================================

def lexical_coverage(model, tokens):
    """
    Calculates the percentage of tokens that exist in the model's vocabulary.
    """
    vocab = model.wv.key_to_index
    # NOTE: This metric is misleading for FastText, as FastText can
    # generate vectors for OOV words. However, it's useful to confirm
    # that both models were built from the same base vocabulary (due to min_count).
    return sum(1 for t in tokens if t in vocab) / max(1, len(tokens))

def read_tokens(f):
    """Reads a _2col.xlsx file and returns a flat list of all tokens."""
    df = pd.read_excel(f, usecols=["cleaned_text"])
    return [t for row in df["cleaned_text"].astype(str) for t in row.split()]

def cos_sim(a, b):
    """Calculates the cosine similarity between two numpy vectors."""
    return float(dot(a, b) / (norm(a) * norm(b)))

def pair_sim(model, pairs):
    """
    Calculates the average similarity for a list of (a, b) word pairs.
    
    This is more robust than gensim's .similarity() because it
    fetches vectors manually and handles KeyErrors (OOV words) gracefully,
    which is common for Word2Vec.
    """
    vals = []
    for a, b in pairs:
        try:
            # Get the vectors for each word
            vec_a = model.wv[a]
            vec_b = model.wv[b]
            vals.append(cos_sim(vec_a, vec_b))
        except KeyError:
            # If a word is OOV (e.g., in Word2Vec), skip this pair.
            # FastText will rarely fail here if the word is plausible.
            pass
    return sum(vals) / max(1, len(vals)) if vals else float('nan')

def neighbors(model, word, k=5):
    """Returns the top k nearest neighbors for a given word."""
    try:
        # .most_similar returns a list of (word, score) tuples
        return [w for w, score in model.wv.most_similar(word, topn=k)]
    except KeyError:
        # Handle OOV words for Word2Vec
        return [] # Return an empty list if the word is not in vocab

# =================================================================
# STAGE 3: MAIN EVALUATION SCRIPT
# =================================================================

def main():
    """Loads and compares the trained embedding models."""
    print("--- Starting Step 6: Model Comparison ---")
    
    # --- 1. Load Models ---
    W2V_PATH = Path("embeddings/word2vec.model")
    FT_PATH = Path("embeddings/fasttext.model")
    
    if not W2V_PATH.exists() or not FT_PATH.exists():
        print(f"ERROR: Model files not found. '{W2V_PATH}' or '{FT_PATH}' is missing.")
        print("Please run 'train_embeddings.py' first.")
        return
        
    print("Loading models (this may take a moment)...")
    w2v = Word2Vec.load(str(W2V_PATH))
    ft = FastText.load(str(FT_PATH))
    print("Models loaded successfully.")

    # --- 2. Quantitative Test: Lexical Coverage ---
    print("\n== 1. Lexical Coverage (In-Vocabulary Rate) ==")
    INPUT_DIR = Path("cleaned_data")
    files = [
        INPUT_DIR / "labeled-sentiment_2col.xlsx",
        INPUT_DIR / "test__1__2col.xlsx",
        INPUT_DIR / "train__3__2col.xlsx",
        INPUT_DIR / "train-00000-of-00001_2col.xlsx",
        INPUT_DIR / "merged_dataset_CSV__1__2col.xlsx",
    ]
    
    for f in files:
        if not f.exists():
            print(f"WARNING: '{f}' data file not found, skipping coverage test.")
            continue
        toks = read_tokens(f)
        cov_w2v = lexical_coverage(w2v, toks)
        cov_ft = lexical_coverage(ft, toks)
        print(f"  {f.name}:")
        print(f"    - Word2Vec: {cov_w2v:.3f} (coverage of {len(toks)} tokens)")
        print(f"    - FastText: {cov_ft:.3f} (Note: FT can also embed OOV tokens)")

    # --- 3. Quantitative Test: Semantic Similarity ---
    # These lists are probes to test the model's understanding.
    # It's okay if some words were filtered out by min_count=3.
    seed_words = [
        "yaxşı", "pis", "çox", "bahalı", "ucuz", "mükəmməl", "dəhşət",
        "<PRICE>", "<RATING_POS>", "gözəl", "yox"
    ]
    syn_pairs = [
        ("yaxşı", "əla"), ("bahalı", "qiymətli"), ("ucuz", "sərfəli"),
        ("pis", "bərbad"), ("gözəl", "qəşəng")
    ]
    ant_pairs = [
        ("yaxşı", "pis"), ("bahalı", "ucuz"), ("gözəl", "çirkin"),
        ("sevirəm", "nifrət") # 'sevirəm' or 'nifrət' might be OOV
    ]

    print("\n== 2. Semantic Similarity Tests ==")
    syn_w2v = pair_sim(w2v, syn_pairs)
    syn_ft = pair_sim(ft, syn_pairs)
    ant_w2v = pair_sim(w2v, ant_pairs)
    ant_ft = pair_sim(ft, ant_pairs)
    
    print(f"  Synonym Similarity (Higher is better):")
    print(f"    - Word2Vec: {syn_w2v:.3f}")
    print(f"    - FastText: {syn_ft:.3f}")
    
    print(f"\n  Antonym Similarity (Lower is better):")
    print(f"    - Word2Vec: {ant_w2v:.3f}")
    print(f"    - FastText: {ant_ft:.3f}")
    
    # Separation score is a key metric:
    # A good model should have a large gap between synonym and antonym scores.
    try:
        sep_w2v = syn_w2v - ant_w2v
        sep_ft = syn_ft - ant_ft
        print(f"\n  Separation Score (Syn - Ant) (Higher is better):")
        print(f"    - Word2Vec: {sep_w2v:.3f}")
        print(f"    - FastText: {sep_ft:.3f}")
    except Exception:
        pass # Skip this calculation if any score was 'nan'

    # --- 4. Qualitative Test: Nearest Neighbors ---
    # This is often the most revealing test. The quantitative scores
    # can be similar, but the "feel" of the neighbors shows
    # if the model truly understood the *morphology* vs. *context*.
    print("\n== 3. Nearest Neighbors (Qualitative Analysis) ==")
    for word in seed_words:
        print(f"\n  Nearest neighbors for '{word}':")
        nn_w2v = neighbors(w2v, word, k=5)
        nn_ft = neighbors(ft, word, k=5)
        print(f"    - Word2Vec: {nn_w2v}")
        print(f"    - FastText: {nn_ft}")

    print("\n--- Model Comparison Complete ---")


if __name__ == "__main__":
    main()