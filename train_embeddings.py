# DESCRIPTION:
# This script handles the "model training" phase of the pipeline.
# It reads the cleaned and standardized text data produced by
# 'process_assignment.py' and trains Word2Vec and FastText
# embedding models using the 'gensim' library.
#
# EXECUTION:
# This script MUST be run *after* 'process_assignment.py' has
# successfully created the files in the 'cleaned_data/' directory.

# =================================================================
# STAGE 1: IMPORTS AND LOGGING SETUP
# =================================================================

import pandas as pd
from pathlib import Path
import logging # Used to display training progress from gensim
from gensim.models import Word2Vec, FastText

# --- Logging Configuration ---
# Configure logging to show INFO-level messages.
# This is crucial for monitoring the training process (e.g., epochs, word counts).
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# =================================================================
# STAGE 2: MAIN TRAINING FUNCTION
# =================================================================

def main():
    """Main training script entry-point."""
    
    # --- 1. Define Source and Output Paths ---
    # This script's *input* is the *output* of the previous script.
    INPUT_DIR = Path("cleaned_data")
    
    # List of files to be read as the training corpus
    files = [
        INPUT_DIR / "labeled-sentiment_2col.xlsx",
        INPUT_DIR / "test__1__2col.xlsx",
        INPUT_DIR / "train__3__2col.xlsx",
        INPUT_DIR / "train-00000-of-00001_2col.xlsx",
        INPUT_DIR / "merged_dataset_CSV__1__2col.xlsx",
    ]
    
    # Create the output directory for the trained models
    OUTPUT_DIR = Path("embeddings")
    OUTPUT_DIR.mkdir(exist_ok=True) # exist_ok=True prevents errors if it already exists

    # --- 2. Load Sentences ---
    # gensim models expect the corpus as a list of lists of tokens
    # e.g., [['bu', 'bir', 'cümle'], ['bu', 'da', 'ikinci']]
    print("Reading cleaned Excel files...")
    sentences = []
    for f in files:
        if not f.exists():
            print(f"WARNING: '{f}' file not found, skipping.")
            continue
        
        try:
            # We only need the 'cleaned_text' column
            df = pd.read_excel(f, usecols=["cleaned_text"])
            
            # Convert each row (string) into a list of tokens
            # and add them to the main 'sentences' list.
            # .astype(str) is a robust way to handle any potential empty/NaN cells.
            sentences.extend(df["cleaned_text"].astype(str).str.split().tolist())
        except Exception as e:
            print(f"ERROR: Error while reading '{f}': {e}")
            
    if not sentences:
        print("CRITICAL ERROR: No sentences found for training. Is 'cleaned_data' folder empty?")
        return

    print(f"Found {len(sentences)} total texts (sentences/documents) for training.")

    # --- 3. Train Word2Vec Model ---
    # This uses the Skip-gram architecture (sg=1) as specified.
    print("\nTraining Word2Vec model...")
    
    w2v_model = Word2Vec(
        sentences=sentences, 
        vector_size=300,  # Dimensionality of the word vectors. 300 is a common standard.
        window=5,         # Max distance between current and predicted word.
        min_count=3,      # Ignores all words with total frequency lower than this.
        sg=1,             # 1 for Skip-gram (predicts context from word), 0 for CBOW.
        negative=10,      # Number of "noise words" to draw for negative sampling.
        epochs=10,        # Number of iterations over the corpus.
        workers=4         # Use 4 CPU cores to parallelize training.
    )
    
    w2v_path = OUTPUT_DIR / "word2vec.model"
    w2v_model.save(str(w2v_path))
    print(f"Word2Vec model saved to: {w2v_path}")

    # --- 4. Train FastText Model ---
    # FastText is similar to Word2Vec but also learns from character n-grams.
    # This allows it to handle Out-of-Vocabulary (OOV) words and typos.
    print("\nTraining FastText model...")
    
    ft_model = FastText(
        sentences=sentences,
        vector_size=300,  # Must be same as Word2Vec for fair comparison
        window=5,
        min_count=3,
        sg=1,             # Use Skip-gram
        min_n=3,          # Minimum length of character n-gram
        max_n=6,          # Maximum length of character n-gram
        epochs=10,        
        workers=4
    )
    
    ft_path = OUTPUT_DIR / "fasttext.model"
    ft_model.save(str(ft_path))
    print(f"FastText model saved to: {ft_path}")
    
    print("\n--- Model Training Complete ---")

# =================================================================
# STAGE 3: SCRIPT EXECUTION
# =================================================================

if __name__ == "__main__":
    main()