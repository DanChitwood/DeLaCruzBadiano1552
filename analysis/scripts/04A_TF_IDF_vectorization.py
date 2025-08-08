import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import pickle
import nltk
from nltk.corpus import stopwords
import unicodedata # New import for accent stripping utility

# --- Download NLTK Spanish stop words if not already present ---
try:
    stopwords.words('spanish')
except LookupError:
    print("NLTK Spanish stopwords not found. Downloading...")
    nltk.download('stopwords')
    print("NLTK Spanish stopwords downloaded.")

# --- Helper function to strip accents (same logic as TfidfVectorizer's 'unicode' option) ---
def strip_accents_from_string(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

INPUTS_MASTER_SHEET_PROCESSING_DIR = os.path.join(PROJECT_ROOT, "outputs", "master_sheet_processing")
AGGREGATED_TEXTS_CSV = os.path.join(INPUTS_MASTER_SHEET_PROCESSING_DIR, "aggregated_plant_texts.csv")

OUTPUTS_TFIDF_VECTORIZATION_DIR = os.path.join(PROJECT_ROOT, "outputs", "TF_IDF_vectorization")
OUTPUTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

os.makedirs(OUTPUTS_TFIDF_VECTORIZATION_DIR, exist_ok=True)
os.makedirs(OUTPUTS_FIGURES_DIR, exist_ok=True)

print(f"--- Starting 04_TF_IDF_vectorization.py ---")
print(f"Loading aggregated texts from: {AGGREGATED_TEXTS_CSV}")

# --- Load Data ---
try:
    df = pd.read_csv(AGGREGATED_TEXTS_CSV)
    print(f"Loaded {len(df)} records from {AGGREGATED_TEXTS_CSV}.")
except FileNotFoundError:
    print(f"Error: The file '{AGGREGATED_TEXTS_CSV}' was not found.")
    print("Please ensure 'aggregated_plant_texts.csv' is in the correct location.")
    print(f"Expected path: {AGGREGATED_TEXTS_CSV}")
    exit()

# --- Prepare Text Data ---
df['aggregated_english_text'] = df['aggregated_english_text'].fillna('')
df['aggregated_spanish_text'] = df['aggregated_spanish_text'].fillna('')

english_texts = df['aggregated_english_text'].tolist()
spanish_texts = df['aggregated_spanish_text'].tolist()
nahuatl_ids = df['ID'].tolist()

print(f"Prepared {len(english_texts)} English texts and {len(spanish_texts)} Spanish texts for vectorization.")

# --- TF-IDF Vectorization for English ---
print("\nPerforming TF-IDF vectorization for English text...")
tfidf_vectorizer_en = TfidfVectorizer(min_df=3, max_df=0.85, stop_words='english', strip_accents='unicode')
tfidf_matrix_en = tfidf_vectorizer_en.fit_transform(english_texts)
english_vocab = tfidf_vectorizer_en.get_feature_names_out()

print(f"English TF-IDF matrix shape: {tfidf_matrix_en.shape} (Documents x Features)")
print(f"English Vocabulary size: {len(english_vocab)}")

# --- TF-IDF Vectorization for Spanish ---
print("\nPerforming TF-IDF vectorization for Spanish text...")
# Get Spanish stop words from NLTK
spanish_stop_words_raw = stopwords.words('spanish')
# Strip accents from the Spanish stop words list itself
spanish_stop_words_processed = [strip_accents_from_string(word) for word in spanish_stop_words_raw]

tfidf_vectorizer_es = TfidfVectorizer(min_df=3, max_df=0.85, stop_words=spanish_stop_words_processed, strip_accents='unicode')
tfidf_matrix_es = tfidf_vectorizer_es.fit_transform(spanish_texts)
spanish_vocab = tfidf_vectorizer_es.get_feature_names_out()

print(f"Spanish TF-IDF matrix shape: {tfidf_matrix_es.shape} (Documents x Features)")
print(f"Spanish Vocabulary size: {len(spanish_vocab)}")

# --- Save Vectorized Data and Vocabularies ---
print("\nSaving vectorized data, vocabularies, and Nahuatl IDs...")

save_npz(os.path.join(OUTPUTS_TFIDF_VECTORIZATION_DIR, 'english_TF_IDF_vectorization_matrix.npz'), tfidf_matrix_en)
with open(os.path.join(OUTPUTS_TFIDF_VECTORIZATION_DIR, 'english_TF_IDF_vocab.pkl'), 'wb') as f:
    pickle.dump(english_vocab, f)

save_npz(os.path.join(OUTPUTS_TFIDF_VECTORIZATION_DIR, 'spanish_TF_IDF_vectorization_matrix.npz'), tfidf_matrix_es)
with open(os.path.join(OUTPUTS_TFIDF_VECTORIZATION_DIR, 'spanish_TF_IDF_vocab.pkl'), 'wb') as f:
    pickle.dump(spanish_vocab, f)

with open(os.path.join(OUTPUTS_TFIDF_VECTORIZATION_DIR, 'nahuatl_ids.pkl'), 'wb') as f:
    pickle.dump(nahuatl_ids, f)

print(f"All processed data saved to '{OUTPUTS_TFIDF_VECTORIZATION_DIR}'.")
print(f"--- 04_TF_IDF_vectorization.py Finished ---")