import os
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_FOLDER = "data"

file_paths = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith('.txt')]
documents = []
for path in file_paths:
    with open(path, 'r', encoding='utf-8') as f:
        documents.append(f.read())

vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r'(?u)\b\w+\b',
    use_idf=True,
    smooth_idf=False,
    norm=None
)

tfidf_matrix = vectorizer.fit_transform(documents)

# View information about sparse matrices
print(f"Shape of a sparse matrix: {tfidf_matrix.shape}")
print(f"Number of non-zero elements stored: {tfidf_matrix.nnz}")
print(f"sparsity: {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4f}")

# Get the list of feature words
feature_names = vectorizer.get_feature_names_out()
print(f"The top 10 words in the dictionary: {feature_names[:10]}")

# Set the number of keywords want to display.
TOP_K = 10

# Traverse all documents
for doc_index in range(tfidf_matrix.shape[0]):
    row = tfidf_matrix[doc_index]
    coo = row.tocoo()

    # Pair words with weights
    word_weight_pairs = [(feature_names[j], v) for i, j, v in zip(coo.row, coo.col, coo.data)]

    # Sort by weight in descending order
    sorted_pairs = sorted(word_weight_pairs, key=lambda x: x[1], reverse=True)

    # Print document name and Top-K keywords
    print(f"\ndoc '{os.path.basename(file_paths[doc_index])}' s Top-{TOP_K} keywords：")
    for word, weight in sorted_pairs[:TOP_K]:
        print(f"  word '{word}' : {weight:.4f}")