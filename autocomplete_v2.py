from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search_autocomplete(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return self._get_words_with_prefix(node, prefix)

    def _get_words_with_prefix(self, node, current_prefix):
        results = []
        if node.is_end_of_word:
            results.append(current_prefix)

        for char, child_node in node.children.items():
            results.extend(self._get_words_with_prefix(
                child_node, current_prefix + char))
        return results

def set_data():
    csv_filename = "output_data_new.csv"
    df = pd.read_csv(csv_filename)
    url_dataset = df["domain"]
    return url_dataset

def preprocess_url(url):
    # Modify the preprocessing to retain dots and slashes
    url = re.sub(r"[^\w\s./]", "", url)
    url = url.strip()
    return url

def correct_and_autocomplete_url(input_url, dataset=set_data()):
    # Preprocess input URL
    input_url = preprocess_url(input_url)
    # Set thresholds for similarity scores
    threshold_levenshtein = 80
    threshold_cosine = 0.3
    threshold_autocomplete = 0.5
    # Levenshtein distance for autocorrection
    best_match_levenshtein = max(dataset, key=lambda url: fuzz.ratio(input_url, url))
    similarity_levenshtein = fuzz.ratio(input_url, best_match_levenshtein)

    # Cosine similarity for autocorrection
    ngram_range = (2, 3)  # Set n-gram range (bi-grams and tri-grams)
    vectorizer = CountVectorizer(ngram_range=ngram_range).fit_transform([input_url] + dataset)
    cosine_similarities = cosine_similarity(vectorizer, vectorizer)[0][1:]
    best_match_cosine = dataset[cosine_similarities.argmax()]
    similarity_cosine = cosine_similarities.max()


    # If the Levenshtein similarity score is above the autocorrect threshold, consider it a match
    if similarity_levenshtein >= threshold_levenshtein:
        corrected_url_levenshtein = best_match_levenshtein
    else:
        corrected_url_levenshtein = input_url

    # If the Cosine similarity score is above the autocorrect threshold, consider it a match
    if similarity_cosine >= threshold_cosine:
        corrected_url_cosine = best_match_cosine
    else:
        corrected_url_cosine = input_url

    # Trie for autocomplete
    autocomplete_trie = Trie()
    for url in dataset:
        autocomplete_trie.insert(url)

    # Search for autocomplete results using trie
    autocomplete_results = autocomplete_trie.search_autocomplete(input_url)


    # Combine autocorrect results based on both Levenshtein and Cosine
    corrected_url_combined = corrected_url_levenshtein if similarity_levenshtein >= similarity_cosine else corrected_url_cosine

    return corrected_url_combined, autocomplete_results

# Load the dataset

# # Correct and autocomplete URL
# corrected_url, autocomplete_results = correct_and_autocomplete_url("www.goog", url_dataset)
# print("Corrected URL:", corrected_url)
# print("Autocomplete Results:", autocomplete_results)
