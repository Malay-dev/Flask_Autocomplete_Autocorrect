from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.data = None  # Store complete data in each node

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, data):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.data = data  # Store complete data in the end node of each word

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
            results.append(node.data)  # Include complete data when a word ends

        for char, child_node in node.children.items():
            results.extend(self._get_words_with_prefix(
                child_node, current_prefix + char))
        return results

def set_data():
    csv_filename = "output_data_new.csv"
    df = pd.read_csv(csv_filename)
    url_dataset =  list(zip(df["metadata_global_index"], df["domain"], df["url"]))
    return url_dataset

def preprocess_input(url):
    url = re.sub(r"[^\w\s./]", "", url)
    url = url.strip()
    return url

def correct_and_autocomplete_url(input_url, dataset=set_data()):
    input_url = preprocess_input(input_url)

    # Set thresholds for similarity scores
    threshold_levenshtein =50
    threshold_cosine = 0.1
    threshold_autocomplete = 0.2
    
   
    best_match_levenshtein = max(dataset, key=lambda url: fuzz.ratio(input_url, url[1]))
    similarity_levenshtein = fuzz.ratio(input_url, best_match_levenshtein[1])

    ngram_range = (2, 5)  # Set n-gram range (bi-grams and tri-grams)
    vectorizer = CountVectorizer(ngram_range=ngram_range).fit_transform([input_url] + [data[1] for data in dataset])
    cosine_similarities = cosine_similarity(vectorizer, vectorizer)[0][1:]
    best_match_cosine = dataset[1][cosine_similarities.argmax()]
    similarity_cosine = cosine_similarities.max()


    if similarity_levenshtein >= threshold_levenshtein:
        corrected_url_levenshtein = best_match_levenshtein
    else:
        corrected_url_levenshtein = input_url

    if similarity_cosine >= threshold_cosine:
        corrected_url_cosine = best_match_cosine
    else:
        corrected_url_cosine = input_url

    autocomplete_trie = Trie()
    for data in dataset:
        url = data[1]
        autocomplete_trie.insert(url, data)

    autocomplete_results = autocomplete_trie.search_autocomplete(input_url)


    corrected_url_combined = corrected_url_levenshtein if similarity_levenshtein >= similarity_cosine else corrected_url_cosine
    autocomplete_results = autocomplete_results + [corrected_url_combined]
    autocomplete_results = [result for result in autocomplete_results if result]
    sorted_data = sorted(autocomplete_results, key=lambda x: x[0])
    sorted_data = autocomplete_results
    return corrected_url_combined, sorted_data

