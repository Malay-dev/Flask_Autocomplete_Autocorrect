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
    csv_filename = "data.csv"
    df = pd.read_csv(csv_filename)
    url_dataset =  list(zip(df["metadata_global_index"], df["domain"], df["url"]))
    return url_dataset

def preprocess_input(url):
    url = re.sub(r"[^\w\s./]", "", url)
    url = url.strip()
    return url

user_history = []
def set_history(history):
    global user_history
    user_history = history
    print(user_history)
    return user_history

def correct_and_autocomplete_url(input_url, dataset=set_data()):
    input_url = preprocess_input(input_url)

    # Set thresholds for similarity scores
    threshold_levenshtein = 50 - len(input_url)
    threshold_cosine = 0.1 + (len(input_url) * 0.01)
    threshold_autocomplete = 0.2
   
    best_match_levenshtein = max(dataset, key=lambda url: fuzz.ratio(input_url, url[1]))
    similarity_levenshtein = fuzz.ratio(input_url, best_match_levenshtein[1])

    if similarity_levenshtein >= threshold_levenshtein:
        corrected_url_levenshtein = best_match_levenshtein
    else:
        corrected_url_levenshtein = input_url

    autocomplete_trie = Trie()
    for data in dataset:
        url = data[1]
        autocomplete_trie.insert(url, data)

    autocomplete_results = autocomplete_trie.search_autocomplete(corrected_url_levenshtein[1])
    
    corrected_url_combined = corrected_url_levenshtein  
    autocomplete_results = autocomplete_results + [corrected_url_combined]
    autocomplete_results = [list(result) for result in autocomplete_results if result]
    for i in autocomplete_results:
        if i[1] in user_history:
            i[0] = 0
    
    sorted_data = sorted(autocomplete_results, key=lambda x: x[0])       
    return corrected_url_combined, sorted_data