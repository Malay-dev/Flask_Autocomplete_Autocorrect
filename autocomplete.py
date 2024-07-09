from thefuzz import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.data = None   
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
        node.data = data   

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
            results.append(node.data)   

        for char, child_node in node.children.items():
            results.extend(self._get_words_with_prefix(
                child_node, current_prefix + char))
        return results

def set_data():
    csv_filename = "data.csv"
    df = pd.read_csv(csv_filename)
    url_dataset = [
        {"metadata_global_index": row["metadata_global_index"], "domain": row["domain"], "url": row["url"]}
        for _, row in df.iterrows() if row["metadata_global_index"] != 0
    ]
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
    if input_url == "":
        return "", []
    
    threshold_levenshtein = 10 + len(input_url) * 10
    
    similarity_scores = [
        {"data": url_data, "score": fuzz.ratio(input_url, url_data["domain"])}
        for url_data in dataset
    ]
    filtered_results = [
        {"data": result["data"], "score": result["score"] / result["data"]["metadata_global_index"]}
        for result in similarity_scores if result["score"] >= threshold_levenshtein and result["data"]["metadata_global_index"] != 0
    ]
    
    if not filtered_results:
        return "", []
    
    corrected_url_levenshtein = max(filtered_results, key=lambda x: x["score"])
    
    autocomplete_trie = Trie()
    for data in dataset:
        url = data["domain"]
        autocomplete_trie.insert(url, data)
    
    autocomplete_results = autocomplete_trie.search_autocomplete(input_url)
    
    corrected_url_combined = corrected_url_levenshtein["data"]
    autocomplete_results = autocomplete_results + [corrected_url_combined]
    
    for result in autocomplete_results:
        if result["domain"] in user_history:
            result["metadata_global_index"] = 0
    
    sorted_data = sorted(autocomplete_results, key=lambda x: x["metadata_global_index"])
    
    return corrected_url_combined, sorted_data