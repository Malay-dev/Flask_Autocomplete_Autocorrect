from thefuzz import fuzz
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
            node = node.children.setdefault(char, TrieNode())
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
            results.extend(self._get_words_with_prefix(child_node, current_prefix + char))
        return results

def load_dataset(csv_filename="data.csv"):
    df = pd.read_csv(csv_filename)
    return [
        {
            "metadata_global_index": row["metadata_global_index"],
            "domain": row["domain"],
            "url": row["url"],
            "index": int(row["index"])
        }
        for _, row in df.iterrows() if row["metadata_global_index"] != 0
    ]

def preprocess_input(url):
    return re.sub(r"[^\w\s./]", "", url).strip()

user_history = []

def set_history(history):
    global user_history
    user_history = history
    print(user_history[:100])
    return user_history

def calculate_score(fuzz_score, fit_rank, weight_fuzz=0.02, weight_index=0.9):
    """
    Calculate the score based on fuzzy matching and ranking.
    
    Equation:
    score = (weight_fuzz * fuzz_score) + (weight_index * (1 / fit_rank))
    
    Variables:
    - fuzz_score: Similarity score from fuzzy matching (0-100)
    - fit_rank: Ranking of the URL in the dataset
    - weight_fuzz: Weight given to the fuzzy score (default: 0.02 or 0.7)
    - weight_index: Weight given to the inverse of the rank (default: 0.9 or 0.3)
    
    Note: Weights are adjusted if fuzz_score > 80 to prioritize close matches.
    """
    if fuzz_score > 80:
        weight_fuzz, weight_index = 0.7, 0.3
    return (weight_fuzz * fuzz_score) + (weight_index * (1 / fit_rank))

def correct_and_autocomplete_url(input_url, dataset=None):
    if dataset is None:
        dataset = load_dataset()
    
    input_url = preprocess_input(input_url)
    if not input_url:
        return "", []
    """
    Calculate Levenshtein distance threshold based on input length
    Equation: threshold = C + (L * K)
    Variables:
    - C: Constant base threshold (10 in this case)
    - L: Length of the input URL
    - K: Multiplier for input length (2 in this case)
    """
    threshold_levenshtein = 10 + len(input_url) * 2

    similarity_scores = [
        {"data": url_data, "FuzzScore": fuzz.ratio(input_url, url_data["domain"])}
        for url_data in dataset
    ]
    
    filtered_results = []
    seen_domains = set()
    for result in similarity_scores:
        if result["FuzzScore"] >= threshold_levenshtein and result["data"]["metadata_global_index"] != 0:
            domain = result["data"]["domain"]
            if domain not in seen_domains:
                fit_rank = result["data"]["metadata_global_index"]
                score = calculate_score(result["FuzzScore"], fit_rank)
                filtered_results.append({
                    "data": result["data"],
                    "FuzzScore": result["FuzzScore"],
                    "FitRank": fit_rank,
                    "score": score
                })
                seen_domains.add(domain)

    if not filtered_results:
        return "", []

    corrected_url_levenshtein = max(filtered_results, key=lambda x: x["score"])

    autocomplete_trie = Trie()
    for data in filtered_results:
        autocomplete_trie.insert(data["data"]["domain"], data)

    autocomplete_results = autocomplete_trie.search_autocomplete(input_url)
    all_results = autocomplete_results + [corrected_url_levenshtein] + filtered_results

    """
    Adjust scores based on user history
    Equation: adjusted_score = original_score * history_multiplier
    Variables:
    - original_score: The score calculated earlier
    - history_multiplier: (total_history_length - index_in_history)
      This gives higher weight to more recent history items
    """
    for result in all_results:
        if result["data"]["domain"] in user_history:
            history_multiplier = len(user_history) - user_history.index(result["data"]["domain"])
            result["score"] *= history_multiplier

    sorted_data = sorted(all_results, key=lambda x: x["score"], reverse=True)
    
    return corrected_url_levenshtein, sorted_data