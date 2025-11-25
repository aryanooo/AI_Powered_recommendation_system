# recommender/content_based.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import load_items

class ContentBasedRecommender:
    def __init__(self):
        self.items = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.vectorizer = None
        self._fitted = False

    def fit(self):
        self.items = load_items()
        self.items["combined_text"] = (
            self.items["title"].fillna("") + " " +
            self.items["description"].fillna("")
        )

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.items["combined_text"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        self._fitted = True

    def recommend_similar_items(self, item_id, top_n=5):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        idx_list = self.items.index[self.items["item_id"] == item_id].tolist()
        if not idx_list:
            return []

        idx = idx_list[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

        recs = []
        for i, score in sim_scores:
            recs.append({
                "item_id": int(self.items.iloc[i]["item_id"]),
                "title": self.items.iloc[i]["title"],
                "score": float(score)
            })
        return recs
