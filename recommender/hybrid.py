# recommender/hybrid.py
from collections import defaultdict

from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender

class HybridRecommender:
    def __init__(self, alpha=0.6):
        """
        alpha: weight for collaborative filtering.
        (1 - alpha): weight for content-based.
        """
        self.alpha = alpha
        self.content_model = ContentBasedRecommender()
        # Use NMF-based collaborative by default
        self.collab_model = CollaborativeRecommender(cf_mode="user", use_nmf=True)

    def fit(self):
        self.content_model.fit()
        self.collab_model.fit()

    def recommend_for_user(self, user_id, liked_item_id=None, top_n=5):
        """
        Hybrid logic:
        - Get recs from collaborative model (NMF)
        - If liked_item_id is provided, get similar items from content-based
        - Normalize scores from both
        - Combine: alpha * collab_score + (1 - alpha) * content_score
        """
        collab_recs = self.collab_model.recommend_for_user(
            user_id=user_id,
            top_n=top_n * 2,  # get a bit more
            use="nmf"
        )

        content_recs = []
        if liked_item_id is not None:
            content_recs = self.content_model.recommend_similar_items(
                liked_item_id,
                top_n=top_n * 2
            )

        # Normalize scores separately
        def normalize(recs):
            if not recs:
                return recs
            scores = [r["score"] for r in recs]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                for r in recs:
                    r["norm_score"] = 1.0
            else:
                for r in recs:
                    r["norm_score"] = (r["score"] - min_s) / (max_s - min_s)
            return recs

        collab_recs = normalize(collab_recs)
        content_recs = normalize(content_recs)

        # Combine via item_id
        combined_scores = defaultdict(lambda: {"collab": 0.0, "content": 0.0, "title": ""})

        for r in collab_recs:
            combined_scores[r["item_id"]]["collab"] = r.get("norm_score", 0.0)
            combined_scores[r["item_id"]]["title"] = r["title"]

        for r in content_recs:
            combined_scores[r["item_id"]]["content"] = r.get("norm_score", 0.0)
            if not combined_scores[r["item_id"]]["title"]:
                combined_scores[r["item_id"]]["title"] = r["title"]

        final_recs = []
        for item_id, vals in combined_scores.items():
            final_score = self.alpha * vals["collab"] + (1 - self.alpha) * vals["content"]
            final_recs.append({
                "item_id": int(item_id),
                "title": vals["title"],
                "score": float(final_score)
            })

        final_recs = sorted(final_recs, key=lambda x: x["score"], reverse=True)[:top_n]
        return final_recs
