# recommender/collaborative.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

from .data_loader import load_ratings, load_items, create_user_item_matrix

class CollaborativeRecommender:
    def __init__(self, cf_mode="user", use_nmf=True, n_components=20):
        """
        cf_mode: 'user' or 'item' for collaborative filtering type.
        use_nmf: whether to also train a matrix factorization model.
        """
        self.cf_mode = cf_mode
        self.use_nmf = use_nmf
        self.n_components = n_components

        self.ratings = None
        self.items = None
        self.user_item_matrix = None
        self.similarity_matrix = None

        # For NMF
        self.nmf_model = None
        self.user_factors = None
        self.item_factors = None

    def fit(self):
        self.ratings = load_ratings()
        self.items = load_items()
        self.user_item_matrix = create_user_item_matrix(self.ratings)

        # Fill NaN with 0 for CF similarity calculations
        matrix_filled = self.user_item_matrix.fillna(0).values

        if self.cf_mode == "user":
            self.similarity_matrix = cosine_similarity(matrix_filled)
        else:  # item-based
            self.similarity_matrix = cosine_similarity(matrix_filled.T)

        # NMF training (matrix factorization)
        if self.use_nmf:
            # NMF requires all values >= 0, so we fill NaNs with 0
            nmf_input = self.user_item_matrix.fillna(0).values
            self.nmf_model = NMF(
                n_components=self.n_components,
                init="random",
                random_state=42,
                max_iter=200
            )
            self.user_factors = self.nmf_model.fit_transform(nmf_input)
            self.item_factors = self.nmf_model.components_

    # ------- Helper functions --------
    def _get_user_index(self, user_id):
        if user_id not in self.user_item_matrix.index:
            return None
        return self.user_item_matrix.index.get_loc(user_id)

    def _get_item_index(self, item_id):
        if item_id not in self.user_item_matrix.columns:
            return None
        return self.user_item_matrix.columns.get_loc(item_id)

    # ------- Collaborative Filtering Prediction (User/Item based) --------
    def predict_rating_cf(self, user_id, item_id):
        """
        Predict rating using user-based or item-based CF.
        """
        if self.user_item_matrix is None or self.similarity_matrix is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        user_idx = self._get_user_index(user_id)
        item_idx = self._get_item_index(item_id)
        if user_idx is None or item_idx is None:
            return None

        matrix = self.user_item_matrix.values

        if self.cf_mode == "user":
            # Ratings of all users for this item
            item_ratings = matrix[:, item_idx]
            sim_scores = self.similarity_matrix[user_idx, :]

            # Exclude the user themself
            sim_scores[user_idx] = 0

        else:  # item-based
            # Ratings of this user for all items
            user_ratings = matrix[user_idx, :]
            sim_scores = self.similarity_matrix[item_idx, :]

            # Exclude the item itself
            sim_scores[item_idx] = 0

        # Weighted average
        if self.cf_mode == "user":
            # users similar to target user
            numerator = np.dot(sim_scores, item_ratings)
            denominator = np.sum(np.abs(sim_scores)) + 1e-8
        else:
            numerator = np.dot(sim_scores, user_ratings)
            denominator = np.sum(np.abs(sim_scores)) + 1e-8

        if denominator == 0:
            return None

        return float(numerator / denominator)

    # ------- NMF Prediction --------
    def predict_rating_nmf(self, user_id, item_id):
        if self.user_factors is None or self.item_factors is None:
            return None

        user_idx = self._get_user_index(user_id)
        item_idx = self._get_item_index(item_id)
        if user_idx is None or item_idx is None:
            return None

        # Dot product of user and item factors
        rating_pred = np.dot(self.user_factors[user_idx], self.item_factors[:, item_idx])
        return float(rating_pred)

    # ------- Top-N Recommendations --------
    def recommend_for_user(self, user_id, top_n=5, use="nmf"):
        """
        use: 'cf' or 'nmf'
        Returns list of dicts: {item_id, title, score}
        """
        if self.user_item_matrix is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if user_id not in self.user_item_matrix.index:
            return []

        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[~user_ratings.isna()].index.tolist()

        all_item_ids = self.user_item_matrix.columns
        scores = []

        for item_id in all_item_ids:
            if item_id in rated_items:
                continue  # do not recommend already-rated items

            if use == "cf":
                score = self.predict_rating_cf(user_id, item_id)
            else:
                score = self.predict_rating_nmf(user_id, item_id)

            if score is not None:
                scores.append((item_id, score))

        # Sort by predicted score
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

        # Map to item titles
        recs = []
        for item_id, score in scores:
            row = self.items[self.items["item_id"] == item_id]
            if not row.empty:
                title = row.iloc[0]["title"]
            else:
                title = f"Item {item_id}"
            recs.append({"item_id": int(item_id), "title": title, "score": float(score)})

        return recs
