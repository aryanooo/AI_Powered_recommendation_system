# recommender/data_loader.py
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def load_items():
    items_path = os.path.join(DATA_DIR, "items.csv")
    items = pd.read_csv(items_path)

    # Basic cleaning
    if "item_id" not in items.columns:
        raise ValueError("items.csv must contain 'item_id' column")

    items["title"] = items["title"].fillna("Unknown Title")
    items["description"] = items.get("description", "").fillna("")

    return items

def load_ratings():
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    ratings = pd.read_csv(ratings_path)

    # Expect columns: user_id, item_id, rating
    required_cols = {"user_id", "item_id", "rating"}
    if not required_cols.issubset(ratings.columns):
        raise ValueError(f"ratings.csv must contain columns: {required_cols}")

    # Clean: drop missing, filter weird ratings (e.g., <0 or >5)
    ratings = ratings.dropna(subset=["user_id", "item_id", "rating"])
    ratings = ratings[(ratings["rating"] >= 0.5) & (ratings["rating"] <= 5)]

    # Optional: normalize ratings later in models
    return ratings

def create_user_item_matrix(ratings, min_user_ratings=1, min_item_ratings=1):
    """Create user-item matrix (rows: user_id, cols: item_id)."""
    # Filter sparse users/items if you want
    user_counts = ratings["user_id"].value_counts()
    item_counts = ratings["item_id"].value_counts()

    ratings = ratings[
        ratings["user_id"].isin(user_counts[user_counts >= min_user_ratings].index) &
        ratings["item_id"].isin(item_counts[item_counts >= min_item_ratings].index)
    ]

    user_item_matrix = ratings.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating"
    )

    return user_item_matrix
