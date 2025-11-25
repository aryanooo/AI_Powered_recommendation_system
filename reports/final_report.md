# Personalized Recommendation System – Final Report

## 1. Introduction

- **Objective**: Build a machine learning-based recommendation engine that provides personalized suggestions to users.
- **Use-case**: (e.g., Movie recommendations / product recommendations)
- **Dataset**: (e.g., MovieLens 100K / custom dataset)
- **Tech Stack**: Python, pandas, scikit-learn, Flask

---

## 2. Dataset & Preprocessing

### 2.1 Dataset Description

- Number of users: …
- Number of items: …
- Number of ratings: …
- Rating scale: (e.g., 1–5)

### 2.2 Data Cleaning

- Removed rows with missing user_id/item_id/rating.
- Filtered ratings outside valid range (e.g., 0.5–5.0).
- Handled missing item metadata (title/description).

### 2.3 Exploratory Data Analysis (Week 1)

- Rating distribution: (describe histogram)
- Ratings per user: (describe long-tail behavior)
- Ratings per item: (most popular items)
- Matrix sparsity: ~… (fraction of missing entries)

(Include 1–2 plots/screenshots from `eda.ipynb`.)

---

## 3. Methods

### 3.1 Collaborative Filtering

- **User-based CF**:
  - Built user-item matrix with ratings.
  - Used cosine similarity between user vectors.
  - Predicted rating using weighted average of similar users’ ratings.

- **Item-based CF** (conceptual, optional if implemented):
  - Similarity between items based on user rating patterns.

### 3.2 Matrix Factorization (NMF)

- Used Non-negative Matrix Factorization (NMF) with:
  - `n_components = 20`
  - Random initialization, `max_iter = 200`.
- Decomposed ratings matrix into:
  - User factors (latent preferences)
  - Item factors (latent attributes)
- Predicted rating as dot product of user and item factors.

### 3.3 Content-Based Filtering

- Built metadata for each item:
  - Title + description combined.
- Used TF-IDF vectorization and cosine similarity.
- For a given liked item, recommended items with highest similarity score.

### 3.4 Hybrid Model

- Combined NMF-based collaborative filtering with content-based scores.
- Normalized scores from both models.
- Final score:
  \[
  \text{score} = \alpha \cdot \text{collab\_score} + (1 - \alpha) \cdot \text{content\_score}
  \]
- Used \(\alpha = 0.6\) in experiments.

---

## 4. Evaluation

### 4.1 Train/Test Split

- Split ratings into train (80%) and test (20%) using random split.
- Ensured evaluation is done only on test data.

### 4.2 Metrics

- **RMSE (Root Mean Squared Error)** – measures rating prediction error.
- **MAE (Mean Absolute Error)** – average absolute difference between predicted and true ratings.
- **Precision@K (K=5)** – quality of top-5 recommendations per user (how many of them are actually relevant).

### 4.3 Results

| Model   | RMSE   | MAE   | Precision@5 |
|--------|--------|-------|-------------|
| CF     | …      | …     | …           |
| NMF    | …      | …     | …           |
| Hybrid | …      | …     | …           |

- **Observations**:
  - Hybrid model achieved the highest Precision@5 of …
  - NMF generally had lower RMSE/MAE than pure CF.
  - Hybrid provided a good balance between rating accuracy and top-K relevance.

(You’ll fill the numbers from `model_evaluation.ipynb`.)

---

## 5. System Design & Interface

### 5.1 Architecture

- Data & models implemented in `recommender/` package.
- Flask app (`app.py`) integrates:
  - User inputs (user_id, liked item).
  - Hybrid recommender for generating recommendations.
- Optional feedback route (`/feedback`) to capture user likes/dislikes.

### 5.2 Web Interface / Dashboard

- **Home Page**:
  - Form to enter user ID and select a liked item.
  - Displays top-N recommendations with scores.

- **Dashboard Page**:
  - Table summarizing RMSE, MAE, Precision@K for each model.
  - Can be extended with plots (bar charts, etc.).

---

## 6. Conclusion & Future Work

### 6.1 Conclusion

- Successfully implemented:
  - Collaborative filtering (user-based).
  - Matrix factorization (NMF).
  - Content-based filtering using TF-IDF.
  - Hybrid recommender combining multiple signals.
- Achieved improved performance with hybrid model in terms of Precision@K.

### 6.2 Future Improvements

- Implement more advanced models (e.g., SVD++, neural recommenders).
- Use implicit feedback (clicks, views) in addition to ratings.
- Online learning: update models in real-time with new feedback.
- Deploy fully on cloud platform (e.g., Render/Heroku) with persistent database.

---

## 7. References

- MovieLens dataset.
- Scikit-Learn documentation.
- Research papers/blogs on collaborative filtering and hybrid recommenders.
