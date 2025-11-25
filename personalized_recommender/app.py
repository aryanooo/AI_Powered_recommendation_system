# app.py
from flask import Flask, render_template, request, redirect, url_for
from recommender.hybrid import HybridRecommender
from recommender.data_loader import load_items

app = Flask(__name__)

hybrid_model = HybridRecommender()

@app.before_first_request
def load_model():
    hybrid_model.fit()
    print("Hybrid model fitted and ready!")

@app.route("/", methods=["GET", "POST"])
def index():
    items = load_items()
    recommendations = []
    selected_item_id = None
    user_id = None

    if request.method == "POST":
        user_id = request.form.get("user_id")
        selected_item_id = request.form.get("item_id")

        user_id_int = int(user_id) if user_id else None
        item_id_int = int(selected_item_id) if selected_item_id else None

        if user_id_int is not None:
            recommendations = hybrid_model.recommend_for_user(
                user_id=user_id_int,
                liked_item_id=item_id_int,
                top_n=5
            )

    return render_template(
        "index.html",
        items=items.to_dict(orient="records"),
        recommendations=recommendations
    )

@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Simple feedback handler. In a full project:
    - Read user_id, item_id, feedback_type (like/dislike)
    - Convert to rating (e.g., like=5, dislike=1)
    - Append to ratings.csv
    - Re-train model periodically or on demand
    """
    user_id = request.form.get("user_id")
    item_id = request.form.get("item_id")
    feedback_type = request.form.get("feedback")  # 'like' or 'dislike'

    print(f"Received feedback: user {user_id}, item {item_id}, {feedback_type}")
    # TODO: Append to ratings.csv & retrain

    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    # In a real project you might load these from a JSON file
    summary = [
        {"model": "CF", "rmse": 0.95, "mae": 0.75, "precision": 0.42},
        {"model": "NMF", "rmse": 0.90, "mae": 0.70, "precision": 0.45},
        {"model": "Hybrid", "rmse": 0.88, "mae": 0.68, "precision": 0.50},
    ]
    return render_template("dashboard.html", summary=summary)


if __name__ == "__main__":
    app.run(debug=True)
