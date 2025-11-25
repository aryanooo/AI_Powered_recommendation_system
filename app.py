# app.py
import os
from flask import Flask, render_template, request, redirect, url_for

from recommender.hybrid import HybridRecommender
from recommender.data_loader import load_items

app = Flask(__name__)

# create a single global model object
hybrid_model = HybridRecommender()

# FIT MODEL ON STARTUP (Flask 3+ compatible)
print("[INFO] Fitting hybrid model...")
hybrid_model.fit()
print("[INFO] Hybrid model fitted and ready!")

# *************  ✨ Windsurf Command 🌟  *************
@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main entry point for the web interface.
    Handles both GET and POST requests.
    """
    # load items to populate the dropdown
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

# *******  c05b6b57-7fce-4981-9997-d5c981cd8a05  *******
@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Placeholder feedback endpoint.
    Currently just prints feedback to console.
    """
    user_id = request.form.get("user_id")
    item_id = request.form.get("item_id")
    feedback_type = request.form.get("feedback")

    print(f"[FEEDBACK] user={user_id}, item={item_id}, feedback={feedback_type}")

    return redirect(url_for("index"))

if __name__ == "__main__":
    print("[INFO] Starting Flask app...")
    app.run(debug=True)
