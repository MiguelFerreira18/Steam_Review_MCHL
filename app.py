from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/submit", methods=['POST'])
def submit_form():
    app_id = request.form.get("app_id")
    app_name = request.form.get("app_name")
    review_id = request.form.get("review_id")
    language = request.form.get("language")
    review = request.form.get("review")
    timestamp_created = request.form.get("timestamp_created")
    timestamp_updated = request.form.get("timestamp_updated")
    recommended = request.form.get("recommended")
    votes_helpful = request.form.get("votes_helpful")
    votes_funny = request.form.get("votes_funny")
    weighted_vote_score = request.form.get("weighted_vote_score")
    comment_count = request.form.get("comment_count")
    steam_purchase = request.form.get("steam_purchase")
    received_for_free = request.form.get("received_for_free")
    written_during_early_access = request.form.get("written_during_early_access")
    author_steamid = request.form.get("author_steamid")
    author_num_games_owned = request.form.get("author_num_games_owned")
    author_num_reviews = request.form.get("author_num_reviews")
    author_playtime_at_review = request.form.get("author_playtime_at_review")
    author_playtime_forever = request.form.get("author_playtime_forever")
    author_playtime_last_two_weeks = request.form.get("author_playtime_last_two_weeks")
    author_last_played = request.form.get("author_last_played")
    

    return "Form submitted successfully!"

if __name__ == "__main__":
    app.run(debug=True)
