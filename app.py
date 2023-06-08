from flask import Flask, render_template, request,jsonify
from RecomendationClass import Recommendation
import json

app = Flask(__name__)

recomendationSystem = None

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/users", methods=['POST'])
def users():
    global recomendationSystem
    print("ENTREI PORRA")
    if(recomendationSystem is None):
        recomendationSystem = Recommendation()
    users = recomendationSystem.users.tolist()
    return jsonify(users)

@app.route("/invoke-function", methods=['POST'])
def invoke_function():
    selected_user = request.form['selected_user']
    allReviews = recomendationSystem.getReviews(selected_user)
    return render_template('index.html', users=recomendationSystem.users, allReviews=allReviews)

if __name__ == "__main__":
    app.run(debug=True)
