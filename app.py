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
    print(recomendationSystem.users[0])
    users = [str(user) for user in recomendationSystem.users] 

    return jsonify(users)

@app.route("/invoke-function", methods=['POST'])
def invoke_function():
    selected_user = request.form.get('selected_user')
    print(selected_user)
    allReviews = recomendationSystem.getReviews(selected_user)
    for x in allReviews:
        print(x)
    users = recomendationSystem.users
    print(allReviews.values.tolist())
    return jsonify(users=users, allReviews=allReviews.values.tolist())

@app.route("/submit", methods=['POST'])
def submit():
    user = request.form.get('selected_user')
    recomendation = recomendationSystem.generate_recommendation(user)
    return jsonify(recomendation=recomendation)


if __name__ == "__main__":
    app.run(debug=True)
