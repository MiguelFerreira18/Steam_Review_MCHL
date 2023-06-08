from flask import Flask, render_template
import RecomendationClass

app = Flask(__name__)

recomendationSystem = RecomendationClass()


@app.route("/")
def hello_world():
    frutas = ['Maçã', 'Laranja', 'Banana']
    conteudo = 'Conteúdo /n de /n exemplo'

    return render_template('index.html', frutas=frutas, conteudo=conteudo)

@app.route("/submit", methods=['POST'])
def submit_form():
    return "Form submitted successfully!"

if __name__ == "__main__":
    app.run(debug=True)
