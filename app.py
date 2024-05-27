from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from forms import UrlForm
from autocomplete import correct_and_autocomplete_url, set_history
app = Flask(__name__)
app.config['SECRET_KEY'] = 'test'
CORS(app)


@app.route("/")
def main():
    return "<p>This is the server</p>"


@app.route("/app", methods=['GET', 'POST', 'OPTIONS'])
def url_form():
    form = UrlForm()
    result = []
    corrected_url = ""
    return render_template('index.html', form=form, result=result, corrected_url=corrected_url)


@app.route("/post_form", methods=["POST"])
def post_form():
    result = []
    corrected_url = ""
    try:
        url = request.json["url"]
        corrected_url, result = correct_and_autocomplete_url(url)
        return jsonify(result=result, corrected_url=corrected_url)

    except KeyError as e:
        # Handle the case where "url" key is not present in the request.json
        return jsonify(error=str(e))

@app.route("/post_ext", methods =["POST"])
def post_ext():
    result = []
    corrected_url = ""
    try:
        url = request.json["url"]
        corrected_url, result = correct_and_autocomplete_url(url)
        return jsonify(result=result, corrected_url=corrected_url)

    except KeyError as e:
        # Handle the case where "url" key is not present in the request.json
        return jsonify(error=str(e))

@app.route("/post_history", methods=["POST"])
def post_history():
    try:
        history = request.json["data"]
        result = set_history(history=history)
        return jsonify(result= "data recieved")
    except KeyError as e:
        return jsonify(error=str(e))
    
if __name__ == "__main__":
    app.run(debug=True, port=8080)
