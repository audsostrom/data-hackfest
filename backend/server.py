# Filename - server.py
 
# Import flask and datetime module for showing date and time
import os
from flask import Flask, jsonify, request
import google.generativeai as genai
from flask_cors import CORS
import datetime
 
x = datetime.datetime.now()
 
# Initializing flask app
app = Flask(__name__)
CORS(app)

api_key = os.getenv('GOOGLE_GENAI_API_KEY')
genai.configure(api_key=api_key)
 
 
# This is just for me to see if it works lol
@app.route('/data')
def dummy():
    print('hi')
    return {
        'Name':"geek", 
        "Age":"22",
        "Date":x, 
        "programming":"python"
    }


@app.route("/getmovieinfo", methods=["POST"])
def prompt():
    print()
    data = request.json
    print(data)
    year = data.get("year", "")
    title = data.get("title", "")


    if not year and not title:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"Tell about the movie {title} made in {year}"

        response = model.generate_content(prompt)

        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": "Request failed"}), 500

     
# Running app
if __name__ == '__main__':
    app.run(debug=True)