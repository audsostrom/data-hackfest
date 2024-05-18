# Filename - server.py
 
# Import flask and datetime module for showing date and time
from flask import Flask, jsonify, request
import google.generativeai as genai
from flask_cors import CORS
import datetime
 
x = datetime.datetime.now()
 
# Initializing flask app
app = Flask(__name__)
CORS(app)

api_key = 'AIzaSyC8WizRY4zsJsqxpC1S9bUZY25yqoZEuOk'
genai.configure(api_key=api_key)
 
 
# Route for seeing a data
@app.route('/data')
def get_time():
    print('hi')
 
    # Returning an api for showing in  reactjs
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