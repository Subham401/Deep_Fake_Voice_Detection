from flask import Flask, request, jsonify, render_template
import os
from detect_audio import predict_audio

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route("/")
def home():
    return render_template("front.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    result = predict_audio(file_path)
    os.remove(file_path)
    return jsonify({"result": result})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
