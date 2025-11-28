from flask import Flask, request, jsonify
from core.processor import TextProcessor
import os

app = Flask(__name__)

# Initialize processor
# We do this lazily or globally. Global is fine for this scale.
try:
    processor = TextProcessor()
except ValueError as e:
    print(f"Error initializing processor: {e}")
    processor = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/preprocess', methods=['POST'])
def preprocess():
    if not processor:
        return jsonify({"error": "Processor not initialized. Check server logs/env vars."}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data['text']
    tasks = data.get('tasks', None) # None means all tasks

    try:
        results = processor.process_text(text, tasks)
        return jsonify({
            "status": "success",
            "results": results
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
