from flask import Flask, request, jsonify, render_template
import json
import os

app = Flask(__name__, template_folder='templates')

# 피드백 데이터를 저장할 리스트
feedback_data = []

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    original = data.get('original')
    feedback = data.get('feedback')

    # 피드백 데이터를 리스트에 추가
    feedback_entry = {
        'original': original,
        'feedback': feedback
    }
    feedback_data.append(feedback_entry)
    save_feedback_data()

    return jsonify({'status': 'success'}), 200

def save_feedback_data():
    with open('feedback_data.json', 'w', encoding='utf-8') as f:
        json.dump(feedback_data, f, ensure_ascii=False, indent=4)

def load_feedback_data():
    global feedback_data
    try:
        with open('feedback_data.json', 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        feedback_data = []

@app.route('/feedback', methods=['GET'])
def get_feedback():
    return jsonify(feedback_data), 200

@app.route('/view_feedback', methods=['GET'])
def view_feedback():
    load_feedback_data()  # Load feedback data from the file
    return render_template('view_feedback.html', feedback_data=feedback_data)

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    load_feedback_data()  # Load feedback data on server start
    app.run(debug=True)