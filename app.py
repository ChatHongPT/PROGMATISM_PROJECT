from flask import Flask, request, jsonify, render_template_string
import json

app = Flask(__name__)

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
    except FileNotFoundError:
        feedback_data = []

@app.route('/feedback', methods=['GET'])
def get_feedback():
    return jsonify(feedback_data), 200

@app.route('/view_feedback', methods=['GET'])
def view_feedback():
    load_feedback_data()  # Load feedback data from the file
    feedback_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Feedback Data</title>
    </head>
    <body>
        <h1>Feedback Data</h1>
        <ul>
            {% for entry in feedback_data %}
                <li><strong>Original:</strong> {{ entry.original }} <br><strong>Feedback:</strong> {{ entry.feedback }}</li>
            {% endfor %}
        </ul>
        <a href="/">Go back to Home</a>
    </body>
    </html>
    """
    return render_template_string(feedback_html, feedback_data=feedback_data)

@app.route('/')
def home():
    home_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Home</title>
    </head>
    <body>
        <h1>Speech Recognition and Feedback</h1>
        <button onclick="startRecording()">Start Recording</button>
        <p><strong>음성 인식 결과:</strong></p>
        <p id="transcription"></p>
        <textarea id="feedback" placeholder="Enter the correct transcription"></textarea>
        <button onclick="submitFeedback()">Submit Feedback</button>
        <br><br>
        <a href="/view_feedback">View Feedback</a>

        <script>
            let recognition;

            function startRecording() {
                recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                recognition.lang = 'ko-KR';
                recognition.start();

                recognition.onresult = function(event) {
                    const transcription = event.results[0][0].transcript;
                    document.getElementById('transcription').innerText = transcription;
                };

                recognition.onerror = function(event) {
                    console.error(event.error);
                };
            }

            function submitFeedback() {
                const originalTranscription = document.getElementById('transcription').innerText;
                const feedbackTranscription = document.getElementById('feedback').value;

                fetch('/submit_feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        original: originalTranscription,
                        feedback: feedbackTranscription
                    })
                }).then(response => response.json())
                  .then(data => alert('Feedback submitted!'))
                  .catch(error => console.error('Error:', error));
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(home_html)

if __name__ == '__main__':
    load_feedback_data()  # Load feedback data on server start
    app.run(debug=True)