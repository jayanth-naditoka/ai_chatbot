from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the model
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction="You are an expert at teaching science to kids...",
)

# Start a chat session
chat_session = model.start_chat(history=[])

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']

    # Send the message to Gemini AI
    response = chat_session.send_message(user_input)
    model_response = response.text

    # Update the chat history
    chat_session.history.append({"role": "user", "parts": [user_input]})
    chat_session.history.append({"role": "model", "parts": [model_response]})

    # Return the response to the frontend
    return jsonify({"reply": model_response})


if __name__ == '__main__':
    app.run(debug=True)
