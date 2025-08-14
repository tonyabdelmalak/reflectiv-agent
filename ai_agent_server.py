"""
Flask-based AI agent server for Reflectiv.

This script provides endpoints for both chat and voice interactions. It uses
OpenAI's ChatCompletion API to generate responses and Twilio's voice
webhooks for phone calls. The persona prompt and knowledge base are loaded
from an external file to allow easy customization for Reflectiv or other
brands.

To run this server locally, install the dependencies:

    pip install flask openai twilio

Set the following environment variables before running:

    OPENAI_API_KEY    - your OpenAI API key
    OPENAI_MODEL      - optional; defaults to 'gpt-3.5-turbo'
    OPENAI_MAX_TOKENS - optional; defaults to '500'
    OPENAI_TEMPERATURE- optional; defaults to '0.7'
    PERSONA_FILE      - optional; path to the persona text file
    TWILIO_AUTH_TOKEN - for Twilio (if voice is enabled)

Note: Voice functionality requires the `twilio` package. If Twilio is not
installed, the `/voice` endpoint will return an error.
"""

import os
import uuid
from typing import Dict, List, Any
from flask import Flask, request, jsonify

# Attempt to import external libraries. If missing, set them to None.
try:
    import openai
except ImportError:
    openai = None  # type: ignore

try:
    from twilio.twiml.voice_response import VoiceResponse
except ImportError:
    VoiceResponse = None  # type: ignore


def load_persona() -> str:
    """Load the persona prompt from a file, or return a default."""
    persona_path = os.getenv("PERSONA_FILE", "reflectiv_persona.txt")
    if os.path.isfile(persona_path):
        with open(persona_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    # Default persona if file not found
    return (
        "You are a helpful virtual assistant representing Reflectiv. "
        "Answer questions to the best of your ability using clear and concise "
        "language. If you don't know the answer, offer to connect the user "
        "to a human representative."
    )


def ensure_openai_configured():
    """Ensure the OpenAI API key and library are available."""
    if openai is None:
        raise RuntimeError(
            "The openai package is not installed. Install it with 'pip install openai'."
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required.")
    openai.api_key = api_key


app = Flask(__name__)

# In-memory conversation store; in production, replace with a persistent store.
conversations: Dict[str, List[Dict[str, Any]]] = {}


def generate_response(session_id: str, user_message: str, persona_prompt: str) -> str:
    """
    Send the user message to OpenAI's ChatCompletion API and return the assistant's
    reply. Maintains history per session_id. If an error occurs, returns an
    explanatory message.
    """
    # Append user message to history
    history = conversations.setdefault(session_id, [])
    history.append({"role": "user", "content": user_message})
    # Build conversation context
    messages = [{"role": "system", "content": persona_prompt}] + history
    # Call OpenAI API
    try:
        ensure_openai_configured()
        response = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=messages,
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", 500)),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7)),
        )
        assistant_msg = response.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I'm unable to process your request right now: {e}"
    # Append assistant message
    history.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Handle chat requests from the web widget."""
    data = request.get_json() or {}
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"error": "Missing message"}), 400
    session_id = data.get('session_id', str(uuid.uuid4()))
    persona_prompt = load_persona()
    reply = generate_response(session_id, user_message, persona_prompt)
    return jsonify({"session_id": session_id, "response": reply})


@app.route('/voice', methods=['POST'])
def voice_endpoint():
    """
    Handle Twilio voice webhooks. Converts speech input to text (Twilio passes
    'SpeechResult'), uses the persona prompt, generates a response and returns
    TwiML to speak the reply.
    """
    if VoiceResponse is None:
        return "Voice support is not enabled on this server.", 501
    # Extract voice call variables
    user_message = request.form.get('SpeechResult', '')
    session_id = request.form.get('CallSid', str(uuid.uuid4()))
    persona_prompt = load_persona()
    reply = generate_response(session_id, user_message, persona_prompt)
    # Build TwiML response
    vr = VoiceResponse()
    vr.say(reply, voice=os.getenv("TWILIO_TTS_VOICE", "Polly.Joanna"))
    # Listen for next utterance
    vr.listen(timeout=5)
    return str(vr), 200


@app.route('/')
def index():
    """Basic index route to verify the server is running."""
    return (
        "<h1>Reflectiv AI Agent</h1>\n"
        "<p>This server provides /chat and /voice endpoints for the AI agent.</p>"
    )


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)