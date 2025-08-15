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

# Try importing gTTS for offline text‑to‑speech. If this import fails,
# the /speak endpoint will return an error. gTTS uses Google's translate
# service to synthesize speech and does not require an API key. It
# fetches audio from Google when called, so a working internet
# connection is required on the server. The generated audio will be
# returned to the client as a base64-encoded MP3.
try:
    from gtts import gTTS  # type: ignore
    import io
    import base64
except ImportError:
    gTTS = None  # type: ignore
    io = None  # type: ignore
    base64 = None  # type: ignore

# Attempt to import external libraries. If missing, set them to None.
# The newer OpenAI Python library (>=1.0) exposes an `OpenAI` client class rather
# than static methods. We attempt to import it here. If the import fails, the
# variable will remain `None` and a RuntimeError will be raised when the API
# client is needed.
try:
    # In openai>=1.0 the primary entry point is the `OpenAI` class.
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore

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


# Global OpenAI client instance. This will be initialized in `ensure_openai_configured`.
openai_client = None


def ensure_openai_configured() -> None:
    """
    Ensure that the OpenAI Python client is available and configured. This
    function initializes a global `openai_client` instance using the
    `OPENAI_API_KEY` environment variable. It should be called before any
    requests to the OpenAI API.
    """
    global openai_client
    if openai_client is not None:
        return
    if OpenAI is None:
        raise RuntimeError(
            "The openai package is not installed. Install it with 'pip install openai'."
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required.")
    # Initialize a new OpenAI client with the provided API key. Additional
    # parameters (such as `organization` or `project`) could be supplied via
    # environment variables if needed.  If a custom API base URL is provided
    # (for example, when using an OpenAI‑compatible provider like OpenRouter),
    # set the client's `base_url` after instantiation.
    openai_client = OpenAI(api_key=api_key)
    # Apply a custom base URL if configured.  The environment variable
    # OPENAI_BASE_URL is recognized by the newer OpenAI client; if set it
    # overrides the default API endpoint.  This is useful when using
    # OpenAI‑compatible providers such as openrouter.ai.
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if base_url:
        try:
            # Cast to string to satisfy type checkers
            openai_client.base_url = str(base_url)
        except Exception:
            # Ignore invalid URLs; the client will fall back to the default
            pass


app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) to allow the web chat widget on
# external domains (e.g., GitHub Pages) to interact with this server. Without
# CORS, browsers will block POST requests from different origins, leading to
# connection errors in the chat widget. The flask-cors package adds the
# appropriate Access-Control-Allow-Origin headers to all responses. If
# flask-cors is not installed, CORS support will not be enabled and the
# server will continue to function for same-origin requests.
try:
    from flask_cors import CORS  # type: ignore
    CORS(app)
except ImportError:
    # If flask_cors isn't available, you can install it via requirements.txt
    # or handle CORS manually by adding headers in individual responses.
    pass

# In-memory conversation store; in production, replace with a persistent store.
conversations: Dict[str, List[Dict[str, Any]]] = {}


def generate_response(session_id: str, user_message: str, persona_prompt: str) -> str:
    """
    Send the user message to OpenAI's chat API and return the assistant's reply.
    Maintains conversation history per session_id. If an error occurs, returns
    an explanatory message instead of raising.

    The newer OpenAI Python library (>=1.0) uses an instance of `OpenAI` to
    access resources. The `chat.completions.create` method is used to
    generate responses from chat models. This function builds a message list
    consisting of a system prompt followed by the conversation history and
    sends it to the API.
    """
    # Append the user's message to the session history
    history = conversations.setdefault(session_id, [])
    history.append({"role": "user", "content": user_message})
    # Combine persona/system prompt with history
    messages = [{"role": "system", "content": persona_prompt}] + history
    try:
        ensure_openai_configured()
        # Determine model and parameters from environment with sensible defaults.
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", 500))
        temperature = float(os.getenv("OPENAI_TEMPERATURE", 0.7))
        # Use the client to create a chat completion. See
        # https://platform.openai.com/docs/api-reference/chat/create for details.
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Extract assistant message
        assistant_msg = response.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I'm unable to process your request right now: {e}"
    # Append assistant response to history
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

# ----------------------------------------------------------------------------
# Text‑to‑Speech endpoint
#
# The /speak endpoint provides an alternative way for the web chat to obtain
# speech audio when client‑side SpeechSynthesis is unreliable. It accepts
# JSON with a `text` field, uses gTTS to synthesize the speech, and
# returns a JSON response with a base64‑encoded MP3. Clients can decode
# this data URI and play it with the HTML5 Audio API. If gTTS is not
# installed or cannot fetch audio, an error message is returned.
@app.route('/speak', methods=['POST'])
def speak_endpoint():
    data = request.get_json() or {}
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "Missing text"}), 400
    # Ensure gTTS is available
    if gTTS is None or io is None or base64 is None:
        return jsonify({"error": "Text-to-speech is not available on this server."}), 500
    try:
        # Use gTTS to generate speech. Language defaults to English.
        tts = gTTS(text=text, lang='en')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        encoded_audio = base64.b64encode(audio_bytes.read()).decode('utf-8')
        return jsonify({"audio": encoded_audio, "format": "mp3"})
    except Exception as e:
        return jsonify({"error": f"TTS generation failed: {e}"}), 500


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