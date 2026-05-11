import os

# Set the agent maximum reasoning steps
MAX_REPLY = 50

os.environ["AUTOGEN_USE_DOCKER"] = "False"

# --- Gemini Configuration ---
# Replace with your own Gemini API key and endpoint
GEMINI_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_URL = "YOUR_GEMINI_API_URL"  # e.g., "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

GEMINI_HEADERS = {
    "x-goog-api-key": GEMINI_KEY,
    "Content-Type": "application/json",
}

# --- GPT-4o Configuration ---
# Replace with your own OpenAI-compatible API URL and key
GPT_PROXY_URL = "YOUR_OPENAI_PROXY_URL"  # e.g., "https://api.openai.com"
GPT_KEY = "YOUR_OPENAI_API_KEY"
GPT_API_MODEL_NAME = "gpt-4o-2024-11-20"

# --- Qwen Configuration (vLLM) ---
# List of Qwen model vLLM server URLs for load balancing
# Replace with your own vLLM server addresses
QWEN_URL_DICT = {
    # "qwen25_vl_7b": ["http://<your-server-ip>:8000/v1"],
    # "qwen_8b_sft": ["http://<your-server-ip>:8000/v1"],
    "qwen_8b_instruct": ["http://<your-server-ip>:8000/v1"],
}

# --- InternVL Configuration (vLLM) ---
# InternVL3 / 3.5 vLLM server URL configuration
# Replace with your own InternVL deployment server addresses
INTERNVL_URL_DICT = {
    "internvl3_8b": ["http://<your-server-ip>:8000/v1"],   # Replace with your InternVL deployment server IP
    "internvl35_8b": ["http://<your-server-ip>:8000/v1"],  # Replace with your InternVL deployment server IP
}