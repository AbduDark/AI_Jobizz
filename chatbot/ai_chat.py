import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT")

def ask_gemini(user_prompt: str, model_name: str = "models/gemini-1.5-flash") -> str:
    model = genai.GenerativeModel(model_name)
    convo = model.start_chat()

    full_prompt = f"{SYSTEM_PROMPT.strip()}\n\nUser: {user_prompt}\nSupport Bot:"
    response = convo.send_message(full_prompt)

    return response.text
