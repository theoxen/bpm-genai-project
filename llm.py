"""
Shared LLM helper used by the GUI (Requirements 3, 4, 5).

The two LLMs compared in this project come from two different companies and both are free to use:
  Model A - Google Gemini 3.5 Flash
  Model B - OpenAI gpt-oss-120b, an open model hosted by Groq
Both services accept the same "OpenAI-compatible" requests, so one library (openai) talks to both.
The keys are read from the GEMINI_API_KEY and GROQ_API_KEY environment variables (or Streamlit secrets).
"""
import os
import re
import time

from openai import OpenAI

# name shown in the GUI -> (model id, address of the service, environment variable that holds the key)
MODELS = {
    "Model A - Gemini 3.5 Flash": ("gemini-3.5-flash",
                                   "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "Model B - gpt-oss-120b (Groq)": ("openai/gpt-oss-120b", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}

# Longest answer allowed. Groq cuts long code answers off at a low default (seen at 3072 tokens), so it gets
# a higher limit. Gemini keeps its default: its limit also counts its hidden "thinking", so a limit cuts it off.
MAX_TOKENS = {"Model B - gpt-oss-120b (Groq)": 16000}


def get_client(model_name):
    _, base_url, key_name = MODELS[model_name]
    if not os.environ.get(key_name):
        raise RuntimeError(f"Missing API key: set {key_name}")
    return OpenAI(base_url=base_url, api_key=os.environ[key_name])


def ask_llm(model_name, system, prompt):
    """Send one prompt and return the answer text + the numbers needed for Requirement 7."""
    start = time.time()
    response = get_client(model_name).chat.completions.create(
        model=MODELS[model_name][0],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        **({"max_tokens": MAX_TOKENS[model_name]} if model_name in MAX_TOKENS else {}),
    )
    return {
        "text": response.choices[0].message.content or "",
        "seconds": round(time.time() - start, 2),
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }


def extract_code(text):
    """Take the Python code out of a ```python ... ``` block in the LLM answer."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text


# List the model names offered by each service
if __name__ == "__main__":
    for name in MODELS:
        print(name)
        for model in get_client(name).models.list():
            print("   ", model.id)
