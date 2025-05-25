import requests
import json  # Import necessario per il parsing sicuro

def generate_response_stream(context: list[str], question: str, model="llama2", lang="it", max_tokens=150):
    url = "http://localhost:11434/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    lang_prompts = {
        "it": (
            "Rispondi basandoti principalmente sul primo contesto, "
            "che è la descrizione dell'immagine caricata. "
            "Gli altri contesti sono informazioni di supporto."
        ),
        "en": (
            "Answer mainly based on the first context, "
            "which is the description of the uploaded image. "
            "Other contexts are supporting information."
        ),
        # aggiungi altre lingue qui se vuoi
    }

    system_prompt = lang_prompts.get(lang, lang_prompts["it"])

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ],
        "stream": True,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        for line in response.iter_lines():
            if line:
                line_data = line.decode('utf-8').replace("data: ", "")
                if line_data.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(line_data)
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    else:
                        yield f"[Error: response missing 'choices' -> {data}]"
                except Exception as err:
                    yield f"[Parsing error: {err}]"
    except Exception as e:
        yield f"[Error: {e}]"
