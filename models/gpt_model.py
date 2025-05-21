import os
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("Chiave API caricata:", api_key)

openai.api_key = api_key  # <-- Importantissimo! 

def generate_response(context: list[str], question: str, max_tokens=150) -> str:
    messages = [
        {"role": "system", "content": "Sei un assistente che risponde usando il contesto fornito."},
        {"role": "user", "content": f"Contesto: {context}\nDomanda: {question}"}
    ]

    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        if "RateLimitError" in str(type(e)):
            return "Errore: Quota API superata. Attendi o aggiorna il tuo piano OpenAI."
        else:
            return f"Errore generico: {e}"

if __name__ == "__main__":
    test_context = ["Questa è una descrizione di esempio."]
    test_question = "Qual è la descrizione?"
    risposta = generate_response(test_context, test_question)
    print("Risposta generata:", risposta)
