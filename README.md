# Rag Images

Rag Images è un'applicazione che utilizza l'API OpenAI per analizzare immagini caricate e rispondere a domande basate sul contenuto visivo. L’interfaccia è realizzata con Streamlit per un uso semplice e interattivo.

## Funzionalità

- Caricamento di immagini e analisi dei dettagli.
- Risposte a domande relative alle immagini tramite il modello GPT-3.5-turbo.
- Gestione degli errori legati alla quota API e chiave invalida.
- Supporto per un ambiente di sviluppo semplice e configurabile.

## Requisiti

- Python 3.8 o superiore
- openai
- streamlit
- python-dotenv
- pillow
- llama-index
- qdrant-client

## Installazione

1. Clona il repository:

```bash
git clone https://github.com/tuo-username/rag-images.git
cd rag-images
Crea e attiva un ambiente virtuale:


python -m venv venv
source venv/bin/activate    # su Windows: venv\Scripts\activate
Installa le dipendenze:


pip install -r requirements.txt
Configura la tua chiave API OpenAI nel file .env:


OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Uso
Per avviare l'app:

streamlit run app.py
Una volta aperta l’interfaccia, puoi caricare un’immagine e porre domande come "quante foglie?" o qualsiasi altra cosa inerente all’immagine.

Note Importanti
Assicurati che la tua chiave API sia valida e che il tuo account OpenAI abbia credito disponibile.

In caso di superamento della quota, l’app mostrerà un messaggio di errore.

Per qualsiasi problema legato all’API, verifica la configurazione del file .env e le impostazioni di rete.

