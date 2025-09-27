# Mini-scientific-agent

# 🧠 Agente RAG Locale con Ollama, LangChain & Tavily

Un agente AI minimale ma completo che combina **LLM locali**, **retrieval-augmented generation (RAG)** e **fallback su motore di ricerca**, progettato per girare in modo efficiente anche su hardware modesto.

L’obiettivo di questo progetto è:

* Imparare a costruire un agente AI passo dopo passo.
* Esplorare l’integrazione di diversi componenti di LangChain.
* Mantenere il tutto leggero abbastanza da girare su un portatile con **8GB di RAM e CPU only**.
* Preparare un ambiente portabile (tramite **Docker**).

---

## ✨ Funzionalità

* 🔹 **Inferenza LLM locale** tramite [Ollama](https://ollama.ai/) (supporto a modelli piccoli/quantizzati come `phi`, `mistral`, `llama2`).
* 🔹 **Pipeline RAG**: caricamento di PDF, suddivisione in chunk e memorizzazione in **Chroma** come vector DB.
* 🔹 **Ricerca web** tramite [Tavily](https://python.langchain.com/docs/integrations/tools/tavily), usata come fallback se il RAG non ha informazioni rilevanti.
* 🔹 **Agente ReAct** basato su LangGraph, capace di decidere quando usare RAG o il motore di ricerca.

---

## 📂 Struttura del progetto

* `pdfs/` → cartella per i documenti da indicizzare.
* `chroma_langchain_db/` → database persistente con gli embeddings.
* `main.py` → entrypoint del progetto, inizializza modelli, DB, strumenti e agente.
* `.env` → variabili d’ambiente (modelli Ollama, path del DB, ecc.).

---

## ✅ Stato del progetto

* [x] Setup ambiente e variabili d’ambiente (.env)
* [x] Integrazione con Ollama per LLM ed embeddings
* [x] Creazione e persistenza di un Vector Store (Chroma)
* [x] Caricamento e split di documenti PDF
* [x] Implementazione del RAG retriever come tool
* [x] Integrazione di un motore di ricerca (Tavily) come fallback
* [x] Creazione dell’agente ReAct con LangGraph
* [ ] Creare un’interfaccia web minimale
* [ ] Internazionalizzazione completa (IT/EN)
* [ ] Containerizzazione con Docker
* [ ] Ottimizzazione prestazioni e logging avanzato

---

## 🚀 Requisiti

* Python 3.10+
* [Ollama](https://ollama.ai/) installato e in esecuzione in locale
* Tavily API key (gratuita, da inserire nel `.env`)
* Librerie Python (vedi `requirements.txt`)

---

## ▶️ Avvio

1. Clona la repo
2. Copia il file `.env.example`, rinominarlo in `.env` ed inserire i parametri necessari (es. modello Ollama, path DB, API key)
3. Inserisci i PDF in `./pdfs`
4. Avvia lo script:

```bash
python main.py
```

---

## 🔜 Prossimi sviluppi

* Creare una **UI web minimale** (Flask/FastAPI + HTMX/VanillaJS).
* Packaging in Docker per facile distribuzione.

---
