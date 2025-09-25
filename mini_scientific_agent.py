###Import section
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools.retriever import create_retriever_tool
from glob import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter

### Configuration section
##Caricamento variabili d'ambiente
load_dotenv()

embedding = OllamaEmbeddings(model=os.getenv("ollama-embedding-model-name"))

def create_vector_store(folder_path):

    list_of_files = glob(f"{folder_path}/*.pdf")
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
    for file in list_of_files:
        loader = PyPDFLoader(
        file,
        mode="page",
        )
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        documents.extend(all_splits)
    

    vector_store = Chroma(
        collection_name="knowlage_base",
        embedding_function=embedding,
        persist_directory=os.getenv("chroma-db-path"),  # Where to save data locally, remove if not necessary
    )
    if vector_store._collection.count() == 0:
        vector_store.add_documents(documents)

def get_retriver():
    if not os.path.exists("chroma_langchain_db"):
        raise ValueError("Vector store does not exist. Please create it first.")
    vector_store = Chroma(
        collection_name="knowlage_base",
        embedding_function=embedding,
        persist_directory=os.getenv("chroma-db-path"),,  # Where to save data locally, remove if not necessary
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})


##Cricamento modello
model = init_chat_model( model=os.getenv("ollama-model-name"), model_provider="ollama", timeout=30)

##Istaziamento di strumenti
search_engine = TavilySearch(max_results=2)
if not os.path.exists("./chroma_langchain_db"):
    print("Vector store non trovato, creazione")
    create_vector_store("./pdfs")
    print("Vector store creato")
retriver = get_retriver()
retrive_tool = create_retriever_tool(retriver,
    "RAG tool to Retrive information from the knowledge base",
    "RAG tool to Search and return information about any topic, use it to search in a complete knowledge base for any topic, use this tool when the user asks for a specific arguments, if you dont find anything use the search tool",
)
tools = [retrive_tool,search_engine]

##Istanza memoria
memory = MemorySaver()


#Istanza agente
agent_executor = create_react_agent(model, tools, checkpointer=memory)



### Function section
def get_streaming_answer(query: str, config: dict):
    """
    Risposta in streaming ad una query da parte dell'agente

    Args:
        query (str): Input per l'agente

    Yields:
        str: chunk di risposta da parte dell'agente
    """
    for chunk in agent_executor.stream(query, config,stream_mode="values"):
        yield chunk["messages"][-1].pretty_print()

def get_all_ollama_env() -> list:
    return [os.getenv(key) for key in os.environ.keys() if "ollama" in key.lower()]

def check_if_model_exists(model_name: str) -> bool:
    """
    Controlla se un modello esiste in ollama

    Args:
        model_name (str): Nome del modello da controllare

    Returns:
        bool: True se il modello esiste, False altrimenti
    """
    import requests

    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        models = response.json()
        for model in models['models']:
            if model["name"] == model_name:
                return True
    return False

def download_model(model_name: str) -> None:
    """
    Scarica un modello in ollama

    Args:
        model_name (str): Nome del modello da scaricare
    """
    import requests

    response = requests.post(
        "http://localhost:11434/api/pull",
        json={"model": model_name, "stream": True},
    )
    
    if response.status_code == 200:
        print(f"Model {model_name} downloaded successfully.")
    else:
        print(f"Failed to download model {model_name}. Status code: {response.status_code}")





### Main section

def main():
    ollama_models = get_all_ollama_env()
    print(f"Ollama-related environment variables:\n{ollama_models}\n")
    for model in ollama_models:
        if not check_if_model_exists(model):
            print(f"Model {model} not found. Downloading...")
            download_model(model)

    config = {"configurable": {"thread_id": "abc1234"}}

    input_message = {"messages": 
                     [{"role": "user", "content": "What did Gustav Muller-franzes wrote about?"},
                      ]}

    print("\nStreaming answer:")
    for chunk in get_streaming_answer(input_message,config):
        print(chunk, end='', flush=True)



### run
if __name__ == "__main__":


    main()