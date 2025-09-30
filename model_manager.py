import requests
import os

ollama_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

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

    response = requests.get(f"{ollama_url}/api/tags")
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
        f"{ollama_url}/api/pull",
        json={"model": model_name, "stream": True},
    )
    
    if response.status_code == 200:
        print(f"Model {model_name} downloaded successfully.")
    else:
        raise Exception(f"Failed to download model {model_name}. Status code: {response.status_code}")


def check_and_download(model_name: str) -> None:
    """
    Controlla se un modello esiste in ollama e lo scarica se non esiste

    Args:
        model_name (str): Nome del modello da controllare e scaricare
    """
    if not check_if_model_exists(model_name):
        print(f"Model {model_name} not found. Downloading...")
        download_model(model_name)
    else:
        print(f"Model {model_name} already exists.")