from dotenv import load_dotenv
import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from glob import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from model_manager import check_and_download

class vector_store:
    def __init__(self):
        load_dotenv()
        embedding_model_name = os.getenv("ollama-embedding-model-name")
        check_and_download(embedding_model_name)
        self.embedding = OllamaEmbeddings(model=embedding_model_name)
        self.pdf_path = os.getenv("pdf-dir")
        self.retriever = None
    def create_vector_store(self,folder_path):
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
            embedding_function=self.embedding,
            persist_directory=os.getenv("chroma-db-path"),  # Where to save data locally, remove if not necessary
        )
        if vector_store._collection.count() == 0:
            vector_store.add_documents(documents)

    def get_retriver(self):
        if not os.path.exists("chroma_langchain_db"):
            raise ValueError("Vector store does not exist. Please create it first.")
        vector_store = Chroma(
            collection_name="knowlage_base",
            embedding_function=self.embedding,
            persist_directory=os.getenv("chroma-db-path"),  # Where to save data locally, remove if not necessary
        )
        return vector_store.as_retriever(search_kwargs={"k": 5})

    def check_and_load_retriever_tool(self):
        if not os.path.exists("chroma_langchain_db"):
            print("Vector store not found, creating...")
            self.create_vector_store("./pdfs")
            print("Vector store created.")
        retriver = self.get_retriver()
        retrive_tool = create_retriever_tool(retriver,
            "RAG tool to Retrive information from the knowledge base",
            "RAG tool to Search and return information about scientific topic, use it to search in a complete knowledge base for scientific topic, use this tool when the user asks for a scientific arguments, if you dont find anything use the search tool",
        )
        return retrive_tool