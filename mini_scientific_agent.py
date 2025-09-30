###Import section
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from vector_store import vector_store
from model_manager import check_and_download

class msa:
    def __init__(self):
        load_dotenv()
        ollama_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

        ##Cricamento modello
        check_and_download(os.getenv("ollama-model-name"))
        model = init_chat_model(model=os.getenv("ollama-model-name"), model_provider="ollama", timeout=30, base_url=ollama_url)

        ##Istaziamento di strumenti
        search_engine = TavilySearch(max_results=2,description = 'A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. It not only retrieves URLs and snippets, but offers advanced search depths, domain management, time range filters, and image search, this tool delivers real-time, accurate, and citation-backed results. Useful to find information about anything that is not in the knowledge base, use it for anything that regards any question. Use it for general purpose needs.Input should be a search query.')

        v_store = vector_store()
        retrive_tool = v_store.check_and_load_retriever_tool()

        tools = [retrive_tool,search_engine]

        ##Istanza memoria
        memory = MemorySaver()

        #Istanza agente
        self.agent_executor = create_react_agent(model, tools, checkpointer=memory)

    ### Function section
    def get_streaming_answer(self,query: str, config: dict):
        """
        Risposta in streaming ad una query da parte dell'agente

        Args:
            query (str): Input per l'agente

        Yields:
            str: chunk di risposta da parte dell'agente
        """
        for chunk in self.agent_executor.stream(query, config, stream_mode="messages"):
            current_chunk = chunk[0] if isinstance(chunk, tuple) else chunk
            if hasattr(current_chunk, "content") and current_chunk.content:
                text = current_chunk.content
                yield text


### run
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "abc1234"}}

    input_message = {"messages": 
                     [{"role": "system", "content": "Use all the tool you are provided with starting from RAG and fallback on web search, provide the user with an answer and reference"},
                      {"role": "user", "content": "What did Gustav Muller-franzes wrote about?"},
                      ]}

    print("\nStreaming answer:")
    agent = msa()
    for chunk in agent.get_streaming_answer(input_message,config):
        print(chunk, end='', flush=True)