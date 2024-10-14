# imports for llm 
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
# imports for RAG
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool # to make the (RAG) retriever as tool
# imports for pre defiened tools
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
# imports for predefined prompts
from langchain import hub 
# imports for agent
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
# imports for env loading
from dotenv import load_dotenv
import os

load_dotenv()



google_api_key: str | None  =  os.getenv("GOOGLE_API_KEY")

# 1: LLM
llm:ChatGoogleGenerativeAI =  ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Specify the model to use
    temperature=0.2,          
)

# response = llm.invoke("what is machine learning")
# print(response)


# RAG as tool
web_loader = WebBaseLoader("https://www.langchain.com/langsmith")
web_pages = web_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(web_pages)

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector = FAISS.from_documents(documents, embedding=embedding)

retriever = vector.as_retriever()

# make the retriver as tool
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "search for information about langsmith"
    )


#wikipedia tool
api_wrapper = WikipediaAPIWrapper(wiki_client= Any,top_k_results=1, doc_content_chars_max=250,)
wikipedia_tool  = WikipediaQueryRun(api_wrapper=api_wrapper)

# arxiv tool
arxiv_wrapper = ArxivAPIWrapper(arxiv_search= Any,arxiv_exceptions= Any,top_k_results=1, doc_content_chars_max=250)
arxiv_tool  = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# tools
tools = [retriever_tool, arxiv_tool,wikipedia_tool]

# 3: Prompt

prompt = hub.pull("hwchase17/openai-functions-agent")


agent = create_tool_calling_agent(llm,tools,prompt)

agent_exicuter = AgentExecutor(agent=agent,tools=tools , verbose=True)

agent_exicuter.invoke({"input":"tell me about langsmith"})