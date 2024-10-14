# importing LLM
from langchain_google_genai import ChatGoogleGenerativeAI
#importing for rag
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# importing for prompt
from langchain.prompts import ChatPromptTemplate
# importing for chaining
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# importing for env variables loader
from dotenv import load_dotenv
import os
load_dotenv()


# load google api key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# 1:RAG
pdf_loader = PyPDFLoader("GenerativeAI.pdf")
pdf_docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(pdf_docs)

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector = FAISS.from_documents(documents, embedding=embedding)

retriever = vector.as_retriever()


# 2:LLM
llm:ChatGoogleGenerativeAI =  ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,          
)

# response = llm.invoke("what is generative AI")
# print(response)


# 3:prompt
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided content. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the the answer helpful.
<context> 
{context}                                       
</context>   

Question:{input}                                     
""")

# 4:creating chains of llm, prompt and retriever
prompt_llm_chain = create_stuff_documents_chain(llm, prompt)

retriever_chain = create_retrieval_chain(retriever, prompt_llm_chain)


# 5:invoking the result
response = retriever_chain.invoke({"input":"what we learn in Quarter 1"})
print(response)
print(response['answer'])