from langchain_community.embeddings import huggingface
from langchain_community.document_loaders import TextLoader, DirectoryLoader, GitLoader
from langchain_community.llms import huggingface_hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import chroma, faiss
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import sys
import re
from urllib.parse import urlparse
import ast
import shutil
from git import Repo
import streamlit as st
from dotenv import load_dotenv
from tempfile import TemporaryDirectory
load_dotenv()
HF_API_KEY = os.getenv('HF_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
def get_vectorstore_from_text(repo_url):
     # print('hi')
     temp = f"./{repo_url}"
     with TemporaryDirectory() as temp:
          loader = GitLoader(
               clone_url=repo_url,
               repo_path=temp+"/test_repo",
               branch="master",
               file_filter=lambda file_path: file_path.endswith('.py') or file_path.endswith('.md')
          )
          document = loader.load()
          print(document)


          # file = f"./{repo_name}_analysis.txt"
          # document = TextLoader(file).load()
          # document = DirectoryLoader("./repo_text", glob=f"{repo_name}_analysis.txt", loader_cls=TextLoader).load()
          # document = DirectoryLoader(f"./{repo_name}/cloned_repo", glob="**/*.py", loader_cls=PythonLoader).load()
          # shutil.rmtree("./example_data")
          # loader = WebBaseLoader(url)
          # document = loader.load()
          text_splitter = RecursiveCharacterTextSplitter()
          document_chunks = text_splitter.split_documents(document)
          embeddings = OpenAIEmbeddings(
               model='text-embedding-ada-002'
          )
          # embeddings = huggingface.HuggingFaceEmbeddings()
          st.session_state.vector_store = chroma.Chroma()
          # for x in range(len(vector_store)):
               # vector_store.delete(ids=[x])
          # print(vector_store.delete_collection())
          st.session_state.vector_store = chroma.Chroma.from_documents(document_chunks, embeddings)
          # return st.session_state.vector_store

def get_context_retriever_chain(vector_store):

     llm = ChatOpenAI(model='gpt-3.5-turbo')
     # llm = huggingface_hub.HuggingFaceHub(
          # huggingfacehub_api_token=HF_API_KEY,
          # task='text-generation'
          # repo_id='CohereForAI/c4ai-command-r-plus'
     # )
     retriever = vector_store.as_retriever()
     
     prompt = ChatPromptTemplate.from_messages([
          MessagesPlaceholder(variable_name="chat_history"),
          ("user", "{input}"),
          ("user", '''
                    Given the above conversation, generate a search query to look up in 
                    order to get information relevant to the conversation.
               ''')
     ])
     
     retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
     return retriever_chain

def get_conversational_rag_chain(retriever_chain): 
    
     llm = ChatOpenAI(model='gpt-3.5-turbo')
     # llm = huggingface_hub.HuggingFaceHub(
          # huggingfacehub_api_token=HF_API_KEY,
          # task='text-generation'
     # )
     prompt = ChatPromptTemplate.from_messages([
          ("system", '''
               Answer the user's questions based on the below context:
               \n\n{context}
          '''),
          MessagesPlaceholder(variable_name="chat_history"),
          ("user", "{input}")
     ])
     
     stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
     return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
     retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
     conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
     
     response = conversation_rag_chain.invoke({
          "chat_history": st.session_state.chat_history,
          "input": user_input
     })
     
     return response['answer']

# app config
def main():
     # __import__('pysqlite3')
     # sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
     st.set_page_config(page_title="Chat", layout='wide')
     st.title("GitHub Bot")
     with st.sidebar:
          st.header("Enter GitHub Repository URL")
          git_url = st.text_input("")

     if git_url is None or git_url == "":
          st.info("Chat is Empty...")

     else:
          if "chat_history" not in st.session_state:
               st.session_state.chat_history = [
                    AIMessage(content="Ask me about the repository..."),
               ]
          if "git_url" not in st.session_state:
               st.session_state.git_url = git_url
          if "vector_store" not in st.session_state:
               # repo_name = extract_repo_name(git_url)
               # save_repo_analysis(git_url, repo_name)
               get_vectorstore_from_text(st.session_state.git_url) 
          # user input
          user_query = st.chat_input("Type your message here...")
          if user_query is not None and user_query != "":
               response = get_response(user_query)
               st.session_state.chat_history.append(HumanMessage(content=user_query))
               st.session_state.chat_history.append(AIMessage(content=response))
          

          # conversation
          for message in st.session_state.chat_history:
               if isinstance(message, AIMessage):
                    with st.chat_message("AI"):
                         st.write(message.content)
               elif isinstance(message, HumanMessage):
                    with st.chat_message("Human"):
                         st.write(message.content)


if __name__ == "__main__":
    main()