# from langchain_community.embeddings.huggingface import 
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.git import GitLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.python import PythonLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

# from langchain_community.llms import huggingface_hub
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders.parsers.language import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.lancedb import LanceDB
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
import os
from git import Repo
import sys
import streamlit as st
from dotenv import load_dotenv
from tempfile import TemporaryDirectory
load_dotenv()
HF_API_KEY = os.getenv('HF_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
def create_directory_loader(file_type, directory_path):
     loaders = {
          '.py': PythonLoader,
          '.txt': TextLoader,
          '.md': UnstructuredMarkdownLoader,
     }
     return DirectoryLoader(
          path=directory_path,
          glob=f"**/*{file_type}",
          loader_cls=loaders[file_type],
     )

def get_vectorstore_from_text(repo_url):
     temp = f"./{repo_url}"
     with TemporaryDirectory() as temp:
          # loader = GitLoader(
          #      clone_url=repo_url,
          #      repo_path=temp+"/test_repo",
          #      branch='master',
          #      file_filter=lambda file_path: file_path.endswith('.py')
          # )

          # document = loader.load()

          Repo.clone_from(repo_url, to_path=temp+"/test_repo")
          # loader = GenericLoader.from_filesystem(
          #      temp+"/test_repo",
          #      glob="**/*",
          #      suffixes=[".py", ".txt"],
          #      exclude=["**/non-utf8-encoding.py"],
          #      parser=LanguageParser()
          # )
          # docs = loader.load()

          py_loader = create_directory_loader('.py', temp+"/test_repo")
          txt_loader = create_directory_loader('.txt', temp+"/test_repo")
          md_loader = create_directory_loader('.md', temp+"/test_repo")

          py_documents = py_loader.load()
          txt_documents = txt_loader.load()
          md_documents = md_loader.load()
          docs = py_documents + txt_documents + md_documents
          # root_dir = temp+"/test_repo"
          # docs = []
          # for dirpath, dirnames, filenames in os.walk(root_dir):
          #      for file in filenames:
          #           file_extension = os.path.splitext(file)[1]
          #           if file_extension in ['.py', '.txt']:
          #                try:
          #                     loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
          #                     docs.extend(loader.load_and_split())
          #                except Exception as e:
          #                     pass

          print(len(docs))
          text_splitter = RecursiveCharacterTextSplitter(
                    # language=Language.PYTHON,
                    chunk_size=1000, chunk_overlap=50
               )
          document_chunks = text_splitter.split_documents(docs)
          print(document_chunks)
          embeddings = OpenAIEmbeddings(
               model='text-embedding-ada-002'
          )
          # embeddings = huggingface.HuggingFaceEmbeddings()
          # st.session_state.vector_store = chroma.Chroma()
          # for x in range(len(vector_store)):
               # vector_store.delete(ids=[x])
          # print(vector_store.delete_collection())
          # if "vector_store" not in st.session_state:
               # st.session_state.vector_store = Chroma.from_documents(document_chunks, embeddings)
          vector_store = LanceDB.from_documents(document_chunks, embeddings)
          return vector_store

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
          ("human", "{input}"),
          ("system", '''
                    Given the above conversation and the latest user question,
                    which might reference context in the chat history,
                    generate a standalone question which can be understood without the chat history.
                    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
               ''')
     ])
     
     retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
     return retriever_chain

def get_conversational_rag_chain(retriever_chain): 
    
     llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.2)
     # llm = huggingface_hub.HuggingFaceHub(
          # huggingfacehub_api_token=HF_API_KEY,
          # task='text-generation'
     # )
     prompt = ChatPromptTemplate.from_messages([
          ("system", '''
                    Instructions:
                    1. Answer based on context given below.
                    2. Focus on repo/code.
                    3. Consider:
                         a. Purpose/features - describe.
                         b. Functions/code - provide details/samples.
                         c. Setup/usage - give instructions.
                    Context: {context}
               '''),
          MessagesPlaceholder(variable_name="chat_history"),
          ("human", "{input}")
     ])
     
     stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
     return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, vector_store):

     retriever_chain = get_context_retriever_chain(vector_store)
     conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
     response = conversation_rag_chain.invoke({
          "chat_history": st.session_state.chat_history,
          "input": user_input
     })


     # simple RAG without history
     # llm = ChatOpenAI(model='gpt-3.5-turbo')
     # # llm = huggingface_hub.HuggingFaceHub(
     # #           huggingfacehub_api_token=HF_API_KEY,
     # #           task='text-generation'
     # #      )
     # retriever = st.session_state.vector_store.as_retriever()
     # prompt = hub.pull("rlm/rag-prompt")
     # rag_chain = (
     #      {'context': retriever, 'question': RunnablePassthrough()}
     #      | prompt
     #      | llm
     #      | StrOutputParser()
     # )
     # response = rag_chain.invoke(user_input)
     return response

     
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
          vector_store = get_vectorstore_from_text(git_url)
          # st.write(vector_store)

          # user input
          user_query = st.chat_input("Type your message here...")
          if user_query is not None and user_query != "":
               response = get_response(user_query, vector_store)
               # st.write(st.session_state.chat_history)
               st.session_state.chat_history.append(HumanMessage(content=user_query))
               st.session_state.chat_history.append(AIMessage(content=response['answer']))
               # st.write(st.session_state)
          

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



































