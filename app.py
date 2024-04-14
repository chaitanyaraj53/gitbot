from langchain_community.embeddings import huggingface
from langchain_community.document_loaders import TextLoader, DirectoryLoader
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
from git import Repo
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
def get_vectorstore_from_text(repo_name):
     # pythonloader
     document = TextLoader(f"{repo_name}_analysis.txt").load()
     # document = DirectoryLoader('./sport', glob="**/*.txt").load()
     # loader = WebBaseLoader(url)
     # document = loader.load()
     text_splitter = RecursiveCharacterTextSplitter()
     document_chunks = text_splitter.split_documents(document)
     # embeddings = OpenAIEmbeddings()
     embeddings = huggingface.HuggingFaceEmbeddings()
     st.session_state.vector_store = chroma.Chroma()
     # for x in range(len(vector_store)):
          # vector_store.delete(ids=[x])
     # print(vector_store.delete_collection())
     st.session_state.vector_store = chroma.Chroma.from_documents(document_chunks, embeddings)
     # return st.session_state.vector_store

def get_context_retriever_chain(vector_store):
     # llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
     llm = huggingface_hub.HuggingFaceHub(
          huggingfacehub_api_token=os.getenv('HF_API_KEY'),
          task='text-generation'
          # repo_id='CohereForAI/c4ai-command-r-plus'
     )
     retriever = vector_store.as_retriever()
     
     prompt = ChatPromptTemplate.from_messages([
          MessagesPlaceholder(variable_name="chat_history"),
          ("user", "{input}"),
          ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
     ])
     
     retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
     return retriever_chain

def get_conversational_rag_chain(retriever_chain): 
    
     # llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
     llm = huggingface_hub.HuggingFaceHub(
          huggingfacehub_api_token=os.getenv('HF_API_KEY'),
          task='text-generation'
          # repo_id='CohereForAI/c4ai-command-r-plus'
     )
     prompt = ChatPromptTemplate.from_messages([
          ("system", "Answer the user's questions based on the below context:\n\n{context}"),
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



def extract_readme(repo_folder):
     readme_content = None
     readme_path = os.path.join(repo_folder, "README.md")
     if os.path.exists(readme_path):
          try:
               with open(readme_path, "r", encoding="utf-8") as f:
                    readme_content = f.read()
          except Exception as e:
               print(f"Error reading README file: {e}")
     return readme_content

def extract_code_files(repo_folder):
     code_files = []
     for root, dirs, files in os.walk(repo_folder):
          # Filter out the .git folder
          if '.git' in dirs:
               dirs.remove('.git')
          for file in files:
               if file.endswith((".txt", ".py", ".md")):
                    file_path = os.path.join(root, file)
                    try:
                         with open(file_path, "r", encoding="utf-8") as f:
                              code_files.append(f.read())
                    except Exception as e:
                         print(f"Error reading file {file_path}: {e}")
     return code_files

def preprocess_code(code):
     cleaned_code = re.sub(r'#.*?\n', '\n', code)
     cleaned_code = re.sub(r'""".*?"""', '', cleaned_code, flags=re.DOTALL)
     return cleaned_code

def analyze_code(code):
     function_names = []
     try:
          parsed_ast = ast.parse(code)
          for node in ast.walk(parsed_ast):
               if isinstance(node, ast.FunctionDef):
                    function_names.append(node.name)
     except SyntaxError:
          print("Syntax error occurred while parsing code.")
     return len(function_names), function_names

def analyze_code_files(code_files):
     total_functions = 0
     all_function_names = []

     for code in code_files:
          num_functions, function_names = analyze_code(code)
          total_functions += num_functions
          all_function_names.extend(function_names)

     return total_functions, all_function_names

def save_repo_analysis(repo_url, repo_name):
     try:
          # Clone the repository
          repo_folder = "temp_repo"
          Repo.clone_from(repo_url, repo_folder)

          # Extract README content
          readme_content = extract_readme(repo_folder)

          # Extract code files
          code_files = extract_code_files(repo_folder)
          preprocessed_code_files = [preprocess_code(code) for code in code_files]

          # Analyze code files
          total_functions, all_function_names = analyze_code_files(preprocessed_code_files)

          # Generate text for analysis
          analysis_text = f"README CONTENT:\n{readme_content}\n\n"
          analysis_text += f"Total functions: {total_functions}\n"
          analysis_text += f"Function names: {', '.join(all_function_names)}\n"

          # Write analysis text to file
          output_file = f"{repo_name}_analysis.txt"
          with open(output_file, "w", encoding="utf-8") as f:
               f.write(analysis_text)

          # Remove temporary repository folder
          if os.name == "nt":  # Windows
               os.system(f"rmdir /s /q {repo_folder}")
          else:  # Unix-like (Linux, macOS)
               os.system(f"rm -rf {repo_folder}")

          print(f"Repository analysis saved to '{output_file}'")
     except Exception as e:
          print(f"Error saving repository analysis: {e}")
# app config
def extract_repo_name(repo_url):
    parsed_url = urlparse(repo_url)
    repo_name = parsed_url.path.strip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]  # Remove the .git suffix
    return repo_name

def main():
     # st.session_state.clear(
     __import__('pysqlite3')
     sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
     st.set_page_config(page_title="Chat", layout='wide')
     st.title("GitHub Bot")
     with st.sidebar:
          st.header("Enter URL")
          git_url = st.text_input("")

     if git_url is None or git_url == "":
          st.info("Please enter a website URL")

     else:
          # session state
          # if 'messages' not in st.session_state:
          #      st.session_state.messages = []

          # for message in st.session_state.messages:
          #      st.chat_message(message['role']).markdown(message['content'])

          if "chat_history" not in st.session_state:
               st.session_state.chat_history = [
                    AIMessage(content="Ask me about the repository..."),
               ]
          if "vector_store" not in st.session_state:
               repo_name = extract_repo_name(git_url)
               save_repo_analysis(git_url, repo_name)
               get_vectorstore_from_text(repo_name)
          if 'git_url' not in st.session_state:
               st.session_state.git_url = git_url
               st.write(st.session_state)

          # user input
          user_query = st.chat_input("Type your message here...")
          if user_query is not None and user_query != "":
               response = get_response(user_query)
               st.session_state.chat_history.append(HumanMessage(content=user_query))
               st.session_state.chat_history.append(AIMessage(content=response))
               # st.session_state.messages.append({'role':'user', 'content': user_query})
          

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