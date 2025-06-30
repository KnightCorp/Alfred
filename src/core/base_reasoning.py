import os
import time
import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
import chromadb

import asyncio
from typing import TypedDict, Annotated, Sequence
from functools import lru_cache
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from langchain_ollama import OllamaLLM
from colorama import Fore, init

from chromadb_utils import create_inmemory_collection, add_file_to_collection, query_collection

init(autoreset=True)
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], list]

def select_model():
    print(Fore.YELLOW + "Select a model:")
    print("1. OpenAI GPT-3.5/4")
    print("2. Anthropic Claude")
    if GEMINI_AVAILABLE:
        print("3. Google Gemini")
    print("4. DeepSeek")
    print("5. Ollama (local)")
    choice = input("Enter choice (1/2/3/4/5): ").strip()
    if choice == "1":
        return "openai"
    elif choice == "2":
        return "anthropic"
    elif choice == "3" and GEMINI_AVAILABLE:
        return "gemini"
    elif choice == "4":
        return "deepseek"
    elif choice == "5":
        return "ollama"
    else:
        print(Fore.RED + "Invalid choice, defaulting to OpenAI.")
        return "openai"

@lru_cache(maxsize=5)
def get_llm(model_name: str = "openai"):
    if model_name == "openai":
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        )
    elif model_name == "anthropic":
        return ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        )
    elif model_name == "gemini" and GEMINI_AVAILABLE:
        return ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-1.5-pro-latest"
        )
    elif model_name == "deepseek":
        return ChatDeepSeek(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-coder")
        )
    elif model_name == "ollama":
        return OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "deepseek-llm"),
            temperature=0.7,
            top_p=0.9,
            num_predict=300
        )
    else:
        raise ValueError(f"Unsupported or unavailable model: {model_name}")

def main_menu():
    print(Fore.CYAN + "\nWhat would you like to do?")
    print("1. Ask anything (chat mode)")
    print("2. Upload file(s) & ask about them")
    print("Q. Quit")
    choice = input("Enter choice (1/2/Q): ").strip().lower()
    return choice

def ask_anything(llm):
    print(Fore.YELLOW + "\n[Chat Mode] Type your prompt (type 'Q' to quit):\n")
    while True:
        text = input(Fore.CYAN + "Ask anything... ").strip()
        if text.lower() == 'q':
            print(Fore.GREEN + "Exiting chat mode...")
            break
        print(Fore.BLUE + "Sending prompt...")
        start_time = time.time()
        response = llm.invoke([HumanMessage(content=text)])
        elapsed = time.time() - start_time
        print(Fore.MAGENTA + f"\nResponse received in {elapsed:.2f} seconds:\n")
        print(Fore.WHITE + getattr(response, "content", str(response)))

def ask_about_files(llm):
    print(Fore.YELLOW + "\n[File Q&A Mode] You can upload multiple files (plain text).")
    collection = create_inmemory_collection()
    file_counter = 0
    while True:
        file_path = input(Fore.CYAN + "Enter file path to upload (or 'done' to finish): ").strip()
        if file_path.lower() == 'done':
            break
        if add_file_to_collection(collection, file_path, f"user_file_{file_counter}"):
            file_counter += 1
    if file_counter == 0:
        print(Fore.RED + "No files uploaded. Returning to main menu.")
        return
    print(Fore.GREEN + f"{file_counter} file(s) uploaded. You can now ask questions about them.")
    while True:
        query = input(Fore.CYAN + "Ask a question about your files (or 'q' to quit): ").strip()
        if query.lower() == 'q':
            print(Fore.GREEN + "Exiting file Q&A mode...")
            break
        context = query_collection(collection, query, n_results=3)
        if not context:
            print(Fore.RED + "No relevant content found in uploaded files.")
            continue
        prompt = (
            "You are a helpful, detailed assistant. "
            "Given the following context from uploaded files, answer the user's question in a detailed, conversational way. "
            "If possible, explain your reasoning and provide relevant background from the files.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Detailed Answer:"
        )   

        print(Fore.BLUE + "Sending prompt to LLM...")
        start_time = time.time()
        response = llm.invoke([HumanMessage(content=prompt)])
        elapsed = time.time() - start_time
        print(Fore.MAGENTA + f"\nResponse received in {elapsed:.2f} seconds:\n")
        print(Fore.WHITE + getattr(response, "content", str(response)))

def main():
    print(Fore.GREEN + "Welcome to the Multi-LLM Assistant with File Q&A!")
    model_name = select_model()
    llm = get_llm(model_name)
    print(Fore.GREEN + f"Agent initialized with model: {model_name}")
    while True:
        choice = main_menu()
        if choice == "1":
            ask_anything(llm)
        elif choice == "2":
            ask_about_files(llm)
        elif choice == "q":
            print(Fore.GREEN + "Goodbye!")
            break
        else:
            print(Fore.RED + "Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
