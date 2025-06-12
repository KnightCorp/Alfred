# 1.1 Base Reasoning System

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM

import os
from dotenv import load_dotenv
import time
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

def run_prompt_loop(llm, model_name="LLM"):
    """Common prompt loop logic for all LLMs."""
    print(Fore.YELLOW + f"\n[{model_name}] Type your prompt (type 'Q' to quit):\n")

    while True:
        try:
            text = input(Fore.CYAN + "Ask anything... ").strip()

            if text.lower() == 'q':
                print(Fore.GREEN + "Exiting...")
                break

            if not text:
                print(Fore.RED + "Please enter a valid input.")
                continue

            print(Fore.BLUE + "Sending prompt...")
            start_time = time.time()

            response = llm.invoke(text)
            elapsed = time.time() - start_time

            print(Fore.MAGENTA + f"\nResponse received in {elapsed:.2f} seconds:\n")
            print(Fore.WHITE + str(response))

        except KeyboardInterrupt:
            print(Fore.RED + "\nKeyboard interrupt received. Exiting...")
            break
        except Exception as e:
            print(Fore.RED + f"An error occurred: {e}")


def Open_ai():
    """Initialize and interact with OpenAI LLM (gpt-3.5-turbo)."""
    print(Fore.YELLOW + "Loading OpenAI model...")
    t1 = time.time()

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo"
    )

    print(Fore.GREEN + f"OpenAI model loaded in {time.time() - t1:.2f} seconds.")
    run_prompt_loop(llm, model_name="OpenAI (gpt-3.5-turbo)")


def anthropic_ai():
    """Initialize and interact with Anthropic Claude model."""
    print(Fore.YELLOW + "Loading Claude model...")
    t1 = time.time()

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    print(Fore.GREEN + f"Anthropic Claude model loaded in {time.time() - t1:.2f} seconds.")
    run_prompt_loop(llm, model_name="Claude (3.5 Sonnet)")


def deepseek_r1():
    """Initialize and interact with DeepSeek model via Ollama."""
    print(Fore.YELLOW + "Loading DeepSeek model... (might take time on first run)")
    t1 = time.time()

    llm = OllamaLLM(
        model="deepseek-llm",
        temperature=0.7,
        top_p=0.9,
        num_predict=300
    )

    print(Fore.GREEN + f"DeepSeek model loaded in {time.time() - t1:.2f} seconds.")
    run_prompt_loop(llm, model_name="DeepSeek LLM")


def main():
    deepseek_r1()
    # Uncomment the one you want to run:
    # Open_ai()
    # anthropic_ai()

if __name__ == "__main__":
    main()
