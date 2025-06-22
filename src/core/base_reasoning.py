import os
import getpass
import time
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from functools import lru_cache
from tools import tools, should_use_deep_research

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
from langgraph.prebuilt import create_react_agent
from colorama import Fore, init

# Fix for nested event loops (Jupyter, etc.)
import nest_asyncio
nest_asyncio.apply()

init(autoreset=True)
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], list]

def select_model():
    print(Fore.YELLOW + "Select a model:")
    print("1. OpenAI GPT-4.1 (Token Limit: 30,000 tokens/min)")
    print("2. Anthropic Claude (Token Limit: see Anthropic docs)")
    if GEMINI_AVAILABLE:
        print("3. Google Gemini (Token Limit: see Google docs)")
    print("4. DeepSeek (Token Limit: see DeepSeek docs)")
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

def create_agent(model_name):
    llm = get_llm(model_name)
    if model_name in {"openai", "anthropic", "gemini", "deepseek"}:
        print(Fore.CYAN + f"[DEBUG] Creating ReAct agent for {model_name} with tools: {[t.name for t in tools]}")
        agent = create_react_agent(llm, tools)
        return agent
    elif model_name == "ollama":
        def ollama_chat_loop():
            print(Fore.YELLOW + "\n[Ollama Chat] Type your prompt (type 'Q' to quit):\n")
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
                    response = llm.invoke([HumanMessage(content=text)])
                    elapsed = time.time() - start_time
                    print(Fore.MAGENTA + f"\nResponse received in {elapsed:.2f} seconds:\n")
                    print(Fore.WHITE + getattr(response, "content", str(response)))
                except KeyboardInterrupt:
                    print(Fore.RED + "\nKeyboard interrupt received. Exiting...")
                    break
                except Exception as e:
                    print(Fore.RED + f"An error occurred: {e}")
        return ollama_chat_loop
    else:
        raise ValueError(f"Unsupported model: {model_name}")

async def run_prompt_loop(agent, model_name="LLM"):
    print(Fore.YELLOW + f"\n[{model_name}] Type your prompt (type 'Q' to quit):\n")
    while True:
        try:
            text = input(Fore.CYAN + "Ask anything... ").strip()
            if text.lower() == 'q':
                print(Fore.GREEN + "Exiting...")
                break

        

            print(Fore.BLUE + "Sending prompt...")
            start_time = time.time()
            initial_state = {"messages": [HumanMessage(content=text)]}
            print(Fore.LIGHTBLACK_EX + f"[DEBUG] Initial state: {initial_state}")
            try:
                result = await agent.ainvoke(initial_state)
            except Exception as e:
                err_str = str(e)
                if "rate_limit_exceeded" in err_str or "429" in err_str or "Request too large" in err_str:
                    print(Fore.RED + f"\n[ERROR] Rate limit or token limit exceeded for this model.")
                    if "gpt-4.1" in err_str or "openai" in model_name.lower():
                        print(Fore.YELLOW + "OpenAI GPT-4.1 has a 30,000 tokens/minute limit. Try a shorter or simpler query, or wait and try again.")
                    else:
                        print(Fore.YELLOW + f"Model '{model_name}' may have hit its token or rate limit. Please check provider docs.")
                    print(Fore.LIGHTBLACK_EX + f"Raw error: {e}")
                    continue
                else:
                    print(Fore.RED + f"An error occurred: {e}")
                    break
            elapsed = time.time() - start_time

            last_message = result["messages"][-1]
            tool_used = False
            if hasattr(last_message, "tool_calls"):
                tool_used = bool(getattr(last_message, "tool_calls", []))
            """
            print(Fore.LIGHTBLACK_EX + f"[DEBUG] Tool used: {tool_used}")
            print(Fore.LIGHTBLACK_EX + f"[DEBUG] Final state: {result}")
            """
            
            print(Fore.MAGENTA + f"\nResponse received in {elapsed:.2f} seconds:\n")
            print(Fore.WHITE + getattr(last_message, "content", str(last_message)))
        except KeyboardInterrupt:
            print(Fore.RED + "\nKeyboard interrupt received. Exiting...")
            break
        except Exception as e:
            print(Fore.RED + f"An error occurred: {e}")

def main():
    print(Fore.GREEN + "Initializing LangGraph ReAct Agent...")
    model_name = select_model()
    agent = create_agent(model_name)
    print(Fore.GREEN + f"Agent initialized with model: {model_name}")
    if model_name == "ollama":
        agent()  # Start Ollama's simple chat loop
    else:
        asyncio.run(run_prompt_loop(agent, model_name=f"ReAct Agent ({model_name})"))

if __name__ == "__main__":
    main()
