"""
Base Reasoning Module for Alfred + Morphic + AgenticSeek Integration
==================================================================

This module provides the core reasoning capabilities that will be shared across
the three integrated systems:
1. Alfred - Multi-LLM assistant with web search and file Q&A
2. Morphic - AI-powered search engine with generative UI
3. AgenticSeek - Fully local AI agent with autonomous capabilities

Key Features:
- Unified reasoning interface for all three systems
- Agent type detection and routing
- Context management and memory
- Tool orchestration and execution
- Error handling and fallback mechanisms
- **AGGRESSIVE SPEED OPTIMIZATION**: Tuned parameters and model prioritization for faster responses.
- **ENHANCED RESEARCH DEPTH & QUALITY**: Prompts crafted for comprehensive, structured, and evidence-based responses.
- **CONVERSATIONAL MEMORY**: Integrates long-term memory from history.py for persistent context.
- **IMPROVED CONTEXT DETECTION**: Leverages memory for automatic follow-up query detection.
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import concurrent.futures
from threading import Lock
import uuid # For session ID generation

# Core AI and ML imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# Model providers
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from langchain_ollama import OllamaLLM

# Utilities
from colorama import Fore, Style, init
from dotenv import load_dotenv
import chromadb
from tavily import TavilyClient

# Import history module components
from history import ConversationMemory, add_to_history # Import the enhanced history components

# Initialize colorama
init(autoreset=True)

# Load environment variables early for logging configuration that might depend on them
load_dotenv()

# Configure logging: Suppress noisy logs globally and for specific libraries
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s') # Set global to WARNING

# Suppress specific chatty loggers aggressively
logging.getLogger("httpx").setLevel(logging.ERROR) # Only show errors from httpx
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING) # Suppress all openai module logs (including _base_client)
logging.getLogger("langchain_core").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)
logging.getLogger("langchain_anthropic").setLevel(logging.WARNING)
logging.getLogger("langchain_google_genai").setLevel(logging.WARNING)
logging.getLogger("langchain_ollama").setLevel(logging.WARNING)
logging.getLogger("langchain_deepseek").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ============================
# Performance Optimization Constants
# ============================

# AGGRESSIVE SPEED OPTIMIZATION: Tuned parameters for faster responses
MAX_SEARCH_RESULTS = 2  # Reduced to 2 for quicker search results, fewer tokens
MAX_SEARCH_CONTENT_LENGTH = 700  # Further truncated for speed
MODEL_TIMEOUT = 15  # Stricter timeout for faster failures (was 45s)
EMBEDDING_RETRIEVAL_RESULTS = 2 # Reduced for faster memory context loading
PARALLEL_SEARCH_ENABLED = True

# ============================
# Core Data Structures
# ============================



class AgentType(Enum):
    """Types of agents available in the integrated system"""
    ALFRED = "alfred"           # Original Alfred chat assistant
    MORPHIC = "morphic"         # Morphic search engine
    AGENTIC_SEEK = "agentic_seek"  # AgenticSeek autonomous agent
    HYBRID = "hybrid"           # Combined capabilities
    AUTO = "auto"               # Automatically select best agent

class TaskType(Enum):
    """Types of tasks that can be performed"""
    CHAT = "chat"
    SEARCH = "search"
    RESEARCH = "research" # Explicitly define RESEARCH task type
    CODE = "code"
    FILE_ANALYSIS = "file_analysis"
    WEB_BROWSE = "web_browse"
    AUTONOMOUS = "autonomous"

class ToolType(Enum):
    """Available tools in the system"""
    WEB_SEARCH = "web_search"
    FILE_READER = "file_reader"
    CODE_EXECUTOR = "code_executor"
    BROWSER = "browser"
    MEMORY = "memory"
    GENERATOR = "generator"

@dataclass
class ReasoningContext:
    """Context object that maintains state across reasoning steps"""
    user_input: str
    task_type: TaskType
    agent_type: AgentType
    conversation_history: List[BaseMessage] = field(default_factory=list) # Short-term history
    tools_used: List[str] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())) # Unique session ID

@dataclass
class AgentCapabilities:
    """Defines what each agent can do"""
    name: str

    agent_type: AgentType
    supported_tasks: List[TaskType]
    available_tools: List[ToolType]
    model_preferences: List[str]
    local_only: bool = False
    requires_api: bool = True

class ReasoningResult(TypedDict):
    """Result of a reasoning operation"""
    success: bool
    result: Any
    agent_used: AgentType
    tools_used: List[str]
    execution_time: float
    metadata: Dict[str, Any]
    error: Optional[str]

# ============================
# Agent Definitions
# ============================

AGENT_DEFINITIONS = {
    AgentType.ALFRED: AgentCapabilities(
        name="Alfred Assistant",
        agent_type=AgentType.ALFRED,
        supported_tasks=[TaskType.CHAT, TaskType.SEARCH, TaskType.RESEARCH, TaskType.FILE_ANALYSIS],
        available_tools=[ToolType.WEB_SEARCH, ToolType.FILE_READER, ToolType.MEMORY],
        model_preferences=["openai", "anthropic", "deepseek"], # For Alfred, prioritize common powerful models
        requires_api=True
    ),
    AgentType.MORPHIC: AgentCapabilities(
        name="Morphic Search",
        agent_type=AgentType.MORPHIC,
        supported_tasks=[TaskType.SEARCH, TaskType.RESEARCH],
        available_tools=[ToolType.WEB_SEARCH, ToolType.GENERATOR],
        model_preferences=["openai", "anthropic"], # For Morphic, prioritize common powerful models
        requires_api=True
    ),
    AgentType.AGENTIC_SEEK: AgentCapabilities(
        name="AgenticSeek Autonomous",
        agent_type=AgentType.AGENTIC_SEEK,
        supported_tasks=[TaskType.AUTONOMOUS, TaskType.CODE, TaskType.WEB_BROWSE, TaskType.RESEARCH, TaskType.SEARCH],
        available_tools=[ToolType.BROWSER, ToolType.CODE_EXECUTOR, ToolType.WEB_SEARCH],
        model_preferences=["ollama", "deepseek"], # For AgenticSeek, prioritize local/code-focused models
        local_only=True,
        requires_api=False
    )
}

# ============================
# Model Cache and Pool Management
# ============================

class ModelCache:
    """
    SPEED OPTIMIZATION: Singleton model cache to avoid repeated model initialization
    This prevents expensive model loading on every request
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models = {}
                    cls._instance._initialized = False
        return cls._instance
    
    def get_model(self, model_name: str, config: Dict[str, Any] = None):
        """Get cached model or create new one"""
        if model_name not in self._models:
            self._models[model_name] = self._create_model(model_name, config or {})
        return self._models[model_name]
    
    def _create_model(self, model_name: str, config: Dict[str, Any]):
        """Create model with optimized settings for speed and research depth"""
        try:
            # AGGRESSIVE SPEED OPTIMIZATION: Use faster model variants and reduced tokens for general chat
            # Max_tokens will be higher for specific RESEARCH tasks
            if model_name == "openai" and os.getenv("OPENAI_API_KEY"):
                return ChatOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"), # Prioritize gpt-3.5-turbo for speed
                    temperature=0.3,  # Lower temperature for more direct, concise answers
                    max_tokens=800, # Reduced for faster responses in general chat
                    timeout=MODEL_TIMEOUT,
                    max_retries=0 # No retries for max speed
                )
            
            elif model_name == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
                return ChatAnthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                    model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"), # Prioritize Haiku for speed
                    temperature=0.3,
                    max_tokens=800, # Reduced for faster responses in general chat
                    timeout=MODEL_TIMEOUT,
                    max_retries=0
                )
            
            elif model_name == "deepseek" and os.getenv("DEEPSEEK_API_KEY"):
                return ChatDeepSeek(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    model=os.getenv("DEEPSEEK_MODEL", "deepseek-coder"),
                    temperature=0.3,
                    max_tokens=800,
                    timeout=MODEL_TIMEOUT,
                    max_retries=0
                )
            
            elif model_name == "gemini" and GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
                return ChatGoogleGenerativeAI(
                    google_api_key=os.getenv("GEMINI_API_KEY"),
                    model="gemini-1.5-flash-latest", # Prioritize Flash for speed
                    temperature=0.3,
                    max_output_tokens=800,
                    timeout=MODEL_TIMEOUT
                )
            
            elif model_name == "ollama":
                return OllamaLLM(
                    model=os.getenv("OLLAMA_MODEL", "llama3"), # Faster Ollama models
                    temperature=0.3,
                    timeout=MODEL_TIMEOUT,
                    num_predict=800 # Reduced for faster responses
                )
            
            else:
                raise ValueError(f"Unknown model: {model_name}")
                
        except Exception as e:
            logger.error(f"Error creating model {model_name}: {e}")
            raise

# Global model cache instance
_model_cache = ModelCache()

# ============================
# Core Reasoning Engine
# ============================

class BaseReasoningEngine:
    """
    Core reasoning engine that orchestrates between Alfred, Morphic, and AgenticSeek
    OPTIMIZED FOR SPEED with caching, async operations, and reduced processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.models = {}
        self.tools = {}
        self.agent_capabilities = AGENT_DEFINITIONS
        self.selected_model: Optional[str] = None # Added for selected model tracking
        self.selected_agent: Optional[AgentType] = None # Added for selected agent tracking
        self.session_id: str = str(uuid.uuid4()) # Unique session ID for persistent memory
        
        # Initialize ConversationMemory (long-term memory)
        # Pass "OPEN_API_KEY" as specified by the user for embedding generation
        self.long_term_memory = ConversationMemory(
            openai_api_key=os.getenv("OPEN_API_KEY") 
        )
        logger.info(f"Initialized ConversationMemory with session ID: {self.session_id}")

        # SPEED OPTIMIZATION: Use thread pool for parallel operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        
        # SPEED OPTIMIZATION: Initialize models asynchronously
        asyncio.create_task(self._initialize_models_async())
        self._initialize_tools()
        
        # SPEED OPTIMIZATION: Cache for frequent agent/task determinations
        self._agent_cache = {}
        self._task_cache = {}

        # User feedback loop setup (conceptual, requires external system)
        self.feedback_data = [] # To store feedback like {'query': '...', 'response': '...', 'rating': '...'}
        
    async def _initialize_models_async(self):
        """
        SPEED OPTIMIZATION: Asynchronous model initialization to prevent blocking
        """
        try:
            # Initialize models in parallel using thread pool
            model_configs = [
                ("openai", {}),
                ("anthropic", {}),
                ("deepseek", {}),
                ("gemini", {}),
                ("ollama", {})
            ]
            
            # Run model initialization in parallel
            tasks = []
            for model_name, config in model_configs:
                task = asyncio.create_task(self._init_single_model(model_name, config))
                tasks.append(task)
            
            # Wait for all models to initialize with timeout
            try:
                # Reduced timeout for faster startup if models are slow
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=15) 
            except asyncio.TimeoutError:
                logger.warning("Model initialization timed out, continuing with available models")
            
        except Exception as e:
            logger.error(f"Error in async model initialization: {e}")
    
    async def _init_single_model(self, model_name: str, config: Dict[str, Any]):
        """Initialize a single model asynchronously"""
        try:
            # Run model creation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self.executor,
                _model_cache.get_model,
                model_name,
                config
            )
            self.models[model_name] = model
            logger.info(f"Model {model_name} initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize model {model_name}: {e}")
    
    def _initialize_tools(self):
        """Initialize available tools with optimized settings"""
        # SPEED OPTIMIZATION: Web search tool with reduced result limits
        if os.getenv("TAVILY_API_KEY"):
            self.tools["web_search"] = TavilyClient(
                api_key=os.getenv("TAVILY_API_KEY"),
                #timeout=10  # Reduced timeout for faster searches
            )

        # The 'memory' tool is now explicitly the ConversationMemory instance
        self.tools["memory"] = self.long_term_memory
    
    @lru_cache(maxsize=128)
    def determine_agent_type(self, user_input: str, context_hash: Optional[str] = None) -> AgentType:
        """
        SPEED OPTIMIZATION: Cached agent type determination with simplified logic
        """
        input_lower = user_input.lower()
        
        # Objective 1: Auto mode agent selection logic
        if self.selected_agent == AgentType.AUTO:
            # Prioritize agents based on model availability and perceived speed for different tasks.
            # For general chat/quick answers, favor models like gpt-3.5-turbo, Haiku, Flash.
            # For deeper research, still allow gpt-4/Sonnet if explicitly requested or if no faster option can fulfill.

            # Check for faster API-based models (e.g., gpt-3.5-turbo, Haiku, Flash)
            if any(m in self.models for m in ["openai", "anthropic", "gemini"]):
                # For quick questions, default to Alfred
                if not any(word in input_lower for word in ["research", "explain in detail", "comprehensive analysis", "scholarly", "history", "future implications", "code", "debug", "program", "autonomous"]):
                     return AgentType.ALFRED
                
                # If a specific task is implied, route accordingly
                if any(word in input_lower for word in ["code", "debug", "program", "autonomous"]):
                    return AgentType.AGENTIC_SEEK
                elif any(word in input_lower for word in ["search", "find", "visual", "interface"]):
                    return AgentType.MORPHIC
                else: # For research or complex queries, still Alfred
                    return AgentType.ALFRED
            # Fallback to AgenticSeek if local models (Ollama) are available
            elif "ollama" in self.models:
                return AgentType.AGENTIC_SEEK
            else:
                # Default to Alfred if no specific preference or local model is viable
                return AgentType.ALFRED
        
        # If not Auto mode, use direct pattern matching or predefined selection
        elif self.selected_agent:
            return self.selected_agent
        
        # Fallback to pattern matching if selected_agent is not set (shouldn't happen with new flow)
        if any(word in input_lower for word in ["code", "debug", "program", "autonomous"]):
            return AgentType.AGENTIC_SEEK
        elif any(word in input_lower for word in ["search", "find", "visual", "interface"]):
            return AgentType.MORPHIC
        else:
            return AgentType.ALFRED
    
    @lru_cache(maxsize=128)
    def determine_task_type(self, user_input: str, agent_type: AgentType) -> TaskType:
        """SPEED OPTIMIZATION: Cached task type determination. Enhanced for RESEARCH."""
        input_lower = user_input.lower()
        
        # SPEED OPTIMIZATION: Fast pattern matching with early returns
        if any(word in input_lower for word in ["research", "explain in detail", "comprehensive analysis", "scholarly", "history", "future implications"]):
            return TaskType.RESEARCH
        elif any(word in input_lower for word in ["code", "program", "debug"]):
            return TaskType.CODE
        elif any(word in input_lower for word in ["search", "find"]):
            return TaskType.SEARCH
        elif any(word in input_lower for word in ["file", "document", "analyze"]):
            return TaskType.FILE_ANALYSIS
        elif any(word in input_lower for word in ["browse", "visit", "scrape"]):
            return TaskType.WEB_BROWSE
        elif any(word in input_lower for word in ["autonomous", "automatic"]):
            return TaskType.AUTONOMOUS
        else:
            return TaskType.CHAT
    
    def select_optimal_model(self, agent_type: AgentType, task_type: TaskType) -> str:
        """
        AGGRESSIVE SPEED OPTIMIZATION: Fast model selection with availability checking.
        Prioritizes self.selected_model if set. Otherwise, prioritizes speed.
        """
        if self.selected_model and self.selected_model in self.models:
            return self.selected_model

        preferences = self.agent_capabilities[agent_type].model_preferences
        available_models = list(self.models.keys())
        
        # Prioritize local models if agent is local-only
        if self.agent_capabilities[agent_type].local_only and "ollama" in available_models:
            return "ollama"
        
        # Aggressive prioritization for speed:
        # 1. Ollama (if available, and if not local_only agent, could still be fast)
        # 2. Gemini Flash (very fast)
        # 3. OpenAI (gpt-3.5-turbo)
        # 4. Anthropic (Haiku)
        # 5. DeepSeek (if available)
        # 6. Other (e.g., gpt-4, Sonnet, Gemini Pro if explicitly configured as default)

        # Iterate through a speed-prioritized list of models, checking availability
        speed_priority_order = ["ollama", "gemini", "openai", "anthropic", "deepseek"]
        
        for model_name in speed_priority_order:
            if model_name in available_models and model_name in preferences:
                # If model is available AND preferred by the agent, pick it.
                return model_name
            
        # Fallback to any available model if no preferred speed-optimized model is found
        # This part ensures a model is always picked if self.models is not empty
        for model_name in speed_priority_order:
            if model_name in available_models:
                return model_name

        return available_models[0] if available_models else "ollama" # Final fallback


    async def reason(self, user_input: str, short_term_history: List[BaseMessage]) -> ReasoningResult:
        """
        Main reasoning method with parallel processing and caching, integrating long-term memory.
        """
        start_time = time.time()
        
        try:
            # Determine agent type based on user's selection or auto-logic
            agent_type_to_use = self.determine_agent_type(user_input)
            task_type = self.determine_task_type(user_input, agent_type_to_use)
            
            context = ReasoningContext(
                user_input=user_input,
                task_type=task_type,
                agent_type=agent_type_to_use,
                conversation_history=short_term_history, # Pass existing short-term history
                session_id=self.session_id # Pass current session ID
            )
            
            # Retrieve relevant long-term memories
            # Reduced n_results for speed, and max_context_length for token efficiency
            long_term_context_str = self.long_term_memory.get_conversation_context(
                query=user_input,
                max_context_length=1000, # Reduced max context length to save tokens/speed
                n_results=EMBEDDING_RETRIEVAL_RESULTS, # Reduced number of retrieved memories
                session_id=self.session_id
            )
            
            # Prepare the full prompt by combining user input and long-term context
            # The system message inside _invoke_model will further guide the response.
            full_prompt_for_llm = user_input
            if long_term_context_str:
                # Add a clear separator and instruction for the LLM to integrate context
                full_prompt_for_llm = (
                    f"Consider the following relevant past interactions from this session:\n"
                    f"{long_term_context_str}\n\n"
                    f"Now, respond to the user's current query, seamlessly building upon the conversation if applicable: {user_input}"
                )

            # Select the optimal model
            model_name = self.select_optimal_model(context.agent_type, context.task_type)
            model = self.models.get(model_name)
            if model is None:
                await asyncio.sleep(0.1) # brief wait
                model = self.models.get(model_name)
                if model is None:
                    model = next(iter(self.models.values())) if self.models else None
                    if model is None:
                        raise ValueError("No models available")
            
            # Route to agent with stricter timeout
            raw_llm_response = await asyncio.wait_for(
                self._route_to_agent(context, model, full_prompt_for_llm),
                timeout=MODEL_TIMEOUT
            )
            
            execution_time = time.time() - start_time
            
            # After successful response, update both short-term and long-term memory
            self.long_term_memory.update_conversation_history(
                history=short_term_history, # Updates the passed list
                user_input=user_input,
                llm_reply=raw_llm_response,
                session_id=self.session_id
            )

            return ReasoningResult(
                success=True,
                result=raw_llm_response,
                agent_used=context.agent_type,
                tools_used=context.tools_used,
                execution_time=execution_time,
                metadata=context.metadata,
                error=None
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"Reasoning timed out after {MODEL_TIMEOUT} seconds.")
            
            return ReasoningResult(
                success=False,
                result="Request timed out",
                agent_used=context.agent_type if context else AgentType.ALFRED,
                tools_used=[],
                execution_time=execution_time,
                metadata={},
                error=f"Request timed out after {MODEL_TIMEOUT} seconds."
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Reasoning failed: {e}")
            
            return ReasoningResult(
                success=False,
                result=None,
                agent_used=context.agent_type if context else AgentType.ALFRED,
                tools_used=[],
                execution_time=execution_time,
                metadata={},
                error=str(e)
            )
            
    async def _route_to_agent(self, context: ReasoningContext, model, full_prompt_for_llm: str) -> str:
        """Route the request to the appropriate agent."""
        
        if context.agent_type == AgentType.ALFRED:
            return await self._handle_alfred_request(context, model, full_prompt_for_llm)
        elif context.agent_type == AgentType.MORPHIC:
            return await self._handle_morphic_request(context, model, full_prompt_for_llm)
        elif context.agent_type == AgentType.AGENTIC_SEEK:
            return await self._handle_agentic_seek_request(context, model, full_prompt_for_llm)
        else:
            raise ValueError(f"Unknown agent type: {context.agent_type}")
    
    async def _handle_alfred_request(self, context: ReasoningContext, model, full_prompt_for_llm: str) -> str:
        """Handle Alfred-style requests (chat, file analysis, basic search, and now RESEARCH)"""
        
        # Adjust max_tokens for specific task types that require more detail
        if context.task_type in [TaskType.RESEARCH, TaskType.FILE_ANALYSIS, TaskType.CODE, TaskType.AUTONOMOUS]:
            # Temporarily set higher max_tokens for detail-intensive tasks
            original_max_tokens = getattr(model, 'max_tokens', None) or getattr(model, 'max_output_tokens', None) or 800
            if hasattr(model, 'max_tokens'): model.max_tokens = 2000
            elif hasattr(model, 'max_output_tokens'): model.max_output_tokens = 2000
            elif hasattr(model, 'num_predict'): model.num_predict = 2000 # For Ollama
        else:
            original_max_tokens = None # No need to restore if not changed

        try:
            if context.task_type == TaskType.RESEARCH:
                search_task = asyncio.create_task(self._web_search(context.user_input, num_results=MAX_SEARCH_RESULTS))
                search_results = await search_task
                context.tools_used.append("web_search")
                
                # Use a more detailed prompt for research tasks, passing original user_input for clarity
                prompt_for_llm_with_search = self._build_research_prompt(context.user_input, search_results)
                response = await self._invoke_model(model, prompt_for_llm_with_search, context.conversation_history)
                return response

            elif context.task_type == TaskType.SEARCH:
                search_task = asyncio.create_task(self._web_search(context.user_input))
                search_results = await search_task
                context.tools_used.append("web_search")
                
                prompt_for_llm_with_search = self._build_search_prompt(context.user_input, search_results)
                response = await self._invoke_model(model, prompt_for_llm_with_search, context.conversation_history)
                return response
            
            elif context.task_type == TaskType.FILE_ANALYSIS:
                context.tools_used.append("file_reader")
                response = await self._invoke_model(model, full_prompt_for_llm, context.conversation_history)
                return response
            
            else: # Default chat
                response = await self._invoke_model(model, full_prompt_for_llm, context.conversation_history)
                return response
        finally:
            # Restore original max_tokens for the model
            if original_max_tokens is not None:
                if hasattr(model, 'max_tokens'): model.max_tokens = original_max_tokens
                elif hasattr(model, 'max_output_tokens'): model.max_output_tokens = original_max_tokens
                elif hasattr(model, 'num_predict'): model.num_predict = original_max_tokens # For Ollama


    async def _handle_morphic_request(self, context: ReasoningContext, model, full_prompt_for_llm: str) -> str:
        """Handle Morphic-style requests (search with generative UI, including RESEARCH aspects)"""
        
        # Adjust max_tokens for specific task types that require more detail
        if context.task_type == TaskType.RESEARCH:
            original_max_tokens = getattr(model, 'max_tokens', None) or getattr(model, 'max_output_tokens', None) or 800
            if hasattr(model, 'max_tokens'): model.max_tokens = 2000
            elif hasattr(model, 'max_output_tokens'): model.max_output_tokens = 2000
            elif hasattr(model, 'num_predict'): model.num_predict = 2000
        else:
            original_max_tokens = None

        try:
            if "web_search" in self.tools:
                search_results = await self._web_search(context.user_input, num_results=MAX_SEARCH_RESULTS)
                context.tools_used.append("web_search")
                context.tools_used.append("generator")
                
                prompt_for_llm_with_search = self._build_morphic_prompt(context.user_input, search_results)
                response = await self._invoke_model(model, prompt_for_llm_with_search, context.conversation_history)
                return response
            
            response = await self._invoke_model(model, full_prompt_for_llm, context.conversation_history)
            return response
        finally:
            if original_max_tokens is not None:
                if hasattr(model, 'max_tokens'): model.max_tokens = original_max_tokens
                elif hasattr(model, 'max_output_tokens'): model.max_output_tokens = original_max_tokens
                elif hasattr(model, 'num_predict'): model.num_predict = original_max_tokens


    async def _handle_agentic_seek_request(self, context: ReasoningContext, model, full_prompt_for_llm: str) -> str:
        """Handle AgenticSeek-style requests (autonomous, coding, complex Browse, including RESEARCH)"""
        
        # Adjust max_tokens for specific task types that require more detail
        if context.task_type in [TaskType.RESEARCH, TaskType.CODE, TaskType.AUTONOMOUS]:
            original_max_tokens = getattr(model, 'max_tokens', None) or getattr(model, 'max_output_tokens', None) or 800
            if hasattr(model, 'max_tokens'): model.max_tokens = 2000
            elif hasattr(model, 'max_output_tokens'): model.max_output_tokens = 2000
            elif hasattr(model, 'num_predict'): model.num_predict = 2000
        else:
            original_max_tokens = None

        try:
            if context.task_type == TaskType.CODE:
                context.tools_used.append("code_executor")
                prompt_for_llm = self._build_coding_prompt(context.user_input)
                
            elif context.task_type == TaskType.WEB_BROWSE:
                context.tools_used.append("browser")
                prompt_for_llm = self._build_Browse_prompt(context.user_input)
                
            elif context.task_type == TaskType.RESEARCH:
                search_task = asyncio.create_task(self._web_search(context.user_input, num_results=MAX_SEARCH_RESULTS))
                search_results = await search_task
                context.tools_used.append("web_search")
                prompt_for_llm = self._build_research_prompt(context.user_input, search_results)
                
            elif context.task_type == TaskType.AUTONOMOUS:
                context.tools_used.extend(["browser", "code_executor", "web_search"])
                prompt_for_llm = self._build_autonomous_prompt(context.user_input)
                
            else:
                prompt_for_llm = full_prompt_for_llm
            
            response = await self._invoke_model(model, prompt_for_llm, context.conversation_history)
            return response
        finally:
            if original_max_tokens is not None:
                if hasattr(model, 'max_tokens'): model.max_tokens = original_max_tokens
                elif hasattr(model, 'max_output_tokens'): model.max_output_tokens = original_max_tokens
                elif hasattr(model, 'num_predict'): model.num_predict = original_max_tokens


    async def _web_search(self, query: str, num_results: int = MAX_SEARCH_RESULTS) -> str:
        """
        AGGRESSIVE SPEED OPTIMIZATION: Fast web search with fewer results and content truncation.
        """
        try:
            client = self.tools["web_search"]
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: client.search(query, max_results=num_results)
            )
            
            if isinstance(response, dict) and "results" in response:
                results_str = []
                for i, item in enumerate(response["results"]):
                    title = item.get('title', '')
                    content = item.get('content', '')
                    url = item.get('url', '')
                    
                    if len(content) > MAX_SEARCH_CONTENT_LENGTH:
                        content = content[:MAX_SEARCH_CONTENT_LENGTH] + "..."
                    
                    images = item.get('images', [])[:1]
                    images_str = f"\n[Image]({images[0]})" if images else ""
                    
                    results_str.append(f"--- Source {i+1} ---\nTitle: {title}\nURL: {url}\nContent: {content}{images_str}\n")
                
                return "\n".join(results_str)
            return str(response)
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Web search failed: {e}"
    
    async def _invoke_model(self, model, prompt: str, history: List[BaseMessage]) -> str:
        """
        Model invocation with timeout and async support. Includes a detailed system prompt.
        """
        try:
            # AGGRESSIVE SPEED OPTIMIZATION: Keep short-term history very concise
            max_history = 3  # Reduced short-term history to minimize token count
            recent_history = history[-max_history:] if len(history) > max_history else history
            
            # Enhanced system message for detailed, contextual, and evidence-based responses
            # Optimized for conciseness while retaining instruction for detail.
            system_message_for_detail = SystemMessage(content=(
                "You are an highly intelligent AI assistant, expertly trained to provide comprehensive, "
                "detailed, and insightful answers efficiently. "
                "For any query, strive for exhaustive coverage: include historical context, current trends, "
                "and future implications where relevant. "
                "Substantiate all claims with verified data and real-world examples, "
                "and *always cite your sources explicitly using '' immediately after the relevant information*. "
                "Structure your responses clearly with headings/sub-headings (e.g., 'Introduction', 'Key Aspects', 'Analysis', 'Conclusion'). "
                "Balance conciseness with depth: ensure all critical details are present without redundancy. Highlight key takeaways. "
                "For follow-up questions, seamlessly build upon previous turns without requiring repetition. "
                "Include cross-references if helpful for broader understanding. Prioritize factual accuracy; avoid speculation."
            ))
            
            messages_to_send = [system_message_for_detail] + recent_history + [HumanMessage(content=prompt)]

            if hasattr(model, 'ainvoke'):
                response = await asyncio.wait_for(
                    model.ainvoke(messages_to_send),
                    timeout=MODEL_TIMEOUT
                )
            else:
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        lambda: model.invoke(messages_to_send)
                    ),
                    timeout=MODEL_TIMEOUT
                )
            
            return getattr(response, "content", str(response))
            
        except asyncio.TimeoutError:
            logger.error(f"Model invocation timed out after {MODEL_TIMEOUT} seconds.")
            raise
        except Exception as e:
            logger.error(f"Model invocation failed: {e}")
            raise
    
    def _build_search_prompt(self, query: str, search_results: str) -> str:
        """
        Concise search prompt. The main system message handles the detail requirement.
        """
        return f"""Based on the provided search results, concisely answer the following query:

Query: {query}

Search Results:
{search_results}

Ensure your answer is accurate and well-supported by the provided sources."""
    
    def _build_research_prompt(self, query: str, search_results: str) -> str:
        """
        Enhanced research prompt. The main system message guides structure and depth.
        """
        return f"""Conduct a comprehensive research on the following query, leveraging the provided search results.

Query: {query}

Search Results (analyze all relevant information from these sources and cite them):
{search_results}

Provide a detailed, well-structured research response. Refer to the overall system instructions for formatting and content depth, but prioritize speed and efficiency."""
    
    def _build_morphic_prompt(self, query: str, search_results: str) -> str:
        """
        Streamlined Morphic prompt.
        """
        return f"""Create a structured response for: {query}

Search results:
{search_results}

Format as:
1. Direct answer
2. Key points
3. Sources (use URLs provided in search results like '')

Keep response clear and concise, adhering to the detailed response guidelines in the system message."""
    
    def _build_Browse_prompt(self, query: str) -> str:
        """
        Concise Browse prompt.
        """
        return f"""Web Browse task: {query}

Provide:
1. Browse strategy
2. Data extraction method
3. Expected results

Keep response focused and actionable, with sufficient detail as per system guidelines."""
    
    def _build_autonomous_prompt(self, query: str) -> str:
        """
        Streamlined autonomous prompt.
        """
        return f"""Autonomous task: {query}

Provide:
1. Task analysis
2. Execution plan
3. Expected outcome

Focus on actionable steps, with detailed breakdown as per system guidelines."""
    
    def _build_coding_prompt(self, query: str) -> str:
        """
        Focused coding prompt.
        """
        return f"""Code request: {query}

Provide:
1. Working code solution
2. Brief explanation
3. Key considerations

Focus on functional, clean code, with detailed explanations as per system guidelines."""

    def record_feedback(self, query: str, response: str, rating: str):
        """
        Placeholder for a user feedback loop.
        """
        self.feedback_data.append({'query': query, 'response': response, 'rating': rating})
        logger.info(f"Feedback recorded: {rating} for query '{query[:30]}...'")


# ============================
# Utility Functions
# ============================

CONFIG_FILE = ".config.json"

def _load_config():
    """Load configuration from file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Error decoding {CONFIG_FILE}, creating new config.")
            return {}
    return {}

def _save_config(config):
    """Save configuration to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def _get_user_model_choice(engine: BaseReasoningEngine) -> str:
    """
    Prompts the user to select a model at startup.
    """
    model_map = {
        "1": "openai",
        "2": "anthropic",
        "3": "gemini",
        "4": "ollama"
    }
    reverse_model_map = {v: k for k, v in model_map.items()}
    
    config = _load_config()
    last_model = config.get("last_selected_model", "openai")

    while True:
        print(Fore.YELLOW + "\nSelect a model to use:")
        print("1] ChatGPT (OpenAI)")
        print("2] Claude (Anthropic)")
        if GEMINI_AVAILABLE:
            print("3] Gemini")
        print("4] Ollama")
        
        default_choice_num = reverse_model_map.get(last_model, "1")
        
        choice = input(Fore.CYAN + f"> [default: {default_choice_num}] ").strip()
        
        if not choice:
            selected_model_name = last_model
        elif choice in model_map:
            selected_model_name = model_map[choice]
        else:
            print(Fore.RED + "Invalid choice. Please select a number from the list.")
            continue
        
        try:
            _model_cache.get_model(selected_model_name)
            if selected_model_name not in engine.models:
                print(Fore.RED + f"Model '{selected_model_name}' is not yet fully initialized or available. Please wait a moment or check configurations.")
                continue
        except Exception as e:
            print(Fore.RED + f"Model '{selected_model_name}' is not available: {e}. Please ensure API keys are set or Ollama is running.")
            continue
        
        return selected_model_name

def _get_user_agent_choice(engine: BaseReasoningEngine) -> AgentType:
    """
    Prompts the user to select an agent at startup.
    """
    agent_map = {
        "1": AgentType.MORPHIC,
        "2": AgentType.ALFRED,
        "3": AgentType.AGENTIC_SEEK,
        "4": AgentType.AUTO
    }
    reverse_agent_map = {v: k for k, v in agent_map.items()}

    config = _load_config()
    last_agent_str = config.get("last_selected_agent", AgentType.AUTO.value)
    last_agent = AgentType(last_agent_str) if last_agent_str in [a.value for a in AgentType] else AgentType.AUTO

    while True:
        print(Fore.YELLOW + "\nSelect an agent to use:")
        print("1] Morphic Agent")
        print("2] Alfred Agent")
        print("3] Agentic Agent")
        print("4] Auto (automatically selects the best one)")
        
        default_choice_num = reverse_agent_map.get(last_agent, "4")

        choice = input(Fore.CYAN + f"> [default: {default_choice_num}] ").strip()

        if not choice:
            selected_agent_type = last_agent
        elif choice in agent_map:
            selected_agent_type = agent_map[choice]
        else:
            print(Fore.RED + "Invalid choice. Please select a number from the list.")
            continue
        
        if selected_agent_type != AgentType.AUTO:
            agent_caps = AGENT_DEFINITIONS.get(selected_agent_type)
            if agent_caps:
                if not any(m in engine.models for m in agent_caps.model_preferences):
                    print(Fore.RED + f"Selected agent '{selected_agent_type.value}' requires models that are not available. Please check API keys or Ollama setup.")
                    continue
            else:
                print(Fore.RED + f"Invalid agent type: {selected_agent_type.value}")
                continue

        return selected_agent_type


def create_reasoning_engine(config: Optional[Dict[str, Any]] = None) -> BaseReasoningEngine:
    """Factory function to create a reasoning engine"""
    return BaseReasoningEngine(config)

def get_agent_capabilities(agent_type: AgentType) -> AgentCapabilities:
    """Get capabilities for a specific agent type"""
    return AGENT_DEFINITIONS.get(agent_type)

def list_available_agents() -> List[AgentType]:
    """List all available agent types"""
    return [agent for agent in AGENT_DEFINITIONS.keys() if agent != AgentType.AUTO]

def validate_agent_for_task(agent_type: AgentType, task_type: TaskType) -> bool:
    """Check if an agent can handle a specific task type"""
    capabilities = get_agent_capabilities(agent_type)
    return task_type in capabilities.supported_tasks if capabilities else False

# ============================
# Example Usage
# ============================

async def run_prompt_loop(engine: BaseReasoningEngine):
    """Interactive prompt loop for the reasoning engine"""
    print(Fore.YELLOW + "\nIntegrated Reasoning Engine (SPEED OPTIMIZED)")
    print(Fore.YELLOW + "Type 'q' to quit, 'help' for available commands, 'history' to view session history\n")

    short_term_conversation_history: List[BaseMessage] = []

    while True:
        try:
            query = input(Fore.CYAN + "Ask anything... ").strip()
            
            if query.lower() == 'q':
                print(Fore.GREEN + "Exiting...")
                break
            elif query.lower() == 'help':
                print(Fore.WHITE + """
Available Commands:
- help : Show this help message
- q    : Quit the application
- agents: List available agents and their capabilities
- history: Show current short-term conversation history
- clear memory: Clears the long-term memory for the current session.
                """)
                continue
            elif query.lower() == 'agents':
                for agent_type in list_available_agents():
                    capabilities = get_agent_capabilities(agent_type)
                    print(f"\n{Fore.GREEN}{capabilities.name}:")
                    print(f"{Fore.WHITE}Tasks: {[t.value for t in capabilities.supported_tasks]}")
                    print(f"Tools: {[t.value for t in capabilities.available_tools]}")
                continue
            elif query.lower() == 'history':
                if not short_term_conversation_history:
                    print(Fore.MAGENTA + "No history yet for this session.")
                else:
                    print(Fore.MAGENTA + "\n--- Current Session History ---")
                    for msg in short_term_conversation_history:
                        if isinstance(msg, HumanMessage):
                            print(Fore.MAGENTA + f"User: {msg.content}")
                        elif isinstance(msg, AIMessage):
                            print(Fore.MAGENTA + f"AI: {msg.content}")
                    print(Fore.MAGENTA + "-----------------------------")
                continue
            elif query.lower() == 'clear memory':
                engine.long_term_memory.clear_memory(session_id=engine.session_id)
                short_term_conversation_history.clear()
                print(Fore.GREEN + "Memory for current session cleared.")
                continue
            
            if not query:
                print(Fore.RED + "Please enter a valid query.")
                continue

            print(Fore.BLUE + "Processing query...")
            start_time = time.time()
            
            result = await engine.reason(query, short_term_conversation_history)
            elapsed = time.time() - start_time
            
            if result["success"]:
                print(f"\n{Fore.GREEN}Selected Agent: {result['agent_used'].value.replace('_', ' ').title()}")
                print(f"{Fore.GREEN}Tools Used: {', '.join(result['tools_used'])}")
                print(f"{Fore.GREEN}Processing Time: {elapsed:.2f}s\n")
                print(f"{Fore.WHITE}{result['result']}\n")

                feedback = input(Fore.YELLOW + "Was this response helpful and comprehensive? (y/n/skip): ").strip().lower()
                if feedback in ('y', 'n'):
                    engine.record_feedback(query, result['result'], 'positive' if feedback == 'y' else 'negative')

            else:
                print(f"\n{Fore.RED}Error: {result['error']}\n")
                if any(err_msg in result['error'].lower() for err_msg in ["failed", "timed out", "api_key", "credit", "rate limit"]):
                    print(Fore.RED + f"Current model ({engine.selected_model}) failed. Please choose another model.")
                    new_model_choice = _get_user_model_choice(engine)
                    engine.selected_model = new_model_choice
                    print(Fore.GREEN + f"Switched to model: {engine.selected_model}")
                
        except KeyboardInterrupt:
            print(Fore.RED + "\nOperation interrupted by user.")
            break
        except Exception as e:
            print(Fore.RED + f"An error occurred: {e}")
            print(Fore.RED + f"An unexpected error occurred with the current model ({engine.selected_model}). Please choose another model.")
            new_model_choice = _get_user_model_choice(engine)
            engine.selected_model = new_model_choice
            print(Fore.GREEN + f"Switched to model: {engine.selected_model}")


async def main():
    """Main entry point"""
    print(Fore.GREEN + "Initializing Speed-Optimized Reasoning Engine...")
    engine = create_reasoning_engine()
    
    for _ in range(15): # Reduced wait time for initial model loading
        if engine.models:
            break
        await asyncio.sleep(0.1)
    if not engine.models:
        print(Fore.RED + "No models initialized. Check your API keys and network.")
        return

    config = _load_config()
    use_last_settings = False
    
    last_agent_from_config = config.get("last_selected_agent")
    last_model_from_config = config.get("last_selected_model")

    if last_agent_from_config and last_model_from_config:
        is_agent_available = last_agent_from_config in [a.value for a in AgentType]
        is_model_available = last_model_from_config in engine.models
        
        if is_agent_available and is_model_available:
            choice = input(Fore.YELLOW + f"Use last settings (Agent: {last_agent_from_config}, Model: {last_model_from_config})? [Y/n]: ").strip().lower()
            if choice in ('y', ''):
                engine.selected_agent = AgentType(last_agent_from_config)
                engine.selected_model = last_model_from_config
                use_last_settings = True
                print(Fore.GREEN + "Using last settings.")
            else:
                print(Fore.WHITE + "Proceeding with new selection.")
        else:
            print(Fore.YELLOW + "Last saved agent/model not available. Proceeding with new selection.")

    if not use_last_settings:
        chosen_agent = _get_user_agent_choice(engine)
        engine.selected_agent = chosen_agent
        config["last_selected_agent"] = chosen_agent.value
        _save_config(config)

        chosen_model = _get_user_model_choice(engine)
        engine.selected_model = chosen_model
        config["last_selected_model"] = chosen_model
        _save_config(config)
    
    print(Fore.GREEN + f"Current Agent Mode: {engine.selected_agent.value.replace('_', ' ').title()}")
    print(Fore.GREEN + f"Current Model: {engine.selected_model}")
    print(Fore.GREEN + f"Available models: {list(engine.models.keys())}")
    await run_prompt_loop(engine)

if __name__ == "__main__":
    asyncio.run(main())