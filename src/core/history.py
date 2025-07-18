import os
from langchain_core.messages import HumanMessage, AIMessage
import chromadb
from chromadb.config import Settings
import openai 
from dotenv import load_dotenv
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

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


def add_to_history(history: list, user_input: str, llm_reply: str) -> None:
    """
    Appends the user and assistant messages to the chat history.
    This is typically for short-term, turn-based context.
    """
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=llm_reply))


class ConversationMemory:
    """
    Long-term memory system using Chroma vector store for semantic search
    and retrieval of past conversations. This provides a persistent and
    contextually aware memory for the AI assistant.
    """
    def __init__(
        self,
        collection_name: str = "conversation_history",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "text-embedding-ada-002",
        openai_api_key: Optional[str] = None
    ):
        """
        Initializes the ConversationMemory with a ChromaDB client and collection.

        Args:
            collection_name (str): Name of the ChromaDB collection.
            persist_directory (str): Directory where ChromaDB data will be stored.
            embedding_model (str): OpenAI embedding model to use for generating embeddings.
            openai_api_key (Optional[str]): OpenAI API key. Required for embedding generation.
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Prioritize the key passed in during initialization, then check environment variable
        # Using "OPEN_API_KEY" as specified by the user.
        final_api_key = openai_api_key or os.getenv("OPEN_API_KEY") 
        
        if not final_api_key:
            logger.warning("OPEN_API_KEY not found. Embedding generation will fail without an API key.")
            self.openai_client = None # Set client to None if no key
        else:
            try:
                # Instantiate the OpenAI client directly here using the imported 'openai' module
                self.openai_client = openai.OpenAI(api_key=final_api_key)
                logger.info("OpenAI client initialized successfully in ConversationMemory.")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client in ConversationMemory: {e}. Please check your API key.")
                self.openai_client = None

        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True)
        )

        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(f"Loaded or created ChromaDB collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error accessing or creating ChromaDB collection {collection_name}: {e}")
            raise


    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text using OpenAI's embedding model.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The embedding vector. Returns a zero vector if embedding fails.
        """
        if self.openai_client is None:
            logger.error("OpenAI client not initialized. Cannot generate embeddings.")
            return [0.0] * 1536  # Default embedding size for ada-002 (common size)

        try:
            if not text.strip():
                return [0.0] * 1536

            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except openai.APIStatusError as e: # Correctly catching API status errors
            logger.error(f"OpenAI API status error generating embedding: {e.status_code} - {e.response.json()}")
            return [0.0] * 1536
        except openai.APITimeoutError as e: # Correctly catching API timeout errors
            logger.error(f"OpenAI API timeout error generating embedding: {e}")
            return [0.0] * 1536
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            return [0.0] * 1536

    def store_conversation(
        self,
        user_input: str,
        llm_reply: str,
        session_id: Optional[str] = None
    ) -> str:
        """
        Store a user input and LLM response pair in the vector database.
        """
        conversation_id = str(uuid.uuid4())
        conversation_text = f"User: {user_input}\nAssistant: {llm_reply}"
        embedding = self._get_embedding(conversation_text)
        
        if all(val == 0.0 for val in embedding) and self.openai_client is None:
            logger.warning(f"Skipping storage for conversation {conversation_id} due to embedding failure (no OpenAI client).")
            return conversation_id

        metadata = {
            "user_input": user_input,
            "llm_reply": llm_reply,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id or "default",
            "conversation_id": conversation_id
        }
        try:
            self.collection.add(
                ids=[conversation_id],
                embeddings=[embedding],
                documents=[conversation_text],
                metadatas=[metadata]
            )
            logger.info(f"Stored conversation with ID: {conversation_id} for session: {session_id or 'default'}")
            return conversation_id
        except Exception as e:
            logger.error(f"Error storing conversation {conversation_id}: {e}")
            return conversation_id

    def retrieve_relevant_memories(
        self,
        query: str,
        n_results: int = 5,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant past conversations based on semantic similarity to the query.
        """
        try:
            query_embedding = self._get_embedding(query)
            if all(val == 0.0 for val in query_embedding):
                logger.warning("Query embedding failed, returning no relevant memories.")
                return []

            where_clause = None
            if session_id:
                where_clause = {"session_id": session_id}
                logger.debug(f"Retrieving memories for session_id: {session_id}")

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            relevant_memories = []
            if results and results.get('documents') and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    memory = {
                        'id': results['ids'][0][i],
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results.get('distances') and results['distances'][0] else None
                    }
                    relevant_memories.append(memory)
            logger.info(f"Retrieved {len(relevant_memories)} relevant memories for query '{query[:30]}...'")
            return relevant_memories
        except Exception as e:
            logger.error(f"Error retrieving memories for query '{query[:30]}...': {e}")
            return []

    def get_conversation_context(
        self,
        query: str,
        max_context_length: int = 2000,
        n_results: int = 3,
        session_id: Optional[str] = None
    ) -> str:
        """
        Generates a formatted string of relevant past conversations to be used as context.
        """
        relevant_memories = self.retrieve_relevant_memories(query, n_results, session_id)
        if not relevant_memories:
            return ""

        context_parts = ["# Relevant Past Conversations (from Long-Term Memory):"]
        current_length = len(context_parts[0])

        for memory in relevant_memories:
            metadata = memory['metadata']
            user_input = metadata.get('user_input', 'N/A')
            llm_reply = metadata.get('llm_reply', 'N/A')
            
            memory_text = f"\n## Past Interaction (ID: {memory['id'][:8]}):\nUser: {user_input}\nAssistant: {llm_reply}\n"
            
            if current_length + len(memory_text) > max_context_length:
                logger.info(f"Context length limit reached ({max_context_length} chars). Truncating relevant memories.")
                break
            
            context_parts.append(memory_text)
            current_length += len(memory_text)
            
        return "".join(context_parts)

    def update_conversation_history(
        self,
        history: List,
        user_input: str,
        llm_reply: str,
        session_id: Optional[str] = None
    ) -> None:
        """
        Enhanced version of add_to_history that also stores in vector database for long-term memory.
        """
        add_to_history(history, user_input, llm_reply)
        self.store_conversation(user_input, llm_reply, session_id)

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the conversation memory collection.
        """
        try:
            count = self.collection.count()
            return {
                "total_conversations": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def clear_memory(self, session_id: Optional[str] = None) -> None:
        """
        Clear conversation memory. If session_id is provided, only clear that session.
        If no session_id is provided, all conversations in the collection are deleted.
        """
        try:
            if session_id:
                results = self.collection.get(where={"session_id": session_id}, include=[])
                if results['ids']:
                    self.collection.delete(ids=results['ids'])
                    logger.info(f"Cleared {len(results['ids'])} conversations for session: {session_id}")
                else:
                    logger.info(f"No conversations found for session: {session_id} to clear.")
            else:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "AI assistant conversation history"}
                )
                logger.info("Cleared all conversation memory by resetting collection.")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")

    def get_relevant_memories(self, query: str, limit: int = 3) -> list:
        """
        Returns a list of dicts with 'user_input' and 'response' keys for compatibility.
        """
        memories = self.retrieve_relevant_memories(query, n_results=limit)
        formatted = []
        for m in memories:
            meta = m.get('metadata', {})
            formatted.append({
                "user_input": meta.get("user_input", ""),
                "response": meta.get("llm_reply", "")
            })
        return formatted

    def add_conversation(self, user_input: str, response: str, agent_type: str = "", tools_used: list = None) -> None:
        """
        Stores a conversation in the vector DB. Ignores agent_type/tools_used for now for simplicity.
        """
        self.store_conversation(user_input, response)


# Example usage and integration (unchanged for this fix)
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    try:
        temp_client = chromadb.PersistentClient(path="./chroma_db_temp")
        temp_client.delete_collection("conversation_history_test")
        logger.info("Cleaned up previous test collection.")
    except Exception:
        pass

    memory = ConversationMemory(
        collection_name="conversation_history_test",
        persist_directory="./chroma_db_temp",
        openai_api_key=os.getenv("OPEN_API_KEY")
    )

    print("\n--- Storing conversations ---")
    memory.store_conversation(
        "What is machine learning?",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        session_id="session1"
    )
    memory.store_conversation(
        "Explain deep learning in simple terms.",
        "Deep learning is a type of machine learning that uses neural networks with many layers to learn from data.",
        session_id="session1"
    )
    memory.store_conversation(
        "How is AI used in healthcare?",
        "AI in healthcare helps with diagnosis, drug discovery, personalized treatment, and operational efficiency.",
        session_id="session2"
    )
    memory.store_conversation(
        "Can AI compose music?",
        "Yes, AI can compose music by learning patterns from existing musical pieces and generating new ones.",
        session_id="session1"
    )


    print("\n--- Retrieving relevant memories for 'Tell me about AI research' ---")
    relevant = memory.retrieve_relevant_memories("Tell me about AI research", n_results=2, session_id="session1")
    for mem in relevant:
        print(f"ID: {mem['id']}, Distance: {mem['distance']:.4f}\nContent:\n{mem['content']}\n")

    print("\n--- Getting conversation context for 'What are neural networks?' ---")
    context = memory.get_conversation_context("What are neural networks?", max_context_length=500, session_id="session1")
    print(context)

    print("\n--- Getting conversation context for 'latest AI in medicine' (session2) ---")
    context_s2 = memory.get_conversation_context("latest AI in medicine", max_context_length=500, session_id="session2")
    print(context_s2)

    print("\n--- Getting conversation context for a new topic (should be empty if no relevant memory) ---")
    new_context = memory.get_conversation_context("What is quantum computing?", session_id="session1")
    print(f"New context: '{new_context}'")

    stats = memory.get_collection_stats()
    print("\n--- Collection stats:", stats)

    print("\n--- Clearing memory for session1 ---")
    memory.clear_memory(session_id="session1")
    stats_after_clear = memory.get_collection_stats()
    print("Collection stats after clearing session1:", stats_after_clear)

    print("\n--- Attempting to retrieve from cleared session1 (should be empty) ---")
    relevant_after_clear = memory.retrieve_relevant_memories("Explain deep learning", session_id="session1")
    print("Relevant memories after clearing session1:", relevant_after_clear)

    print("\n--- Storing a new conversation after clearing session1 ---")
    memory.store_conversation("New query after clear", "New reply after clear", session_id="session1")
    stats_after_new = memory.get_collection_stats()
    print("Collection stats after new convo in session1:", stats_after_new)

    print("\n--- Clearing all memory (including session2 and the new one) ---")
    memory.clear_memory()
    stats_final = memory.get_collection_stats()
    print("Collection stats after clearing all:", stats_final)