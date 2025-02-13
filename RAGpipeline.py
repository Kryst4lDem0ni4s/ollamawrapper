"""
title: Llama Index Ollama Memory Pipeline
author: Khwaish Arora
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retaining conversation memory, summarizing history, and managing project architecture using LangChain and Ollama with Llama Index.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, langchain
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os
import json

from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
import chromadb

# File paths for memory persistence
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL_NAME = "nomic-embed-text"
MEMORY_FILE = "/app/backend/data/conversation_memory.json"
ARCHITECTURE_FILE = "/app/backend/data/project_architecture.json"
HISTORY_LIMIT = 30000


class Pipeline:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str
        OLLAMA_EMBEDDING_MODEL_NAME: str
        DEEPSEEK_CHAT_MODEL: str
        DEEPSEEK_CODER_MODEL: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            **{
                "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
                "DEEPSEEK_CHAT_MODEL": os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-r1:7b"),
                "DEEPSEEK_CODER_MODEL": os.getenv("DEEPSEEK_CODER_MODEL", "deepseek-coder-v2:16b"),
                "OLLAMA_EMBEDDING_MODEL_NAME": os.getenv("OLLAMA_EMBEDDING_MODEL_NAME", OLLAMA_EMBEDDING_MODEL_NAME),
            }
        )

        # Initialize memory and architecture storage
        self.memory = self.load_memory()
        self.project_architecture = self.load_architecture()
        self.chroma_client = chromadb.PersistentClient("/app/backend/data/memory_db")
        self.collection = self.chroma_client.get_or_create_collection("chat_memory")

    def detect_task_type(self, user_message: str) -> str:
        """Detects if the query is general or coding-related."""
        code_keywords = ["def", "import", "class", "function", "#include", "{", "}", "print(", "code", "python", "script", "py", "coding", "file", "json", "()", "command"]
        return self.valves.DEEPSEEK_CODER_MODEL if any(kw in user_message for kw in code_keywords) else self.valves.DEEPSEEK_CHAT_MODEL

    async def on_startup(self):
        """Initialize LlamaIndex and load document data on server startup."""
        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.OLLAMA_EMBEDDING_MODEL_NAME,
            base_url=self.valves.OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.DEEPSEEK_CHAT_MODEL,
            base_url=self.valves.OLLAMA_BASE_URL,
        )
        os.makedirs("/app/backend/data", exist_ok=True)
        self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)

    def load_memory(self):
        """Load stored conversation memory from file."""
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as file:
                try:
                    data = json.load(file)
                    return ConversationBufferMemory(chat_memory=ChatMessageHistory(messages=data.get("memory", [])))
                except json.JSONDecodeError:
                    return ConversationBufferMemory()
        return ConversationBufferMemory()

    def save_memory(self):
        """Save current conversation memory to file."""
        with open(MEMORY_FILE, "w") as file:
            json.dump({"memory": self.memory.chat_memory.messages}, file)

    def load_architecture(self):
        """Load stored project architecture from file."""
        if os.path.exists(ARCHITECTURE_FILE):
            with open(ARCHITECTURE_FILE, "r") as file:
                try:
                    return json.load(file)
                except json.JSONDecodeError:
                    return {}
        return {}

    def save_architecture(self):
        """Save project architecture details to file."""
        with open(ARCHITECTURE_FILE, "w") as file:
            json.dump(self.project_architecture, file)

    def summarize_conversation(self, conversation_history):
        """Summarizes conversation history when it gets too long."""
        summarizer = ChatOllama(model=self.valves.DEEPSEEK_CHAT_MODEL, base_url=self.valves.OLLAMA_BASE_URL)
        summary_prompt = f"Summarize the following conversation for context retention:\n\n{conversation_history}\n\nSummary:"
        return summarizer.invoke(summary_prompt)

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """Processes user messages, retains memory, updates project architecture, and retrieves information via LlamaIndex."""

        # Store architecture details if mentioned
        if "architecture" in user_message.lower():
            self.project_architecture["latest"] = user_message
            self.save_architecture()

        # Check if history exceeds limit and summarize
        conversation_history = self.memory.chat_memory.messages
        if len(conversation_history) > HISTORY_LIMIT:
            summarized_history = self.summarize_conversation(conversation_history)
            self.memory.chat_memory.clear()
            self.memory.chat_memory.add_user_message("Summary of previous conversation")
            self.memory.chat_memory.add_ai_message(summarized_history)
            self.save_memory()

        # Custom prompt with memory and project architecture
        prompt_template = PromptTemplate(
            input_variables=["history", "architecture", "input"],
            template="""You are an AI assistant that retains conversation history and project architecture details.\n"
                     Conversation History:\n{history}\n\n
                     Project Architecture:\n{architecture}\n\n
                     Now, respond to the user's latest query\n
                     User: {input}\nAssistant:"""
        )

        selected_model = self.detect_task_type(user_message)
        llm = ChatOllama(model=selected_model, base_url=self.valves.OLLAMA_BASE_URL)
        conversation = ConversationChain(llm=llm, memory=self.memory, prompt=prompt_template)

        response = conversation.predict(input=user_message, architecture=self.project_architecture.get("latest", "No architecture details available."))

        # Save updated memory
        self.save_memory()

        # Retrieve knowledge from LlamaIndex
        query_engine = self.index.as_query_engine()
        rag_response = query_engine.query(user_message)

        return f"{response}\n\nAdditional Information:\n{rag_response.response}"
