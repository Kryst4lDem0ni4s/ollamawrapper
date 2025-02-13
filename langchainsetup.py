"""
title: Basic Langchain implementation
author: Khwaish Arora
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retaining conversation memory using LangChain and Ollama
requirements: ollama, langchain
"""

from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()

llm = ChatOpenAI(model_name="ollama", base_url="http://127.0.0.1:11434")
conversation = ConversationChain(llm=llm, memory=memory)
import json
from pathlib import Path

memory_file = Path("/app/chat_memory/conversations.json")

def save_memory():
    with open(memory_file, "w") as f:
        json.dump(memory.load_memory_variables({}), f)

def load_memory():
    if memory_file.exists():
        with open(memory_file) as f:
            memory.chat_memory.add_messages(json.load(f))
            
def chat(user_input):
    response = conversation.predict(input=user_input)
    return response

# Example Usage
while True:
    user_input = input("You: ")
    print("Bot:", chat(user_input))


