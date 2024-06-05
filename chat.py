import json
import os
import aiofiles
import shutil
from openai import OpenAI
from index_base import RAG, default_prompt
from async_zip import zip, unzip
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


TOKEN = os.getenv("TOKEN")

client = OpenAI(
    api_key=TOKEN,
    base_url="https://api.deepinfra.com/v1/openai")

sys_msg = "You are a helpful assistant."

contexts = dict()
instances = dict() # instances of RAG class, unique for every chat
indexes = dict()

default_message_limit = 30

class llm():
    default_prompt = [{"role": "system", "content": sys_msg}]
    
    @staticmethod
    def get_unique_id():
        return len(contexts) + 1

    async def save_context(id):
        context_dir = f"./chats/{id}"
        context_file_path = f"{context_dir}/context.json"
        
        os.makedirs(context_dir, exist_ok=True)
        
        async with aiofiles.open(context_file_path, "w") as f:
            await f.write(json.dumps(contexts[id], indent=4))

    async def start_context(model, prompt, dialogue): # /chat/start
        id = llm.get_unique_id()
        contexts[id] = {"prompt": prompt,
                        "model": model,
                        "dialogue": dialogue}
        await llm.save_context(id)
        return id
    
    async def add_rag(id, path_to_docs=None): # /chat/add_rag
        chat_dir = f"./chats/{id}"
        instances[id] = RAG(TOKEN)
        rag = instances[id]
        
        if not path_to_docs:
            index = await rag.load_index(chat_dir+"/chroma_db")
        else:
            documents = await rag.load_docs(path_to_docs)
            index = await rag.create_index_db(chat_dir+"/chroma_db", documents)
        
        indexes[id] = index

    async def load_chat(zip_path, id):  # /chat/load-context
        directory_path = f"./chats/{id}"
        await unzip(zip_path, directory_path)
        context_file_path = f"{directory_path}/context.json"

        async with aiofiles.open(context_file_path, "r") as f:
            context_data = await f.read()
            contexts[id] = json.loads(context_data)

        return id

    async def delete_chat(id):
        contexts.pop(id)
        directory_path = f"./chats/{id}"
        zip_path = f"./{id}.zip"
        await zip(directory_path, zip_path)
        shutil.rmtree(directory_path)
        return zip_path

    async def clear_chat(id): # /chat/clear-context
        contexts[id]["dialogue"] = []
        await llm.save_context(id)

    async def delete_message(id, message_index): # /chat/delete-message
        contexts[id]["dialogue"].pop(message_index)
        contexts[id]["dialogue"].pop(message_index)
        await llm.save_context(id)

    async def change_prompt(id, prompt): # /chat/change-prompt
        contexts[id]["prompt"] = prompt
        await llm.save_context(id)

    async def change_model(id, model): # /chat/change-model
        contexts[id]["model"] = model
        await llm.save_context(id)

    async def chat_generate(id, message): # /chat/chat-generate
        if len(contexts[id]["dialogue"]) > 28:
            contexts[id]["dialogue"] = contexts[id]["dialogue"][-28:]
        contexts[id]["dialogue"].append({"role": "user", "content": message})
        messages = contexts[id]["prompt"] + contexts[id]["dialogue"]

        chat_response = client.chat.completions.create(
            model=contexts[id]["model"],
            messages=messages
        )

        response_text = chat_response.choices[0].message.content
        contexts[id]["dialogue"].append({"role": "assistant", "content": response_text})

        await llm.save_context(id)
        
        return response_text

    async def rag_chat_generate(id, message): # /chat/rag-generate

        rag = instances[id]

        retrieved_text = await rag.retrieve(indexes[id], message)

        composed_message = \
        "Please, respond to this question:\n" + message \
        + "\n\nBased on this context:\n" + retrieved_text \
        + "\n\nSend only response to the question, without mentioning this prompt."

        
        response_text = await llm.chat_generate(id, composed_message)

        return response_text


    



