import requests
import time
import os
from dotenv import load_dotenv

base_url = os.getenv("URL")

# Test /chat/start
response = requests.post(
    base_url + "start",
    json={
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "prompt": [{"role": "system", "content": "This is the default prompt"}],
        "dialogue": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo wassup"}]
    }
)
start_data = response.json()
print("Start:\t", start_data['status'])
context_id = start_data["id"]
# time.sleep(3)

# Test /chat/clear-context
response = requests.post(
    base_url + "clear-context",
    json={"id": context_id}
)
print("Clear Context:\t", response.json()['status'])
# time.sleep(1)

# Test /chat/change-prompt
response = requests.post(
    base_url + "change-prompt",
    json={
        "id": context_id,
        "prompt": [{"role": "system", "content": "This is the new prompt"}]
    }
)
print("Change Prompt:\t", response.json()['status'])
# time.sleep(1)

# Test /chat/change-model
response = requests.post(
    base_url + "change-model",
    json={
        "id": context_id,
        "model": "microsoft/WizardLM-2-8x22B"
    }
)
print("Change Model:\t", response.json()['status'])
# time.sleep(1)

# Test /chat/generate
response = requests.post(
    base_url + "generate",
    json={"id": context_id, "msg": "Hello, how are you?"}
)
print("Generate:\t", response.json()['status'])
# time.sleep(1)

# Test /chat/delete-message
response = requests.post(
    base_url + "delete-message",
    json={"id": context_id, "message_index": 0}
)
print("Delete Message:\t", response.json()['status'])
# time.sleep(1)

# Test /chat/end
response = requests.get(
    base_url + f"end/{context_id}"
)
end_data = response.json()
print("End Chat:\t", end_data['status'])
print("Chat history:\n" + str(end_data['file']))

# time.sleep(3)

# Test /chat/load
response = requests.post(
    base_url + "load",
    json={"zip_path": "1.zip"}
)
load_data = response.json()
print("Load Chat:\t", load_data['status'])
loaded_context_id = load_data["id"]
# time.sleep(3)

# Test /chat/add_rag with file
with open("data/pamyatka.txt", "rb") as f:
    response = requests.post(
        base_url + "add_rag",
        data={"id": loaded_context_id},
        files={"doc_file": f}
    )
print("Add RAG with file:\t", response.json()['status'])
# time.sleep(3)

# Test /chat/rag_generate
response = requests.post(
    base_url + "rag_generate",
    json={"id": loaded_context_id, "msg": "Что такое желтая волна?"}
)
print("RAG Generate:\t", response.json()['status'])
print("RAG Response:\t", response.json()['response'])
