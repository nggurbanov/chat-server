from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from chat import llm
import os 
import aiofiles

app = FastAPI()

# Define request models
class Message(BaseModel):
    id: int
    msg: str

class Context(BaseModel):
    model: str
    prompt: list
    dialogue: list

class ChangePrompt(BaseModel):
    id: int
    prompt: list

class ChangeModel(BaseModel):
    id: int
    model: str

class DeleteMessage(BaseModel):
    id: int
    message_index: int

class ClearContext(BaseModel):
    id: int

class EndChat(BaseModel):
    id: int

class LoadChat(BaseModel):
    zip_path: str

class AddRAG(BaseModel):
    id: int
    path_to_docs: str = None

# Existing endpoints from old version
@app.post("/chat/start")
async def start_chat(context: Context):
    id = await llm.start_context(context.model, context.prompt, context.dialogue)
    return {"status": "SUCCESSFUL", "id": id}

@app.post("/chat/change-prompt")
async def change_prompt(change_prompt: ChangePrompt):
    await llm.change_prompt(change_prompt.id, change_prompt.prompt)
    return {"status": "SUCCESSFUL"}

@app.post("/chat/change-model")
async def change_model(change_model: ChangeModel):
    await llm.change_model(change_model.id, change_model.model)
    return {"status": "SUCCESSFUL"}

@app.post("/chat/generate")
async def generate(message: Message):
    try:
        response = await llm.chat_generate(message.id, message.msg)
        return {"status": "SUCCESSFUL", "response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat/delete-message")
async def delete_message(delete_message: DeleteMessage):
    await llm.delete_message(delete_message.id, delete_message.message_index)
    return {"status": "SUCCESSFUL"}

@app.post("/chat/clear-context")
async def clear_context(clear_context: ClearContext):
    await llm.clear_chat(clear_context.id)
    return {"status": "SUCCESSFUL"}

@app.get("/chat/end/{chat_id}")
async def end_chat(chat_id: int):
    chat = await llm.delete_chat(chat_id)
    return FileResponse(chat, filename=chat)

@app.post("/chat/load")
async def load_chat(chat_file: UploadFile = File(...)):
    id = llm.get_unique_id()
    chat_dir = f"./chats/{id}"
    os.makedirs(chat_dir, exist_ok=True)
    file_path = os.path.join(chat_dir, chat_file.filename)
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await chat_file.read()
        await out_file.write(content)
    
    await llm.load_chat(file_path, id)
    return {"status": "SUCCESSFUL", "id": id}

@app.post("/chat/add_rag")
async def add_rag(id: int = Form(...), doc_file: UploadFile = File(None)):
    doc_dir = f"./{id}/docs"
    if doc_file:
        os.makedirs(doc_dir, exist_ok=True)  # Create directory if it doesn't exist
        doc_path = os.path.join(doc_dir, doc_file.filename)
        async with aiofiles.open(doc_path, 'wb') as out_file:
            content = await doc_file.read()
            await out_file.write(content)
        await llm.add_rag(id, doc_dir)
    else:
        await llm.add_rag(id)
    return {"status": "SUCCESSFUL"}

@app.post("/chat/rag_generate")
async def rag_generate(message: Message):
    try:
        response = await llm.rag_chat_generate(message.id, message.msg)
        return {"status": "SUCCESSFUL", "response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1205)
