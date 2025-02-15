from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import aiofiles
import os

app = FastAPI()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "LLM Automation Agent is running!"}

@app.get("/read")
async def read_file(path: str):
    file_path = os.path.join(DATA_DIR, path.lstrip("/"))

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
        content = await file.read()

    return {"filename": path, "content": content}

# Define a Pydantic model for the task input
class Task(BaseModel):
    task: str

@app.post("/run")
def run_task(task: Task):
    # This is a placeholder - actual logic will be implemented in Step 3
    return {"task": task.task, "status": "Task execution logic will be implemented here"}

