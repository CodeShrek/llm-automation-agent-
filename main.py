from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import aiofiles
import os
from dotenv import load_dotenv
import openai
import subprocess
import json
from datetime import datetime
from PIL import Image
import logging
import traceback
from fastapi.responses import JSONResponse
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

app = FastAPI()

# Get API keys from environment variables
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
openai.api_key = AIPROXY_TOKEN

# Define API key header
api_key_header = APIKeyHeader(name="X-API-KEY")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_SECRET_KEY:
        logging.warning(f"Invalid API key attempt: {api_key}")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.get("/")
def home():
    return {"message": "LLM Automation Agent is running!"}

@app.get("/read")
async def read_file(path: str, api_key: str = Security(api_key_header)):
    validate_api_key(api_key)
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
def run_task(task: Task, api_key: str = Security(api_key_header)):
    validate_api_key(api_key)
    logging.info(f"Task received: {task.task}")
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an automation agent that executes tasks."},
                {"role": "user", "content": task.task}
            ]
        )
        command = response["choices"][0]["message"]["content"]
        logging.info(f"Interpreted command: {command}")
        
        result = execute_task(command)
        if result["status"] == "success":
            return JSONResponse(status_code=200, content=result)
        else:
            return JSONResponse(status_code=400, content=result)
    except Exception as e:
        logging.error(f"Error executing task: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": "Internal Server Error"}
        )

def execute_task(command):
    try:
        if not command or len(command.strip()) == 0:
            return {"status": "error", "message": "No valid task command provided."}

        if "install uv" in command:
            subprocess.run(["pip", "install", "uv"], check=True)
            return {"status": "success", "message": "uv installed successfully."}

        elif "format" in command and "prettier" in command:
            if not os.path.exists(os.path.join(DATA_DIR, "format.md")):
                return {"status": "error", "message": "format.md file not found!"}
            subprocess.run(["npx", "prettier", "--write", f"{DATA_DIR}/format.md"], check=True)
            return {"status": "success", "message": "Markdown formatted successfully."}

        elif "count Wednesdays" in command:
            dates_path = os.path.join(DATA_DIR, "dates.txt")
            if not os.path.exists(dates_path):
                return {"status": "error", "message": "dates.txt file not found!"}
            
            with open(dates_path, "r") as f:
                dates = [datetime.strptime(line.strip(), "%Y-%m-%d").weekday() for line in f]
            wednesday_count = dates.count(2)
            return {"status": "success", "message": f"Wednesdays count: {wednesday_count}"}

        elif "sort contacts" in command:
            contacts_path = os.path.join(DATA_DIR, "contacts.json")
            if not os.path.exists(contacts_path):
                return {"status": "error", "message": "contacts.json file not found!"}
            
            try:
                with open(contacts_path, "r") as f:
                    contacts = json.load(f)
            except json.JSONDecodeError:
                return {"status": "error", "message": "Invalid JSON in contacts.json"}
                
            sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))
            with open(os.path.join(DATA_DIR, "contacts-sorted.json"), "w") as f:
                json.dump(sorted_contacts, f, indent=2)
            return {"status": "success", "message": "Contacts sorted successfully."}

        elif "extract first lines of logs" in command:
            logs_dir = os.path.join(DATA_DIR, "logs")
            if not os.path.exists(logs_dir):
                return {"status": "error", "message": "logs directory not found!"}
                
            log_files = sorted(os.listdir(logs_dir), 
                             key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), 
                             reverse=True)
            with open(os.path.join(DATA_DIR, "logs-recent.txt"), "w") as f:
                for file in log_files[:10]:
                    with open(os.path.join(logs_dir, file), "r") as log:
                        f.write(log.readline())
            return {"status": "success", "message": "Log lines extracted successfully."}

        elif "extract email sender" in command:
            with open("data/email.txt", "r") as f:
                email_text = f.read()
            sender = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract the sender's email from this message."},
                    {"role": "user", "content": email_text}
                ]
            )["choices"][0]["message"]["content"]
            with open("data/email-sender.txt", "w") as f:
                f.write(sender.strip())

        elif "extract credit card number" in command:
            image = Image.open("data/credit-card.png")
            text = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract the credit card number from this image."},
                    {"role": "user", "content": image}
                ]
            )["choices"][0]["message"]["content"]
            with open("data/credit-card.txt", "w") as f:
                f.write(text.replace(" ", ""))

        else:
            return {"status": "error", "message": "Unsupported command. Please check the input."}

    except Exception as e:
        error_message = f"Task failed: {str(e)}"
        logging.error(error_message)
        traceback.print_exc()
        return {"status": "error", "message": error_message}

# Add this at the very end of the file
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

