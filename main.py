from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import aiofiles
import os
from dotenv import load_dotenv
import base64
import openai
import subprocess
import json
from datetime import datetime
from PIL import Image
import logging
import traceback
from fastapi.responses import JSONResponse, Response
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import requests
import sqlite3
import re
import markdown
import shutil
from collections import defaultdict
import glob
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create necessary directories
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Fix: Changed 'ios' to 'os'
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "docs"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "logs"), exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define API key header
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)
API_SECRET_KEY = "mysecret123"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def validate_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.get("/")
def home():
    return {"message": "LLM Automation Agent is running!"}

class Task(BaseModel):
    task: str

@app.post("/run")
async def run_task(task: Task, api_key: str = Depends(validate_api_key)):
    logging.info(f"Task received: {task.task}")
    try:
        result = execute_task(task.task)
        logging.info(f"Task execution result: {result}")
        return JSONResponse(status_code=200, content=result) if result["status"] == "success" else JSONResponse(status_code=400, content=result)
    except Exception as e:
        error_msg = f"Error executing task: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

def execute_task(command):
    try:
        # A1: Install uv and run datagen.py
        if "install uv" in command.lower():
            subprocess.run(["pip", "install", "uv"], check=True)
            return {"status": "success", "message": "uv installed successfully"}

        if "run datagen" in command.lower():
            email = os.getenv("USER_EMAIL", "test@example.com")
            subprocess.run(["python", "-m", "pip", "install", "requests"], check=True)
            response = requests.get("https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py")
            with open("datagen.py", "w") as f:
                f.write(response.text)
            subprocess.run(["python", "datagen.py", email], check=True)
            return {"status": "success", "message": "Data generation completed"}

        # A2: Format markdown
        if "format markdown" in command.lower():
            subprocess.run(["npx", "prettier@3.4.2", "--write", "data/format.md"], check=True)
            return {"status": "success", "message": "Markdown formatted"}

        # A3: Count wednesdays
        if "count wednesdays" in command.lower():
            with open("data/dates.txt", "r") as f:
                dates = [datetime.strptime(line.strip(), "%Y-%m-%d").weekday() for line in f]
            wednesdays = sum(1 for d in dates if d == 2)
            with open("data/dates-wednesdays.txt", "w") as f:
                f.write(str(wednesdays))
            return {"status": "success", "message": "Wednesdays counted"}

        # A4: Sort contacts
        if "sort contacts" in command.lower():
            with open("data/contacts.json", "r") as f:
                contacts = json.load(f)
            contacts.sort(key=lambda x: (x["last_name"], x["first_name"]))
            with open("data/contacts-sorted.json", "w") as f:
                json.dump(contacts, f, indent=4)
            return {"status": "success", "message": "Contacts sorted"}

        # A5: Write first line of 10 most recent log files
        if "log files" in command.lower():
            log_files = glob.glob(os.path.join(DATA_DIR, "logs", "*.log"))
            log_files.sort(key=os.path.getmtime, reverse=True)
            recent_logs = log_files[:10]
            
            first_lines = []
            for log_file in recent_logs:
                try:
                    with open(log_file, 'r') as f:
                        first_line = f.readline().strip()
                        first_lines.append(first_line)
                except Exception as e:
                    logging.error(f"Error reading log file {log_file}: {str(e)}")
                    first_lines.append(f"Error reading file: {os.path.basename(log_file)}")
            
            with open(os.path.join(DATA_DIR, "logs-recent.txt"), 'w') as f:
                f.write('\n'.join(first_lines))
            
            return {"status": "success", "message": "Recent log first lines extracted"}

        # A6: Create docs index.json
        if "create index" in command.lower():
            md_files = glob.glob(os.path.join(DATA_DIR, "docs", "**", "*.md"), recursive=True)
            index = {}
            
            for md_file in md_files:
                try:
                    with open(md_file, 'r') as f:
                        content = f.read()
                        
                    # Find the first H1 header
                    h1_match = re.search(r'^# (.+)$', content, re.MULTILINE)
                    if h1_match:
                        title = h1_match.group(1).strip()
                        # Get relative path
                        rel_path = os.path.relpath(md_file, os.path.join(DATA_DIR, "docs"))
                        index[rel_path] = title
                except Exception as e:
                    logging.error(f"Error processing markdown file {md_file}: {str(e)}")
            
            with open(os.path.join(DATA_DIR, "docs", "index.json"), 'w') as f:
                json.dump(index, f, indent=4)
            
            return {"status": "success", "message": "Docs index created"}

        # A7: Extract email sender
        if "extract email sender" in command.lower():
            with open("data/email.txt", "r") as f:
                content = f.read()
            sender_email = re.search(r"From: .*?<(.*?)>", content).group(1)
            with open("data/email-sender.txt", "w") as f:
                f.write(sender_email)
            return {"status": "success", "message": "Email sender extracted"}

        # A8: Extract credit card
        if "extract credit card" in command.lower():
            # Load the image
            image_path = os.path.join(DATA_DIR, "credit-card.png")
            
            # Open the image file
            with open(image_path, "rb") as image_file:
                # Create a request to OpenAI
                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract the credit card number from this image. Return only the digits without any spaces or formatting."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}",
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                )
            
            # Extract the card number from the response
            extracted_text = response.choices[0].message.content.strip()
            # Remove any non-digit characters
            card_number = ''.join(filter(str.isdigit, extracted_text))
            
            with open(os.path.join(DATA_DIR, "credit-card.txt"), "w") as f:
                f.write(card_number)
                
            return {"status": "success", "message": "Credit card extracted"}

        # A9: Find similar comments
        if "find similar comments" in command.lower():
            with open(os.path.join(DATA_DIR, "comments.txt"), "r") as f:
                comments = [line.strip() for line in f if line.strip()]
            
            # Get embeddings for all comments
            all_embeddings = []
            for comment in comments:
                response = client.embeddings.create(
                    input=comment,
                    model="text-embedding-ada-002"
                )
                all_embeddings.append(response.data[0].embedding)
            
            # Find the most similar pair
            max_similarity = -1
            most_similar_pair = (0, 0)
            
            for i in range(len(comments)):
                for j in range(i+1, len(comments)):
                    # Calculate cosine similarity
                    similarity = compute_cosine_similarity(all_embeddings[i], all_embeddings[j])
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_pair = (i, j)
            
            # Write the most similar comments to the output file
            with open(os.path.join(DATA_DIR, "comments-similar.txt"), "w") as f:
                f.write(f"{comments[most_similar_pair[0]]}\n{comments[most_similar_pair[1]]}")
            
            return {"status": "success", "message": "Similar comments found"}

        # A10: Calculate ticket sales
        if "calculate ticket sales" in command.lower():
            conn = sqlite3.connect(os.path.join(DATA_DIR, "ticket-sales.db"))
            cursor = conn.cursor()
            cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type='Gold'")
            total_sales = cursor.fetchone()[0] or 0
            conn.close()
            with open(os.path.join(DATA_DIR, "ticket-sales-gold.txt"), "w") as f:
                f.write(str(total_sales))
            return {"status": "success", "message": "Ticket sales calculated"}

        return {"status": "error", "message": "Unknown task"}
    except Exception as e:
        logging.error(f"Error executing task: {str(e)}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors"""
    import numpy as np
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
