# DataWorks Automation Agent

A FastAPI-based automation agent for handling various data operations and business tasks with built-in security features and LLM integration.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Task Types](#task-types)
- [Security Features](#security-features)
- [Error Handling](#error-handling)
- [Contributing](#contributing)

## Features

### Operations Tasks
- File system operations within `/data` directory
- Markdown formatting and processing
- Date processing and analysis
- JSON data sorting and manipulation
- Log file processing
- Document indexing
- Email data extraction
- Credit card number extraction from images
- Text similarity analysis using embeddings
- Database operations

### Business Tasks
- API data fetching
- Git repository operations
- SQL query execution
- Web scraping
- Image processing
- Audio transcription
- Markdown to HTML conversion
- CSV filtering API endpoint

### Security Features
- Path traversal protection
- Data access restrictions to `/data` directory
- Prevention of file deletion operations
- API key authentication
- Input validation
- Secure file operations

## Prerequisites

- Python 3.8+
- Node.js (for Prettier)
- Git
- OpenAI API key
- SQLite/DuckDB

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dataworks-automation-agent.git
cd dataworks-automation-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Install Node.js dependencies:
```bash
npm install prettier@3.4.2 -g
```

5. Create necessary directories:
```bash
mkdir -p data/docs data/logs
```

## Configuration

1. Create a `.env` file in the project root:
```env
API_SECRET_KEY=your_secret_key
OPENAI_API_KEY=your_openai_api_key
USER_EMAIL=your_email@example.com
```

2. Update configuration in `config.py` if needed:
```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
```

## Usage

1. Start the server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

2. Make API requests:
```bash
curl -X POST "http://localhost:8000/run" \
     -H "X-API-KEY: your_secret_key" \
     -H "Content-Type: application/json" \
     -d '{"task": "format markdown in /data/format.md"}'
```

## API Endpoints

### POST /run
Execute an automation task.

**Request Headers:**
- `X-API-KEY`: API authentication key
- `Content-Type`: application/json

**Request Body:**
```json
{
    "task": "task description"
}
```

**Response:**
```json
{
    "status": "success|error",
    "message": "result message"
}
```

## Task Types

### Operations Tasks
1. **Install UV and Run Datagen**
   ```
   task: "install uv and run datagen.py"
   ```

2. **Format Markdown**
   ```
   task: "format markdown in /data/format.md"
   ```

3. **Count Weekdays**
   ```
   task: "count wednesdays in /data/dates.txt"
   ```

[Additional task examples...]

### Business Tasks
1. **API Data Fetching**
   ```
   task: "fetch api data url=https://api.example.com/data"
   ```

2. **Git Operations**
   ```
   task: "git clone url=https://github.com/user/repo.git"
   ```

[Additional business task examples...]

## Security Features

### Path Validation
- All file operations are restricted to the `/data` directory
- Path traversal attempts are blocked
- Deletion operations are prevented

### Authentication
- API key required for all operations
- Key validation middleware

### Error Handling
- Custom exception handling
- Detailed error logging
- Secure error responses

## Error Handling

### Common Error Codes
- 400: Bad Request
- 403: Invalid API Key
- 404: Resource Not Found
- 500: Internal Server Error

### Error Response Format
```json
{
    "status": "error",
    "message": "Error description"
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any queries or support, please contact [your-email@example.com](mailto:your-email@example.com)
