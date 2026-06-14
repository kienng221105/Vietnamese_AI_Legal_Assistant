# Vietnamese AI Legal Assistant

Graph RAG Legal Assistant application using FastAPI, Neo4j, BM25, and Sentence-Transformers.

## 🛠️ Installation & Setup

1. **Clone the repository and navigate inside:**
   ```bash
   cd Vietnamese_AI_Legal_Assistant
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Modify the `.env` file at the root directory with your correct Neo4j credentials and Gemini API Key.

5. **Start the FastAPI application:**
   ```bash
   python main.py
   ```