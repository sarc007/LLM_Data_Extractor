# LLM Data Extractor

A local-first document extraction tool that uses LLMs (Ollama or OpenAI) to classify and extract structured financial data from PDFs and Excel files.

## Features

- **Document Classification**: Automatically identifies document types (Profit & Loss, Balance Sheet, Cash Flow, etc.) with confidence scores.
- **Structured Extraction**: Extracts complex tables and metadata into strict JSON formats.
- **Specialized Handling**: Custom logic for multi-period financial statements (Annual/Quarterly).
- **Local Privacy**: Runs entirely locally with Ollama (Llama 3, Mistral, etc.) by default.
- **Fallback Support**: Optional fallback to OpenAI for difficult documents.
- **History**: Keeps track of processed files and results.

## Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/ammarfitwalla/LLM_Data_Extractor.git
    cd LLM_Data_Extractor
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```ini
    # App Security
    SECRET_KEY=your_secret_key_here
    AUTH_ENABLED=false

    # LLM Settings (Ollama is default)
    OLLAMA_HOST=http://localhost:11434
    OLLAMA_MODEL=llama3.1:8b

    # OpenAI Fallback (Optional)
    USE_OPENAI_FALLBACK=false
    OPENAI_API_KEY=your_openai_key
    OPENAI_MODEL=gpt-4o
    ```

4.  **Install Ollama**
    Download from [ollama.com](https://ollama.com) and pull a model:
    ```bash
    ollama pull llama3.1:8b
    ```

## Usage

0. **Start the Ollama Local Server**
    ```bash
    ollama serve
    ```

2.  **Start the FastAPI Server**
    ```bash
    uvicorn app.main:app --reload
    ```

3.  **Open in Browser**
    Navigate to `http://localhost:8000`.

4.  **Upload & Process**
    - Upload a PDF or Excel file.
    - Click "Process with LLM".
    - View the classified type and extracted tables side-by-side.

## Project Structure

- `app/pipeline.py`: Core logic for classification, prompt handling, and LLM interaction.
- `app/extraction_templates.py`: JSON schemas for different document types.
- `app/templates/`: HTML frontend templates (Tailwind CSS + HTMX).
- `input_files/`: Directory for placing raw files manually (optional).
- `processed/`: Directory where extracted JSON results are saved.
