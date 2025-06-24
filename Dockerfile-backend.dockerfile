FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements-backend.txt .
COPY frequency_dictionary_en_82_765.txt .
RUN pip install --upgrade pip && pip install -r requirements-backend.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy code and resources
COPY main_fastapi.py .
COPY data/frequency_dictionary_en_82_765.txt ./data/frequency_dictionary_en_82_765.txt

EXPOSE 8000

CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]