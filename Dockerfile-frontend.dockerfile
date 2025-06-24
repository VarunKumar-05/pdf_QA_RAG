FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements-frontend.txt .
RUN pip install --upgrade pip && pip install -r requirements-frontend.txt

# Copy app code
COPY frontend_streamlit.py .

EXPOSE 8501

CMD ["streamlit", "run", "frontend_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]