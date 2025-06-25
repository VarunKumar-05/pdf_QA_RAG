# 📄 PDF QA RAG

A powerful, production-ready PDF Question-Answering (QA) system using Retrieval-Augmented Generation (RAG). This project lets you query PDFs with natural language and get precise, context-aware answers, leveraging state-of-the-art NLP models.

---

## 🚀 Features

- **Ask Anything:** Query one or multiple PDFs with natural language.
- **Retrieval-Augmented Generation (RAG):** Combines retrieval of relevant passages & generative AI for accurate answers.
- **Scalable Backend:** Designed for production & large-scale document sets.
- **Easy to Extend:** Modular code—swap models, add new features, or integrate with your stack.
- **Docker Support:** One command to run anywhere.

---

## 🛠️ Tech Stack

- **Backend:** Python (FastAPI or Flask)
- **NLP:** HuggingFace Transformers, Sentence Transformers
- **Vector Store:** FAISS / ChromaDB / Pinecone
- **PDF Parsing:** PyPDF2 / pdfplumber
- **Containerization:** Docker

---

## 📦 Installation

### 1. Clone the Repo

```bash
git clone https://github.com/VarunKumar-05/pdf_QA_RAG.git
cd pdf_QA_RAG
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Build & Run with Docker

```bash
docker build -t pdf_qa_rag .
docker run -p 8000:8000 pdf_qa_rag
```

---

## 💡 Usage

### 1. Upload PDFs

Place your PDF files in the designated folder (e.g., `data/`).

### 2. Start the API

```bash
python main.py
```

### 3. Query PDFs

Send a POST request to the `/ask` endpoint:

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main topic of the document?", "pdf": "sample.pdf"}'
```

Or use the included web UI (if available).

---

## 🧩 Example

> **Q:** *"Summarize the key findings on page 5."*  
> **A:** “Page 5 discusses...”

---

## ⚙️ Configuration

- **Model selection:** Change the model in `config.py` (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Store:** Switch between FAISS, ChromaDB, or Pinecone in `vectorstore.py`.
- **Chunk size, max docs, etc.:** Tunable in `config.py`.

---

## 📝 Project Structure

```
pdf_QA_RAG/
├── data/                # PDF files to be processed
├── main.py              # Entry point/API server
├── modules/             # Core modules (retriever, reader, etc.)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Containerization config
└── README.md            # This file
```

---

## 🛣️ Roadmap

- [ ] Add web frontend for interactive QA
- [ ] Support for more file formats (Word, HTML)
- [ ] Multi-lingual support
- [ ] Authentication & user management

---

## 🤝 Contributing

Pull requests, issues, and feature requests are welcome!  
See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 🛡️ License

[MIT License](LICENSE)

---

## 📬 Contact

**Varun Kumar**  
[GitHub Profile](https://github.com/VarunKumar-05)

---

> **Star ⭐ the repo if you find it useful!**
