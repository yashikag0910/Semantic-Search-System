# lightweight python image
FROM python:3.10-slim

# set working directory
WORKDIR /app

# set huggingface cache location inside container
ENV HF_HOME=/app/.cache

# copy requirements first (better caching)
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# download embedding model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# copy project files
COPY . .

# expose API port
EXPOSE 8000

# start FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]