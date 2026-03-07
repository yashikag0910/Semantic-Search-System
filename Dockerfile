# use lightweight python image
FROM python:3.10-slim

# set working directory
WORKDIR /app

# copy requirements first (better docker caching)
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# download the embedding model during build
# this prevents downloading it every time the container starts
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# copy project files
COPY . .

# expose fastapi port
EXPOSE 8000

# start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]