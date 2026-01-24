# 1. Base Image: Start with a lightweight Linux with Python 3.11 installed
# "slim" means it has fewer pre-installed tools, keeping it smaller.
FROM python:3.11-slim

# 2. System Dependencies: Install tools needed to compile C++ libraries
# llama-cpp-python and faiss often need a C++ compiler (build-essential).
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Working Directory: Create a folder inside the container to hold our app
WORKDIR /app

# 4. Copy Requirements: Move just the requirements file first
# Docker caches steps. If requirements.txt hasn't changed, it skips the slow install step.
COPY requirements.txt .

# 5. Install Python Dependencies
# --no-cache-dir keeps the image smaller by not saving downloaded wheels
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Code: Move the rest of your source code into the container
COPY . .

# 7. Environment Variables: Set defaults (can be overridden)
ENV PORT=8000
ENV HOST=0.0.0.0

# 8. Command: What happens when you run the container?
# We use the command we discussed earlier to start the server.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]