FROM python:3.9-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install tensorflow==2.18.0
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application
COPY . .

# Make port available to the world outside this container
ENV PORT=7860
EXPOSE 7860

# Run the application
CMD uvicorn main:app --host 127.1.1.1 --port $PORT