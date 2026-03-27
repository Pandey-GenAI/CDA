# Career Discovery Assistant – Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit default port
EXPOSE 8502

# Launch the app
CMD ["streamlit", "run", "main.py", "--server.port=8502", "--server.address=0.0.0.0"]
