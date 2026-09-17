FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Copy the entire AnyCall project
COPY . .

# Expose port 7860 (standard for Hugging Face Spaces / Render)
EXPOSE 7860

# Launch the FastAPI server
CMD ["python", "-m", "anycall.portal.server", "--host", "0.0.0.0", "--port", "7860"]
