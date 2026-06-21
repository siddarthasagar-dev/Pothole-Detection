FROM python:3.11-slim

# Install minimal system libraries required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY . .

# Expose the application port
EXPOSE 5001

# Run the Flask backend
CMD ["python", "backend/app.py"]
