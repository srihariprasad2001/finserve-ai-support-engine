# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY finserve_main_app.py .

# Expose port 8000
EXPOSE 8000

# Set environment variable for production
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "finserve_main_app:app", "--host", "0.0.0.0", "--port", "8000"]
