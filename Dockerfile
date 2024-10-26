# Dockerfile

FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project
COPY . .

# Expose ports
EXPOSE 5001 5002

# Run the CLI (adjust as necessary)
CMD ["python", "cli.py", "--git_url", "https://github.com/your/repository.git", "--model", "gpt-neo"]