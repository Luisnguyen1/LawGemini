# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Set environment variables for the cache directory (use a writable directory)
ENV TRANSFORMERS_CACHE=/home/user/.cache/huggingface

# Create and set the cache directory so the user can write to it
RUN mkdir -p /home/user/.cache/huggingface && chown -R user:user /home/user/.cache

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Upgrade pip to the latest version to avoid issues with outdated pip
RUN pip install --no-cache-dir --upgrade pip

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download any necessary assets
RUN mkdir content
ADD --chown=user https://huggingface.co/datasets/manhteky123/LawVietnamese/resolve/main/data.csv content/data.csv
ADD --chown=user https://huggingface.co/datasets/manhteky123/LawVietnamese/resolve/main/faiss_index.bin content/faiss_index.bin
ADD --chown=user https://huggingface.co/datasets/manhteky123/LawVietnamese/resolve/main/vectors.npy content/vectors.npy

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]
