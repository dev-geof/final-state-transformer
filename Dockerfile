FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /final-state-transformer

# Copy the rest of the application code into the container
COPY . .

# Install the dependencies from requirements.txt
RUN if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi

# Install the final-state-transformer package
RUN pip install .

# Default command to keep the container running
CMD ["tail", "-f", "/dev/null"]
