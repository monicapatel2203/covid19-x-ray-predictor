# Get python image
FROM python:3.9

# Copy the content 
COPY . .

# Set Workdir
WORKDIR /

# install the requirements
RUN pip install --no-cache-dir -r ./requirements.txt

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]