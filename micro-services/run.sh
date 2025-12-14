#!/bin/bash

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/Scripts/activate
fi

# Install requirements
pip install -r requirements.txt

# Create uploads directory
# mkdir -p uploads

# Run the application
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload