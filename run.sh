#!/bin/bash

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment and install dependencies
echo "Installing dependencies..."
source .venv/bin/activate
uv pip install -r requirements.txt

echo ""
echo "Starting GLD Price Distribution Analyzer..."
echo "Opening browser at http://localhost:8501"
echo ""

# Run the Streamlit app with auto-reload enabled
streamlit run gld_app.py --server.runOnSave true