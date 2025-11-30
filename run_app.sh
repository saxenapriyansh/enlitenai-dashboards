#!/bin/bash

# Seizure Analytics Dashboard - Quick Launch Script

echo "🏥 Launching Seizure Analytics Dashboard..."
echo ""

# Check if requirements are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

echo "🚀 Starting Streamlit application..."
echo ""
echo "➡️  The dashboard will open in your browser at http://localhost:8501"
echo "➡️  Press Ctrl+C to stop the server"
echo ""

streamlit run app.py

