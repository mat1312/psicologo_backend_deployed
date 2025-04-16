#!/bin/bash
echo "Starting Psicologo Virtuale API..."
export PORT=${PORT:-8000}
gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT 