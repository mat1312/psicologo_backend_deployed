#!/bin/bash
echo "Starting Psicologo Virtuale API..."
uvicorn main:app --host 0.0.0.0 --port $PORT 