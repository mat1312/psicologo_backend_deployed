# Psicologo Virtuale Backend

## Installation

1. Install Python 3.8+ and pip
2. Install dependencies:
```
pip install -r requirements.txt
```

## Running the server

### On Windows PowerShell
PowerShell on Windows doesn't support the `&&` operator for command chaining. Use these commands instead:

```powershell
# Navigate to the backend directory
cd backend

# Start the server
python main.py
```

### On Bash/Linux/macOS Terminal
```bash
cd backend && python main.py
```

## API Endpoints

- `/api/chat` - Main chat endpoint
- `/api/patient-chat` - Patient-specific chat endpoint
- `/api/recommend-resources` - Resource recommendations
- `/patients/{patient_id}/recommendations` - Patient-specific recommendations
- `/api/public/recommendations` - Public recommendations (no auth required) 