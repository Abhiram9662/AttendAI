# AI-Powered Attendance Management System
Phase 1 starter scaffold for CSE471. This is a runnable development foundation, not the finished face-recognition system.

## Requirements
- Windows 11
- Python 3.10 or 3.11
- Node.js 20 LTS+
- VS Code

## Backend (PowerShell)
```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API docs: http://127.0.0.1:8000/docs

## Frontend (second VS Code terminal)
```powershell
cd frontend
npm install
npm run dev
```
Open the URL printed by Vite (normally http://localhost:5173).

Camera access is allowed on localhost. Production camera access requires HTTPS.
The current API detects faces using OpenCV's Haar cascade only. It does NOT identify people or mark attendance yet.
