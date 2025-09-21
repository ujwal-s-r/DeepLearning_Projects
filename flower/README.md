```
python -m venv venv
venv/Scripts/activate

pip install -r requirements.txt

cd backend 
python app.py

seperate terminal

# Start frontend (new terminal)
cd frontend && python -m http.server 8000
```