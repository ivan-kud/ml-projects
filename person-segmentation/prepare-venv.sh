# Prepare directories and files
mkdir person-segmentation
cd person-segmentation
touch Dockerfile
mkdir app
touch app/main.py
touch app/__init__.py

# Prepare python environment
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Run application locally
uvicorn app.main:app --reload
