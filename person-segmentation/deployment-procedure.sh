# Prepare directories and files
# Create project directory and go to it
mkdir project_path
cd project_path
# Create 'Dockerfile' file
touch Dockerfile
# Create 'app' directory and 'main.py' file into
mkdir app
touch app/main.py

# Prepare python environment
# Create python virtual environment
python3 -m venv env
# Activate environment
source env/bin/activate
# Upgrade pip
pip install -U pip
# Install requirements
pip install -r requirements.txt


# Run application locally
uvicorn app.main:app --reload
