deactivate
rm -r venv
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install \
    accelerate \
    evaluate \
    gensim \
    ipywidgets \
    jupyterlab \
    matplotlib \
    nltk \
    notebook \
    pandas \
    pyarrow \
    requests \
    scikit-learn \
    tensorboard \
    torch-lr-finder \
    tqdm
