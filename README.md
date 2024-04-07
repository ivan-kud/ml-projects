# My works of Machine Learning

## Service with some ML models

The service where you could apply machine learning models to solve some problems, such as:

- Classification of handwritten digit
- Instance segmentation
- Classification of sentiment polarity
- Search for similar images

**Technical description:**

- Backend: FastAPI
- Frontend: HTML, CSS, JavaScript
- Cloud service: DigitalOcean (GCP was used before 02.2022)

## Fine-tuning of OpenAI model

OpenAI provides the ability to fine-tune their base models in order to adapt them to a specific task. Here I implemented fine-tuning of the "ada" base model for binary text classification task.

## Text classification with catboost and fastText

Binary classification problem: to determine if there is contact information in the advertisement.

## Hierarchical classification

A lot of new products arrive in the marketplace every day and each of them must be assigned to a specific category in the category tree (there are more than 1000 categories). It takes a lot of effort and time, so I want to learn how to predict a category based on short description and parameters of products.

**Technical description:**

1. Approaches: Local Classifier per Node (LCN) and Local Classifier per Parent Node (LCPN);
2. Base model: Random Forest;
3. Dataset: 354316 instances;
4. Libraries: hiclass, scikit-learn, networkx.

## Some other notebooks

Notebooks from SkillFactory course and test tasks.
