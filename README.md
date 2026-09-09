# Iris Flower Classification API 🌸

An end-to-end Machine Learning project that predicts the species of an Iris flower based on its sepal and petal measurements.

The project demonstrates the complete workflow from model training to API development, containerization, and CI/CD automation.

## Project Overview

The model classifies Iris flowers into three species:

- Setosa
- Versicolor
- Virginica

The prediction is based on four input features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## Tech Stack

- Python
- pandas
- scikit-learn
- Random Forest
- FastAPI
- Uvicorn
- Docker
- Git & GitHub
- GitHub Actions
- Docker Hub

## Project Workflow

```text
Iris Dataset
     ↓
Data Preparation
     ↓
Train/Test Split
     ↓
Random Forest Model
     ↓
Model Evaluation
     ↓
Save Trained Model (model.pkl)
     ↓
FastAPI REST API
     ↓
Docker Image
     ↓
GitHub Actions CI/CD
     ↓
Docker Hub
```

## Machine Learning Model

The Iris dataset is loaded using scikit-learn.

The dataset contains 150 samples and four input features.

The data is divided into:

- 80% training data
- 20% testing data

A Random Forest Classifier with 100 decision trees is used for classification.

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

The trained model is saved as `model.pkl` using Joblib so that it can be loaded for predictions without retraining every time.

## API

FastAPI is used to expose the trained machine learning model through a REST API.

### Prediction Endpoint

```text
POST /predict
```

Example request:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example response:

```json
{
  "prediction": "setosa"
}
```

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

## Docker

The application is containerized using Docker so that the model and API can run consistently across different environments.

Build the Docker image:

```bash
docker build -t iris-ml-api .
```

Run the container:

```bash
docker run -d --name iris-container -p 8000:8000 iris-ml-api
```

The API can then be accessed at:

```text
http://localhost:8000
```

## CI/CD Pipeline

GitHub Actions is used to automate the CI/CD workflow.

Whenever code is pushed to the `main` branch, the pipeline automatically:

1. Checks out the source code
2. Sets up Python 3.11
3. Installs project dependencies
4. Trains the machine learning model
5. Builds the Docker image
6. Logs in to Docker Hub securely using GitHub Secrets
7. Pushes the latest Docker image to Docker Hub

```text
Code Change
     ↓
Git Push
     ↓
GitHub
     ↓
GitHub Actions
     ↓
Install Dependencies
     ↓
Train Model
     ↓
Build Docker Image
     ↓
Push Image to Docker Hub
```

## Project Structure

```text
iris-ml-cicd/
│
├── .github/
│   └── workflows/
│       └── ci.yaml
│
├── Dockerfile
├── main.py
├── train.py
├── model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Start the FastAPI application:

```bash
uvicorn main:app --reload
```

Open the Swagger documentation:

```text
http://localhost:8000/docs
```

## Future Improvements

- Deploy the containerized application to AWS
- Add automated model/API tests to the CI pipeline
- Add Docker image versioning
- Add model performance monitoring

## Author

**Pooja Basavaraju**

M.Sc. Data Science  
Interested in Machine Learning Engineering, AI Engineering and MLOps
