# Breast Cancer MLOps Pipeline

**MLOps Assignment 2  Containerizing an ML Prediction Service with Docker, Docker Compose, Docker Hub, and DVC**

| Student Detail | Information |
|---|---|
| Student Name | Noor ul Huda |
| Roll Number | F230035 |
| Class | BS AI-7A |
| Dataset | Wisconsin Breast Cancer Dataset |
| Selected Model | Logistic Regression |

## Project Overview

This project implements an end-to-end MLOps pipeline for breast cancer prediction using the Wisconsin Breast Cancer dataset.

The project demonstrates dataset versioning and pipeline management with **DVC**, machine learning model training and evaluation, a **FastAPI** prediction service, containerization using **Docker**, service orchestration through **Docker Compose**, and image publishing through **Docker Hub**.

Redis is integrated to maintain a persistent prediction counter. The final application can be deployed using the published Docker image without building the image locally.

## Project Links

- **GitHub Repository:** https://github.com/f230035-Huda/breast-cancer-mlops-pipeline
- **Docker Hub Repository:** https://hub.docker.com/r/huda2503/breast-cancer-triage-api

## Technologies Used

- Python
- Scikit-learn
- Pandas and NumPy
- FastAPI
- Redis
- Docker
- Docker Compose
- Docker Hub
- DVC
- Git and GitHub

## Project Architecture

The project follows this workflow:

1. Dataset preparation and validation.
2. Dataset versioning using DVC.
3. Model training and evaluation.
4. Model serving through FastAPI.
5. Application containerization using Docker.
6. Redis integration for prediction counting.
7. Service orchestration using Docker Compose.
8. Docker image publishing to Docker Hub.
9. Clean-room deployment using the published image.

## Dataset

The project uses the Wisconsin Breast Cancer dataset available through Scikit-learn.

- **Total original samples:** 569
- **Input features:** 30
- **Target:** Breast cancer classification
- **Classes:** Malignant and Benign

DVC is used to track dataset versions and manage data artifacts.

Two dataset versions were demonstrated:

| Version | Number of Samples |
|---|---:|
| V1 | 569 |
| V2 | 539 |

Version 2 was created after removing 30 records to demonstrate dataset versioning and reproducibility.

## Machine Learning Model

Multiple classification models were evaluated, and Logistic Regression was selected for the prediction service.

Reported Logistic Regression results:

| Metric | Cross-Validation | Test Set |
|---|---:|---:|
| Accuracy | 96.99% | 97.22% |
| Malignant Recall | 95.15% | 97.56% |
| ROC-AUC | 99.05% | 99.56% |

These values are the results recorded during the assignment experiments.

## DVC Workflow

DVC is used for dataset and machine learning pipeline versioning.

### Initialize DVC

```bash
dvc init
```

### Configure the Local Remote

```bash
dvc remote add -d localremote dvc-remote
```

### Pull Tracked Data

```bash
dvc pull
```

### Reproduce the Pipeline

```bash
dvc repro
```

### Check Pipeline Status

```bash
dvc status
```

**Note:** The configured remote in this assignment is local. A local DVC remote is useful for demonstrating versioning and restoration on the same environment, but it is not automatically accessible to other users. For a fully reproducible deployment on another machine, configure an accessible remote or provide the required data separately.

## Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/f230035-Huda/breast-cancer-mlops-pipeline.git
cd breast-cancer-mlops-pipeline
```

### 2. Create and Activate a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / Ubuntu:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Restore Data and Run the Pipeline

```bash
dvc pull
dvc repro
```

The commands above assume that the configured DVC remote is accessible from your environment.

## Docker Deployment

The application is containerized using Docker. Both a basic Docker build and an optimized production image were demonstrated.

### Build the Production Image

Use the production Dockerfile included in the repository:

```bash
docker build -f Dockerfile.prod -t breast-cancer-triage-api:1.0.1 .
```

### Run the Container

```bash
docker run -d --name breast-cancer-api -p 8000:8000 breast-cancer-triage-api:1.0.1
```

### Check Running Containers

```bash
docker ps
```

### View Container Logs

```bash
docker logs breast-cancer-api
```

*Use the actual production Dockerfile filename in the build command if it differs in the repository.*

## Docker Hub

The production application is published on Docker Hub.

**Repository:** [huda2503/breast-cancer-triage-api](https://hub.docker.com/r/huda2503/breast-cancer-triage-api)

Published tags:

| Tag | Description |
|---|---|
| `1.0.0` | Earlier application image |
| `1.0.1` | Application with Redis prediction counter |
| `latest` | Latest published application image |

### Pull the Latest Redis-Integrated Image

```bash
docker pull huda2503/breast-cancer-triage-api:1.0.1
```

## Docker Compose Deployment

The application uses Docker Compose to run the FastAPI service and Redis together.

The Compose configuration includes:

- FastAPI prediction API
- Redis service
- Redis named volume for persistent data
- Service health checks
- API port mapping
- Published Docker image

### Start Services

From the directory containing the Compose configuration:

```bash
docker compose pull
docker compose up -d
```

### Check Services

```bash
docker compose ps
```

### View Logs

```bash
docker compose logs
```

### Stop Services

```bash
docker compose down
```

To remove the Redis named volume as well:

```bash
docker compose down -v
```

**Important:** Removing the volume also removes the persisted Redis data.

## API Endpoints

The FastAPI service provides endpoints for checking service health, viewing model information, and making predictions.

| Endpoint | Purpose |
|---|---|
| `/health` | Check API health |
| `/model/info` | Retrieve model information |
| `/predict` | Generate a breast cancer prediction |
| `/stats` | Retrieve prediction statistics |

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

If using the assignment's clean-room Compose configuration, the host port is `8006`:

```text
http://localhost:8006/docs
```

### Example API Testing

Health check:

```bash
curl http://localhost:8006/health
```

Prediction statistics:

```bash
curl http://localhost:8006/stats
```

For prediction requests, use the `/docs` interface and provide the input features in the request format defined by the API.

## Redis Prediction Counter

Redis is integrated with the FastAPI application to maintain the prediction count.

The `/stats` endpoint reports the prediction counter and its data source.

During the clean-room deployment test:

- Initial prediction count: `0`
- One successful prediction was submitted.
- Final prediction count: `1`
- Counter source: Redis

This demonstrates communication between the API and Redis service.

## Clean-Room Deployment

A clean-room deployment was performed to verify that the published Docker image could run without relying on the locally built production image.

The verification process included:

1. Creating a separate deployment directory.
2. Using the Docker Hub image in the Compose configuration.
3. Removing the existing local production image.
4. Pulling the published image from Docker Hub.
5. Starting the API and Redis services.
6. Checking container health.
7. Testing the prediction endpoint.
8. Verifying that Redis updated the prediction counter.
9. Comparing the local image digest with the published Docker Hub digest.

The published `1.0.1` image digest verified during testing was:

```text
sha256:4668ce2831650d097ee6d9b292dcb5cec195f9fb92d34d97c3133603e2e99ed2
```

## Key Learning Outcomes

This assignment demonstrates practical understanding of:

- Dataset and pipeline versioning using DVC.
- Reproducible machine learning workflows.
- Model training and evaluation.
- Building REST APIs using FastAPI.
- Docker images and container lifecycle management.
- Production-oriented Docker image optimization.
- Multi-service deployment using Docker Compose.
- Redis integration and persistent data.
- Publishing and pulling images from Docker Hub.
- Testing deployments in a clean environment.

## Conclusion

The project demonstrates an end-to-end containerized machine learning prediction service. DVC provides dataset and pipeline versioning, FastAPI exposes the trained model through REST endpoints, Docker packages the application, Docker Compose manages the API and Redis services, and Docker Hub provides a published image for deployment.

The clean-room deployment test confirms that the published Redis-integrated application image can be pulled and executed independently of the original local build environment.

---

**Developed by:** Noor ul Huda  
**Roll Number:** F230035  
**Course:** MLOps — Assignment 2  
**Class:** BS AI-7A
