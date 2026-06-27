# Heart Rate Monitor API
A backend API for the Heart Rate Monitor mobile app (iOS & Android), managing user profiles, heart rate measurements, and stress analysis using machine learning.

## Disclaimer
This is not a medical app. The stress analysis and HRV insights are intended for entertainment and educational purposes only.

## Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Getting Started](#getting-started)

## Features
- **Authentication & Profile**: Secure user registration, login via JWT, and profile management.
- **Heart Rate Tracking**: Store, retrieve, and batch-delete heart rate records.
- **Stress Prediction (ML)**: Uses a trained machine learning model to estimate stress levels from HRV features.
- **Explainable AI (SHAP)**: Provides SHAP feature contributions to explain which signals drive the stress prediction.
- **Retrieval-Augmented Generation (RAG)**: Integrates with a local vector database to retrieve relevant HRV medical knowledge.
- **LLM Insights**: Leverages an external LLM to provide readable context and explanations combining the ML prediction, SHAP drivers, and retrieved medical knowledge.

## Tech Stack
- **FastAPI**: High-performance async web framework for building the REST API.
- **SQLAlchemy + PostgreSQL**: Robust ORM for database management.
- **Scikit-Learn + SHAP**: Machine learning inference and model explainability.
- **Qdrant + Sentence-Transformers**: Vector database and embeddings for the local RAG pipeline.
- **OpenAI**: LLM integration for natural language health explanations.
- **Docker**: Included `Dockerfile` for easy containerized deployment.

## Project Architecture
**`main.py`**: FastAPI application setup, global exception handlers, routing, middleware (CORS), database initialization, and startup logic.

**API Layer (api/)**: Responsible for HTTP endpoints and request handling.
- `api/routes/auth.py`: Registration, login, and logout endpoints.
- `api/routes/profile.py`: User profile retrieval and updates.
- `api/routes/heart_rate.py`: Heart rate CRUD operations.
- `api/routes/stress.py`: Stress prediction endpoints.
- `api/dependencies.py`: Shared FastAPI dependencies such as database sessions and authentication helpers.

**Service Layer (services/)**: Contains business logic and keeps routes thin.
- `auth_service.py`: User registration and authentication logic.
- `profile_service.py`: Profile management operations.
- `heart_rate_service.py`: Heart rate storage, retrieval, and deletion logic.
- `stress_service.py`: Orchestrates ML inference, SHAP explainability, RAG retrieval, and LLM explanations.
- `stress_analysis.py`: Coordinates AI components and aggregates final response.
- `stress_model.py`: Loads trained model artifacts, performs inference, and generates SHAP explanations.
- `rag_pipeline.py`: Retrieves relevant HRV medical knowledge using embeddings and vector search.
- `llm_explainer.py`: Uses LLM to generate human-readable explanations.

**Database Layer (db/)**: Handles persistence and ORM models.
- `database.py`: SQLAlchemy engine and session management.
- `models.py`: Database models (User, HeartRateRecord).

**Schema Layer (schemas/)**: Defines API contracts and validation models.
- `schemas.py`: Pydantic request and response schemas.

**Core Layer (core/)**: Application-wide configuration and security utilities.
- `config.py`: Environment configuration and settings management.
- `security.py`: Password hashing and JWT token utilities.

**Utility Layer (utils/)**:
- `openai.py`: OpenAI client initialization and helper methods.

**Knowledge Base (knowledge_base/)**: Contains local HRV medical knowledge used by RAG.
- `hrv_medical_knowledge.json`: Curated HRV and stress-related medical information.

**Request Flow Example - Stress Prediction Request**
Mobile App → FastAPI Route (`stress.py`) → StressService → StressModel → RAG Pipeline → LLM Explainer → JSON Response

## Getting Started
**Prerequisites**:
- Python 3.9+
- PostgreSQL
- OpenAI API Key (for LLM explanations)

**Setup**:
```bash
git clone https://github.com/vesc0/heart-rate-monitor-api.git
cd heart-rate-monitor-api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Environment Variables**:
Configure your `.env` file with the following variables:

**Database**
```env
DATABASE_URL=your_database_connection_string
```

**Authentication**
```env
SECRET_KEY=your_secret_key
ALGORITHM=your_jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES=your_token_expiration_time
```

**LLM / AI Configuration**
```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_llm_provider_base_url
OPENAI_MODEL=your_model_name
```

**Run Server**:
```bash
uvicorn main:app --reload
```
