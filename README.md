# NephroEx Backend

This repository contains the backend server code for NephroEx, a Computer-Aided Diagnosis (CAD) tool for interpreting diuretic renography in urinary tract obstruction (UTO) with Explainable AI analysis capabilities.

## Related Repositories

- [NephroEx Frontend](https://github.com/AI-Diagnostic-Assistant/NephroEx-App)
- [NephroEx ML Environment](https://github.com/AI-Diagnostic-Assistant/ML-Environment)

## Overview

This Flask-based API server provides the machine learning and image processing capabilities that power the NephroEx diagnostic system. The server processes DICOM files containing diuretic renography data, applies AI models to detect kidney obstructions, and provides explainable AI outputs to assist in medical diagnosis.

## Features

- DICOM file processing and analysis
- AI-powered classification of urinary tract obstruction
- Explainable AI (XAI) capabilities for transparent diagnostic reasoning
- Image segmentation for kidney region detection using U-Net
- Dynamic generation of renogram curves and analysis
- Integration with Supabase for model storage and authentication
- Docker support for easy deployment

## Technologies

- [Flask](https://flask.palletsprojects.com/): Web framework
- [PyTorch](https://pytorch.org/): Deep learning framework for neural networks (U-Net, CNN)
- [PyDICOM](https://pydicom.github.io/): For DICOM file processing
- [OpenCV](https://opencv.org/): Image processing
- [SHAP](https://shap.readthedocs.io/): For explainable AI features
- [Supabase](https://supabase.com): Authentication and storage
- [Google Generative AI](https://ai.google.dev/): Advanced AI capabilities

## Models

The server uses the following machine learning models:

- U-Net model for kidney segmentation
- Stacking ensemble model for classification

## Prerequisites

Before running this project, make sure you have:

- Python 3.11 or later
- pip (Python package manager)
- Docker Desktop (optional, for containerized deployment) - must be installed and running for Docker deployment steps to work

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
SUPABASE_PROJECT_URL=your_supabase_project_url
SUPABASE_SERVICE_ACCOUNT_KEY=your_supabase_service_role_key
SUPABASE_USER_ACCOUNT_KEY=your_supabase_anon_key
GOOGLE_API_KEY=your_google_api_key
```

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/AI-Diagnostic-Assistant/NephroEx-Server.git
cd NephroEx-Server
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the development server:

```bash
python run.py
```

The API will be available at http://localhost:8080.

4. **Alternative**: Run using Docker:

```bash
# Build the Docker image
docker build -t nephroex-server .

# Run the container
docker run -p 8080:8080 nephroex-server
```

For more details on Docker deployment, see the [Docker Deployment](#docker-deployment) section below.

## API Endpoints

### Health Check

- **Endpoint**: `GET /` or `/index`
- **Description**: Simple health check endpoint to verify the server is running
- **Response**: Returns a status message

### Classification Endpoint

- **Endpoint**: `POST /classify`
- **Description**: Main endpoint for DICOM analysis and classification
- **Parameters**:
  | Parameter | Type | Required | Description |
  | --- | --- | --- | --- |
  | `patientId` | String | Yes | ID of an existing patient in the system |
  | `file` | File (DICOM) | Yes | DICOM file from the diuretic renography dataset |
  | `diuretic` | Integer | Yes | Time (in minutes) when diuretic was administered |
- **Request Format**: `multipart/form-data`
- **Response**:
  - **Status**: 200 OK on success
  - **Body**: JSON object containing:
    ```json
    {
      "message": "classify endpoint",
      "id": "<report_id>"
    }
    ```
  - The `id` is the unique identifier for the created report containing all analysis results

## Docker Deployment

1. Build the Docker image:

```bash
docker build -t nephroex-server .
```

2. Run the container:

```bash
docker run -p 8080:8080 nephroex-server
```

## License

This project is proprietary and is not available for public use, reproduction, or distribution. This repository is maintained solely for demonstration and portfolio purposes as part of a master's thesis project at NTNU. All rights reserved © 2025.

## Contact

### Magnus Rosvold Farstad

NTNU - Norwegian University of Science and Technology  
Email: magnus.r.farstad@gmail.com  
LinkedIn: [Magnus Rosvold Farstad](https://www.linkedin.com/in/magnusrosvoldfarstad/)  
GitHub: [@magnus-farstad](https://github.com/Magnus-Farstad)

### Simen Klemp Wergeland

NTNU - Norwegian University of Science and Technology  
Email: simenk2312@gmail.com  
LinkedIn: [Simen Klemp Wergeland](https://www.linkedin.com/in/simen-klemp-wergeland-b684411ba/)  
GitHub: [@SimenKlemp](https://github.com/SimenKlemp)
