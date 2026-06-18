# Steps to Run — Water Potability Classifier

> Full PowerShell commands to run this project locally, plus AWS deployment instructions.

---

## Part 1 — Run Locally (PowerShell)

### Prerequisites

| Tool | Minimum Version | Check Command |
|------|-----------------|---------------|
| Python | 3.10+ | `python --version` |
| pip | latest | `pip --version` |
| Git | any | `git --version` |
| Docker *(optional)* | 20+ | `docker --version` |

---

### Step 1: Clone the Repository

```powershell
git clone https://github.com/<your-username>/ML-project-waterPotability.git
cd ML-project-waterPotability
```

### Step 2: Create & Activate a Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Train All Models (generates artifacts/)

```powershell
python train.py
```

> This takes ~3-5 minutes. It trains 4 models with Optuna HPO, selects the best one, and saves the model + plots + metrics to `artifacts/`.

### Step 5: Run the FastAPI Server

```powershell
uvicorn api.main:app --reload --port 8000
```

### Step 6: Open in Browser

```powershell
Start-Process "http://127.0.0.1:8000"
```

- **Web UI** → `http://127.0.0.1:8000`
- **API Docs (Swagger)** → `http://127.0.0.1:8000/docs`
- **Health Check** → `http://127.0.0.1:8000/health`

### Step 7: Test the `/predict` Endpoint (optional)

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"ph":7.0,"Hardness":200,"Solids":20000,"Chloramines":7.0,"Sulfate":333,"Conductivity":400,"Organic_carbon":14,"Trihalomethanes":66,"Turbidity":4.0}'
```

### Step 8: Deactivate Virtual Environment (when done)

```powershell
deactivate
```

---

### Alternative — Run with Docker Locally

```powershell
# Build the Docker image
docker build -t water-potability-api .

# Run the container
docker run -p 8000:8000 water-potability-api

# Or use Docker Compose
docker-compose up --build
```

Then open `http://127.0.0.1:8000` in your browser.

---

---

## Part 2 — Deploy on AWS (Elastic Beanstalk + ECR)

### Prerequisites

| Tool | Install Command / Link |
|------|------------------------|
| AWS CLI v2 | [Install Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) |
| Docker | [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| EB CLI *(optional)* | `pip install awsebcli` |
| AWS Account | With IAM credentials configured |

---

### Step 1: Configure AWS CLI

```powershell
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Default region: us-east-1
# Default output: json
```

> **If using AWS Learner Lab**: copy-paste the temporary credentials from the Learner Lab "AWS Details" page into `~/.aws/credentials`.

### Step 2: Create an ECR Repository

```powershell
aws ecr create-repository --repository-name water-potability-api --region us-east-1
```

Save the `repositoryUri` from the output (e.g., `864423586496.dkr.ecr.us-east-1.amazonaws.com/water-potability-api`).

### Step 3: Authenticate Docker with ECR

```powershell
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.us-east-1.amazonaws.com
```

Replace `<your-account-id>` with your 12-digit AWS account ID.

### Step 4: Build, Tag & Push Docker Image

```powershell
# Build the image
docker build -t water-potability-api .

# Tag it for ECR
docker tag water-potability-api:latest <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/water-potability-api:latest

# Push to ECR
docker push <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/water-potability-api:latest
```

### Step 5: Update `Dockerrun.aws.json`

Make sure the `Image.Name` field in `Dockerrun.aws.json` matches your ECR URI:

```json
{
  "AWSEBDockerrunVersion": "1",
  "Image": {
    "Name": "<your-account-id>.dkr.ecr.us-east-1.amazonaws.com/water-potability-api:latest",
    "Update": "true"
  },
  "Ports": [
    {
      "ContainerPort": 8000
    }
  ]
}
```

### Step 6: Create Elastic Beanstalk Application (Console)

1. Go to **AWS Console → Elastic Beanstalk → Create Application**
2. **Application name**: `water-potability-api`
3. **Platform**: Docker
4. **Platform branch**: Docker running on 64bit Amazon Linux 2023
5. **Application code**: Upload the `Dockerrun.aws.json` file
6. Click **Create environment**

### Step 6 (Alternative): Create via EB CLI

```powershell
# Initialize EB in the project directory
eb init water-potability-api --platform docker --region us-east-1

# Create the environment and deploy
eb create water-potability-env --single --instance-type t3.micro
```

### Step 7: Grant ECR Access to Elastic Beanstalk

Elastic Beanstalk needs permission to pull from ECR. Attach the `AmazonEC2ContainerRegistryReadOnly` policy to the Beanstalk EC2 instance role:

```powershell
aws iam attach-role-policy `
  --role-name aws-elasticbeanstalk-ec2-role `
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
```

### Step 8: Verify Deployment

```powershell
# Get the environment URL
eb status

# Or open it directly
eb open
```

Your API will be live at: `http://water-potability-env.<region>.elasticbeanstalk.com`

### Step 9: Clean Up AWS Resources (when done)

```powershell
# Terminate the Elastic Beanstalk environment
eb terminate water-potability-env --force

# Delete the ECR repository
aws ecr delete-repository --repository-name water-potability-api --region us-east-1 --force
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Make sure your venv is activated: `.\venv\Scripts\Activate` |
| `FileNotFoundError: best_model.joblib` | Run `python train.py` first to generate artifacts |
| Docker build fails | Ensure Docker Desktop is running |
| ECR push denied | Re-run `aws ecr get-login-password ...` (tokens expire after 12 hours) |
| EB deploy fails with "image not found" | Verify ECR URI in `Dockerrun.aws.json` matches your pushed image |
| Learner Lab credentials expired | Re-copy credentials from the Learner Lab "AWS Details" page |
