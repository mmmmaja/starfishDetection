# Training

## Environment
Create a dedicated environment to keep track of the packages for the project
```bash
conda create --name starfish-env python=3.11
conda activate starfish-env
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install -e .
```

## Data
Download the data for the project from our Google Cloud Bucket, which requires the Google Cloud SDK.
```bash
invoke download-data
```

## Train
Train with default arguments
```bash
train
```
Train with data downloaded from the bucket rather than accessing it directly from the cloud
```bash
train data.data_from_bucket=false
```

## Profiling
Train with profiling
```bash
train profiling=True
```
Profile forward pass
```bash
invoke profile-forward-pass
```

## Docker
Build the training dockerfile into a Docker image
```bash
invoke build-train-image
```
Run a container spawned from the docker image
```bash
invoke run-train-image
```

## Vertex AI
Get the Docker image built from `train.dockerfile` in the Artifact Registry on Google Cloud, for example by creating a trigger and using the `cloudbuild.yaml` file. Then you can train a model using that Docker image through the Vertex AI service. This also automatically logs the training to Wandb if your API key has been stored as a secret in Google Cloud.
```bash
invoke train-vertex
```

## Wandb
Login
```bash
wandb login
```
Hyperparameter sweep
```bash
invoke sweep
wandb agent ENTITY/PROJECT_NAME/AGENT_ID
```

## Tests
Run all tests and calculate coverage
```bash
invoke test
```
Run data tests
```bash
invoke test-data
```
Run model tests
```bash
invoke test-model
```
