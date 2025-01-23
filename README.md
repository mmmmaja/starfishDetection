# Starfish Detection
This is the repository for the project in DTU's Winter 2025 course 02476 Machine Learning Operations. Check out our documentation site at [Starfish Detection](https://mmmmaja.github.io/starfishDetection/).

## Project description

**Goal**

The crown-of-thorns starfish is an overpopulated species in the Great Barrier Reef, which destroys reefs by eating corals. Controlling outbreaks of the starfish requires locating them on the reefs, which is a tedious process when performed by divers and boats. As in the corresponding [Kaggle competition](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/overview), the goal of the project is to automatically detect crown-of-thorns starfish in underwater images from the Great Barrier Reef. In particular, we will train an object detection model that places bounding boxes around the starfish in the images and assigns them a confidence level.

**Data**

We are going to use a ~15GB [dataset from Kaggle](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/data) containing images with zero or more starfish. The images are snapshots from a variety of videos taken in different parts of the Great Barrier Reef. Each image has associated metadata including the ID of the video it comes from and its frame number in the video. The images are in the form of a pixel coordinate corresponding to the upper left corner of a bounding box along with the box’s width and height.

**Framework**

We will integrate the [Albumentations](https://albumentations.ai) framework, an image augmentation library, into our project. We chose this library because it supports a variety of transform operations as blur, fog, and hue shifts, which we hope will be useful augmentations for our underwater imagery. We plan to apply augmentations in real-time during the training process to improve the model’s learning and robustness.

**Model**

We plan to use one of the YOLO (You Only Look Once) models, which have been state-of-the-art in diverse object detection tasks. We expect to use [YOLO11](https://docs.ultralytics.com/models/yolo11/#citations-and-acknowledgements), the newest model in the series. The model is made up of a backbone, a neck, and a head. The backbone uses convolutional neural networks to perform feature extraction on the images. The neck then enhances the feature representations at different scales. Finally, the head generates predictions while considering the multiple scales in the feature maps.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

## Documentation
### Environment
Create a dedicated environment to keep track of the packages for the project
```bash
conda create --name starfish-env python=3.11
conda activate starfish-env
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install -e .
```

### Data
Download the data for the project from our Google Cloud Bucket, which requires the Google Cloud SDK.
```bash
invoke download-data
```

### Train
Train with default arguments
```bash
train
```
Train with data downloaded from the bucket rather than accessing it directly from the cloud
```bash
train data.data_from_bucket=false
```


### Profiling
Train with profiling
```bash
train profiling=True
```
Profile forward pass
```bash
invoke profile-forward-pass
```

### Docker
Build the training dockerfile into a Docker image
```bash
invoke build-train-image
```
Run a container spawned from the docker image
```bash
invoke run-train-image
```

### Vertex AI
Get the Docker image built from `train.dockerfile` in the Artifact Registry on Google Cloud, for example by creating a trigger and using the `cloudbuild.yaml` file. Then you can train a model using that Docker image through the Vertex AI service. This also automatically logs the training to Wandb if your API key has been stored as a secret in Google Cloud.
```bash
invoke train-vertex
```

### Wandb
Login
```bash
wandb login
```
Hyperparameter sweep
```bash
invoke sweep
wandb agent ENTITY/PROJECT_NAME/AGENT_ID
```

### Tests
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
Load test
```bash
locust -f tests/performancetests/locustfile.py --headless --users 10 --spawn-rate 1 --run-time 1m --host https://backend-638730968773.us-central1.run.app
```

### Documentation Site
Build site locally
```bash
mkdocs build
```
Deploy site to `gh-pages` branch
```bash
mkdocs gh-deploy
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
