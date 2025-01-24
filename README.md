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
.
├── .dvc/                      # DVC configuration
│   ├── config                 # Configuration file for DVC
│   └── .gitignore             # Ignore DVC files in Git
├── .github/                   # GitHub workflows and automation
│   ├── workflows/             # CI/CD workflows
│   │   ├── cml_data.yaml      # CML (Continuous Machine Learning) workflow
│   │   ├── pre_commit.yaml    # Pre-commit checks
│   │   ├── stage_model.yaml   # Workflow to stage and deploy models
│   │   └── tests.yaml         # Workflow for running tests
│   └── dependabot.yaml        # Dependabot configuration for dependency updates
├── configs/                   # Configuration files
│   ├── callbacks/             # Callback configurations
│   │   ├── default_callbacks.yaml
│   │   └── wandb_image_logger.yaml
│   ├── experiment/            # Experiment configurations
│   │   ├── profile.yaml
│   │   └── train_local.yaml
│   ├── logger/                # Logging configurations
│   │   └── wandb_logger.yaml
│   ├── model/                 # Model configurations
│   │   └── default_model.yaml
│   ├── trainer/               # Trainer configurations
│   │   └── default_trainer.yaml
│   ├── starfish_data.yaml     # Dataset configurations
│   ├── sweep_config.yaml      # Hyperparameter sweep configurations
│   ├── main_config.yaml       # Main configuration file
│   └── vertex_ai_config.yaml  # Vertex AI configuration
├── dockerfiles/               # Dockerfiles for deployment and training
│   ├── inference_backend.dockerfile
│   ├── inference_frontend.dockerfile
│   ├── train.dockerfile
│   └── data_drift.dockerfile  # Dockerfile for data drift detection
├── docs/                      # Project documentation
│   ├── figures/               # Documentation-related figures
│   │   ├── app-image1.png
│   │   ├── app-image2.png
│   │   ├── app-image3.png
│   │   ├── app-image4.png
│   │   ├── app-image5.png
│   │   ├── app-image6.png
│   ├── README.md              # Docs README file
│   ├── application.md         # Documentation about the application
│   ├── index.md               # Main page of the documentation
│   └── code.md                # Code-related documentation
├── reports/                   # Reports and visualizations
│   ├── figures/               # Result figures
│   │   ├── image_logging.png
│   │   ├── loss_logging.png
│   │   └── sweep.png
│   ├── README.md              # Explanation of reports
│   └── report.py              # Report generation script
├── src/                       # Source code
│   ├── starfish/              # Main project package
│   │   ├── __init__.py        # Module initialization
│   │   ├── apis/              # API-related files
│   │   │   ├── inference_backend.py
│   │   │   ├── inference_frontend.py
│   │   │   └── data_drift.py  # Data drift detection utilities
│   │   ├── callbacks.py       # Callbacks for training
│   │   ├── data.py            # Data processing utilities
│   │   ├── evaluate.py        # Model evaluation
│   │   ├── model.py           # Model definitions
│   │   ├── onnx_model.py      # ONNX model conversion and inference
│   │   ├── profile_forward_pass.py # Profiling scripts
│   │   ├── train.py           # Training script
│   │   └── visualize.py       # Visualization utilities
├── tests/                     # Test suite
│   ├── integrationtests/      # Integration tests
│   │   └── test_api.py        # API endpoint tests
│   ├── performancetests/      # Performance tests
│   │   └── locustfile.py      # Load testing with Locust
│   └── unittests/             # Unit tests
│       ├── __init__.py
│       ├── test_data.py       # Data-related tests
│       └── test_model.py      # Model-related tests
├── .dvcignore                 # Ignore patterns for DVC
├── .gcloudignore              # Ignore patterns for Google Cloud
├── .gitignore                 # Ignore patterns for Git
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
├── LICENSE                    # License file
├── README.md                  # Project README
├── cloudbuild.yaml            # Cloud Build configuration
├── data.dvc                   # DVC tracking file
├── mkdocs.yml                 # MkDocs configuration
├── pyproject.toml             # Python project metadata
├── requirements.txt           # Python dependencies
├── requirements_dev.txt       # Development dependencies
├── tasks.py                   # Task runner (e.g., Invoke or similar tools)
├── vertex_ai_train.yaml       # Vertex AI training configuration

```

## Documentation

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
