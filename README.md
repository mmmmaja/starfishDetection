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
│   ├── config
│   └── .gitignore
├── .github/                   # GitHub workflows and automation
│   ├── workflows/
│   │   ├── cml_data.yaml
│   │   ├── pre_commit.yaml
│   │   ├── stage_model.yaml
│   │   └── tests.yaml
│   └── dependabot.yaml
├── configs/                   # Configuration files
│   ├── callbacks/
│   │   ├── default_callbacks.yaml
│   │   └── wandb_image_logger.yaml
│   ├── experiment/
│   │   ├── profile.yaml
│   │   └── train_local.yaml
│   ├── logger/
│   │   └── wandb_logger.yaml
│   ├── model/
│   │   └── default_model.yaml
│   ├── trainer/
│   │   └── default_trainer.yaml
│   ├── starfish_data.yaml
│   ├── sweep_config.yaml
│   ├── main_config.yaml
│   └── vertex_ai_config.yaml
├── dockerfiles/               # Dockerfiles for deployment and training
│   ├── inference_backend.dockerfile
│   ├── inference_frontend.dockerfile
│   └── train.dockerfile
├── docs/                      # Project documentation
│   ├── README.md
│   ├── application.md
│   ├── index.md
│   └── training.md
├── reports/
│   ├── figures/
│   │   ├── image_logging.png
│   │   ├── loss_logging.png
│   │   └── sweep.png
│   ├── README.md
│   └── report.py
├── src/                       # Source code
│   ├── starfish/
│   │   ├── __init__.py
│   │   ├── callbacks.py
│   │   ├── data.py
│   │   ├── data_drift.html
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   ├── onnx_model.py
│   │   ├── profile_forward_pass.py
│   │   ├── train.py
│   │   └── visualize.py
├── tests/                     # Test suite
│   ├── integrationtests/
│   │   └── test_api.py
│   ├── performancetests/
│   │   └── locustfile.py
│   └── unittests/
│       ├── __init__.py
│       ├── test_data.py
│       └── test_model.py
├── .dvcignore
├── .gcloudignore
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md                  # Project README
├── cloudbuild.yaml            # Cloud Build configuration
├── data.dvc                   # DVC tracking file
├── mkdocs.yml                 # MkDocs configuration
├── pyproject.toml             # Python project file
├── requirements.txt           # Dependencies
├── requirements_dev.txt       # Development dependencies
├── tasks.py
├── vertex_ai_train.yaml       # Vertex AI training configuration

```

## Documentation

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
