# Welcome to Starfish Detection

## Project layout

```txt
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
