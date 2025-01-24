import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "starfish"
PYTHON_VERSION = "3.11"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands


@task
def conda(ctx):
    ctx.run("conda create --name starfish-env python=3.11", echo=True)
    ctx.run("conda activate starfish-env", echo=True)
    ctx.run("pip install -r requirements.txt", echo=True)
    ctx.run("pip install -r requirements_dev.txt", echo=True)
    ctx.run("pip install -e .", echo=True)


@task
def download_data(ctx) -> None:
    ctx.run("gsutil -m cp -r gs://starfish-detection-data .", pty=not WINDOWS)


@task
def data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def profile_forward_pass(ctx):
    ctx.run("python src/starfish/profile_forward_pass.py", echo=True)


@task
def build_train_image(ctx):
    ctx.run("docker build -f dockerfiles/train.dockerfile . -t train:latest", echo=True, pty=not WINDOWS)


@task
def backend_image_to_cloud(ctx):
    ctx.run("docker build -t backend:latest -f dockerfiles/inference_backend.dockerfile .", echo=True, pty=not WINDOWS)
    ctx.run(
        "docker tag backend:latest us-central1-docker.pkg.dev/starfish-detection/frontend-backend/backend:latest",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        "docker push us-central1-docker.pkg.dev/starfish-detection/frontend-backend/backend:latest",
        echo=True,
        pty=not WINDOWS,
    )


@task
def frontend_image_to_cloud(ctx):
    ctx.run(
        "docker build -t frontend:latest -f dockerfiles/inference_frontend.dockerfile .", echo=True, pty=not WINDOWS
    )
    ctx.run(
        "docker tag frontend:latest us-central1-docker.pkg.dev/starfish-detection/frontend-backend/frontend:latest",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        "docker push us-central1-docker.pkg.dev/starfish-detection/frontend-backend/frontend:latest",
        echo=True,
        pty=not WINDOWS,
    )


@task
def deploy_backend(ctx):
    ctx.run(
        "gcloud run deploy backend --image=us-central1-docker.pkg.dev/starfish-detection/frontend-backend/backend:latest --region=us-central1 --platform=managed --allow-unauthenticated --port=8080",
        echo=True,
        pty=not WINDOWS,
    )


@task
def deploy_frontend(ctx):
    ctx.run(
        "gcloud run deploy frontend --image=us-central1-docker.pkg.dev/starfish-detection/frontend-backend/frontend:latest --region=us-central1 --platform=managed --allow-unauthenticated --port=8080",
        echo=True,
        pty=not WINDOWS,
    )

@task
def run_train_image(ctx):
    ctx.run("docker run --rm --name RUN_NAME IMAGE_NAME:latest", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def train_vertex(ctx):
    ctx.run("gcloud builds submit --config=vertex_ai_train.yaml", echo=True)


@task
def sweep(ctx):
    ctx.run("wandb sweep configs/sweep_config.yaml", echo=True)


@task
def test_data(ctx):
    ctx.run("pytest tests/unittests/test_data.py", echo=True)


@task
def test_model(ctx):
    ctx.run("pytest tests/unittests/test_model.py", echo=True)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run --source=src -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
