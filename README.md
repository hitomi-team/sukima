![logo](banner.png)

## Overview
Sukima is a ready-to-deploy container that implements a REST API for Language Models designed with the specific purpose of easy deployment and scalability.

### Curent API Functions
- **models** : Fetch a list of ready-to-use Language Models for inference.
- **load** : Allocate a Language Model.
- **generate** : Use a Language Model to generate tokens.
- **classify** : Use a Language Model to classify tokens and retrieve scores.

To view more information for API Usage, see ``/docs`` endpoint.

### Setup
This following guide assumes that you cloned the repository on your machine and that you have Docker and docker-compose.
If you wish to leverage your GPU for inference, make sure you have the appropiate CUDA drivers installed.

#### Linux 

  1. In the root directory of the repository run the ``docker-compose up`` command; on the first run, `docker-compose` up should configure the FastAPI app and PostgreSQL. Uvicorn options are set to `reload=True`, so any code changes you make on your host machine will be automatically applied to the container.

  **NOTE**: If you have an appropriate NVIDIA GPU that you want to use, run `docker-compose -f docker-compose_nvidia-gpu.yaml up` instead. Remember to install [nvidia-container-runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu) on your host machine.
  
  2. Run ``docker-compose run web alembic upgrade head`` to create ``db tables`` if you're setting up, or to apply migrations if you made any db schema changes.

#### Kubernetes

1. kubectl apply -f k8s
2. TODO

#### Windows

This is assuming that you're using Docker for Desktop Windows.

1. Follow the steps from the Linux guide.
2. Curl up into a ball because you're not using a Unix system.
3. If you want to use your GPU, good luck. Refer to [Docker](https://docs.docker.com/desktop/windows/wsl/#gpu-support) and [NVIDIA's guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) to set up CUDA for WSL. If I were you, I'd just skip the pain and use [distilgpt2](https://huggingface.co/distilgpt2) or [gpt-j-random-tinier](https://huggingface.co/hakurei/gpt-j-random-tinier) models for testing.

### Using the app

0. Navigate to the apps `/docs` endpoint (usually found at http://0.0.0.0:8000/docs). Alternatively, if for some reason that is not possible, you can send curl requests to the app.
1. Register an user. Make sure to set the ``permission_level`` to a value greater than 0 and give it a username/password combination.
2. Generate a token. Input the username/password but leave all other fields blank.
3. Log into the app by using the `Authorize` button, using your username/password.
4. Load you first model. For the purposes of debugging you may want to use a smaller model like `hakurei/gpt-j-random-tinier` (for testing only) or `iokru/c1-1.3B` (one of the smallest 'coherent' models).
5. After the model's loading has been completed, use the `Get model list` function to check if your model is in the database and flagged as `ready`.
6. Use the `generate` function to test the model. Make sure to set the correct values for the `model` you have loaded, as well as add some text that you want the bot to reply to in the `prompt` value.

### Todo
- Autoscaling
- HTTPS Support
- Rate Limiting
- Support for other Language Modeling tasks such as Sentiment Analysis and Named Entity Recognition.

### License
[GPL-2.0](LICENSE)
