![logo](banner.png)

## Overview
Sukima is a ready-to-deploy container that implements a REST API for Language Models.

### Curent API Functions
- **models** : Fetch a list of ready-to-use Language Models for inference.
- **load** : Allocate a Language Model.
- **delete** : Free a Language Model from memory.
- **generate** : Use a Language Model to generate tokens.

### Setup
1. Customize the ports and host in the [Dockerfile](Dockerfile) to your liking.
2. Install Docker and run ``docker-compose up`` in the directory with the Dockerfile to deploy.
3. That's it!

### License
[Simplified BSD License](LICENSE)
