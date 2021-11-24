![logo](banner.png)

## Overview
Sukima is a ready-to-deploy container that implements a REST API for Language Models designed with the specific purpose of easy deployment and scalability.

### Curent API Functions
- **models** : Fetch a list of ready-to-use Language Models for inference.
- **load** : Allocate a Language Model.
- **delete** : Free a Language Model from memory.
- **generate** : Use a Language Model to generate tokens.

### Setup
1. Customize the ports and host in the [Dockerfile](Dockerfile) to your liking.
2. Install Docker and run ``docker-compose up`` in the directory with the Dockerfile to deploy.
3. That's it!

### Todo
- HTTPS Support
- Rate Limiting
- Support for other Language Modeling tasks such as Sentiment Analysis and Named Entity Recognition.
- Soft Prompt tuning endpoint

### License
[Simplified BSD License](LICENSE)
