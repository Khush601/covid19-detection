#!/bin/bash
# Install required system dependencies
apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    libopenblas-dev \
    liblapack-dev \
    libpq-dev
