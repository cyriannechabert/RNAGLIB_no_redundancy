FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone and install RNAGlib
RUN git clone https://github.com/cgoliver/rnaglib.git
RUN pip install ./rnaglib
RUN pip install tensorboard
RUN pip install torch


# Install fr3d-python required by forgi
RUN pip install git+https://github.com/cgoliver/fr3d-python.git

# Default command
# CMD ["python"]

# CMD ["/bin/bash"]
