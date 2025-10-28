# This is our first build stage, it will not persist in the final image
FROM ubuntu as intermediate
RUN apt-get -y update && apt-get install -y git
ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa
RUN chmod 400 /root/.ssh/id_rsa
# Make sure your domain is accepted
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
# Download the computer vision framework
RUN git clone git@github.com:pcr-upm/students_landmarks.git students_landmarks
ADD data /students_landmarks/data

# Copy the repository from the previous image
FROM nvcr.io/nvidia/cuda:12.2.0-base-ubuntu22.04
ENV LANG=C.UTF-8
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get -y update && apt-get install -y build-essential wget cmake libgl1-mesa-glx libsm6 libxext6 libglib2.0-dev
RUN mkdir /home/username
WORKDIR /home/username
COPY --from=intermediate /students_landmarks /home/username/students_landmarks
LABEL maintainer="roberto.valle@upm.es"
# Setup conda environment
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/username/miniconda.sh
RUN chmod +x /home/username/miniconda.sh
RUN /home/username/miniconda.sh -b -p /home/username/conda
RUN /home/username/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /home/username/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN /home/username/conda/bin/conda create --name students python=3.10
# Activate conda environment
ENV PATH /home/username/conda/envs/students/bin:/home/username/conda/bin:$PATH
# Make RUN commands use the new environment (source activate students)
SHELL ["conda", "run", "-n", "students", "/bin/bash", "-c"]
# Install dependencies
RUN pip install images-framework tqdm scikit-learn torch pytorch-lightning torchvision torchinfo tensorboard segmentation-models-pytorch
