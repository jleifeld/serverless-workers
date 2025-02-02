ARG BASE_IMAGE=runpod/pytorch:3.10-2.0.0-117
FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-c"]

WORKDIR /src

# Install missing dependencies
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN pip install torchvision==0.15.1
RUN pip install python-slugify==8.0.1
RUN pip install huggingface-hub==0.13.3
RUN apt-get install zstd -y

ADD mainrunpodA1111.py /src/
ADD mainrunpodDreamboothv2.py /src/
ADD install.py /src/
RUN python -u install.py

# Replace paths.py with the one that works with the new paths
RUN cd /workspace/sd/stable-diffusion-webui/modules && wget -q -O paths.py https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/AUTOMATIC1111_files/paths.py
RUN cd /workspace/sd/stable-diffusion-webui/modules && sed -i 's@/content/gdrive/MyDrive/sd/stablediffusion@/workspace/sd/stablediffusion@' /workspace/sd/stable-diffusion-webui/modules/paths.py

RUN python -m pip install --upgrade pip
RUN pip install bitsandbytes==0.35
RUN pip install transformers==4.25.1
RUN pip install xformers==0.0.18
RUN pip install runpod==0.9.1
RUN pip install setuptools==67.4.0
# RUN pip install https://github.com/runpod/runpod-python/archive/main.zip

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs -y
ADD model_fetcher.sh /src/
RUN sh /src/model_fetcher.sh

RUN git clone https://github.com/TheLastBen/fast-stable-diffusion.git
RUN git clone --depth 1 --branch main https://github.com/TheLastBen/diffusers

ENV DEBIAN_FRONTEND noninteractive
RUN apt update && apt install python3.10-venv --yes --no-install-recommends
RUN apt clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

RUN cd /workspace/sd/stable-diffusion-webui/models/
RUN mkdir ESRGAN
ADD 4x-UltraSharp.pth /workspace/sd/stable-diffusion-webui/models/ESRGAN/

RUN pip install boto3>=1.26.15

ADD dreambooth.py /src/
ADD custom_model.py /src/
ADD handler.py /src/
ADD start.sh /src/
ADD test_input.json /src/

RUN chmod +x /src/start.sh
CMD ["/src/start.sh"]
