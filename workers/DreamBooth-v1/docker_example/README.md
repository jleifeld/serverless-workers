https://github.com/TheLastBen/fast-stable-diffusion/blob/main/fast-DreamBooth.ipynb

https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast-DreamBooth.ipynb#scrollTo=LC4ukG60fgMy

https://github.com/TheLastBen/diffusers

https://huggingface.co/datasets/TheLastBen/RNPD

DOCKER_BUILDKIT=1 docker build -t jleifeld/serverless-dreambooth-v1:latest .
docker push jleifeld/serverless-dreambooth-v1:latest

docker run --rm -it -p 8000:8000 --entrypoint bash -v $(pwd)/host-system:/host-system jleifeld/serverless-dreambooth-v1:latest