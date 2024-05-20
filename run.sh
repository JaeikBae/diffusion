docker build -t diffusion .
docker run -it \
    -v .:/ws \
    --gpus all \
    diffusion \
    /bin/bash