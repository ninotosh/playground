## what

This is a directory of Dockerized Jupyter notebooks for [_Grokking Deep Learning_](https://www.manning.com/books/grokking-deep-learning).

## requirements

Install [Docker](https://www.docker.com/what-docker).

## how to run

1. Build a Docker image.
  * `docker build -t grokking_dl .`
2. Start a container.
  * `docker run -it --rm -p 8888:8888 -v `pwd`:/mnt grokking_dl start-notebook.sh --notebook-dir=/mnt`
3. Open `http://localhost:8888/`
