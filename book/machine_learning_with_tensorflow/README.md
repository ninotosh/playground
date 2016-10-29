## what

This is a directory of Dockerized Jupyter notebooks for [_Machine Learning with TensorFlow_](https://www.manning.com/books/machine-learning-with-tensorflow).

## requirements

* [Docker](https://www.docker.com/products/docker).

## how to run

1. Build a Docker image.
  * `docker build -t ml_with_tf .`
2. Start a container.
  * ``docker run -it --rm -p 8888:8888 -v `pwd`:/mnt ml_with_tf sh -c 'jupyter notebook --notebook-dir=/mnt --ip=0.0.0.0 --no-browser'``
3. Open `http://localhost:8888/`
