FROM tensorflow/tensorflow:latest-py3

ENV PYTHONIOENCODING UTF-8

RUN apt-get update && apt-get install -y git python3-tk
RUN git clone https://github.com/google/seq2seq.git && cd seq2seq && pip3 install -e .
RUN mkdir -p "$HOME/.config/matplotlib" && echo "backend : Agg" >> "$HOME/.config/matplotlib/matplotlibrc"
