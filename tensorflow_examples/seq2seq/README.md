### [seq2seq](https://google.github.io/seq2seq/) example

#### 1. build a Docker image

```sh
$ docker build -t seq2seq .
```

#### 2. prepare training files

Modify `data/input/train/sources.txt` and `data/input/train/targets.txt` if you would like to.

#### 3. generate vocabulary files

source:

```sh
$ docker run -i --rm seq2seq python seq2seq/bin/tools/generate_vocab.py < data/input/train/sources.txt > data/input/train/sources_vocab.txt
```

target:

```sh
$ docker run -i --rm seq2seq python seq2seq/bin/tools/generate_vocab.py < data/input/train/targets.txt > data/input/train/targets_vocab.txt
```

#### 4. run training

Modify `config/model.yml` and `config/train.yml` if you would like to.

Run `rm -r data/output/*` to start over a new training.

```sh
$ docker run -it --rm -v `pwd`:/mnt -w="/mnt" -e "PYTHONPATH=/mnt/src" seq2seq python bin/main.py --train
```

#### 5. prepare a test file

Modify `data/input/test/sources.txt`.

#### 6. run inference

Modify `config/infer.yml` if you would like to, and run:

```sh
$ docker run -it --rm -v `pwd`:/mnt -w="/mnt" -e "PYTHONPATH=/mnt/src" seq2seq python bin/main.py
```
