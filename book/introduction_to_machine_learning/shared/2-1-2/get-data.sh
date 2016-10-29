#!/bin/sh

dir="data"
mkdir -p $dir
wget -O $dir/bank.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
unzip -d $dir $dir/bank.zip
