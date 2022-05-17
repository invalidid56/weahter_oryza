#!/bin/sh

BASEDIR=$(pwd)

cd ..
python datagen.py raw_data temp
python train.py temp result

cd BASEDIR