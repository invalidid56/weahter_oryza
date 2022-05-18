#!/bin/sh

BASEDIR=$(pwd)

cd ..
python datagen.py raw_data temp
python train.py temp result
python plot.py result

cd $BASEDIR