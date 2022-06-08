#!/bin/sh

BASEDIR=$(pwd)

cd ..
python datagen.py raw_data temp
python train.py temp result GPP
python plot.py result temp GPP
python train.py temp result LEAF
python plot.py result temp LEAF
rm -rf temp

cd $BASEDIR