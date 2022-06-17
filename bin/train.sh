#!/bin/sh

BASEDIR=$(pwd)

cd ..
python datagen.py raw_data temp
python train.py temp result RECO
python plot.py result temp RECO
python train.py temp result LEAF
python plot.py result temp LEAF
python train.py temp result GPP
python plot.py result temp GPP
rm -rf temp

cd $BASEDIR