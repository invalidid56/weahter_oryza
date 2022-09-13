#!/bin/sh

BASEDIR=$(pwd)

cd ..

rm -rf result
echo Generating Data
python weather_oryza/datagen.py raw_data temp train

echo Training Leaf Temp. Model
python weather_oryza/train.py temp result LEAF
python weather_oryza/plot.py result temp LEAF

echo Training GPP Model
python weather_oryza/train.py temp result GPP
python weather_oryza/plot.py result temp GPP

echo Training RECO Model
python weather_oryza/train.py temp result RECO
python weather_oryza/plot.py result temp RECO

echo Training Process Finished Check Result Folder

cd $BASEDIR