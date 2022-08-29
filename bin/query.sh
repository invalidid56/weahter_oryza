#!/bin/sh

BASEDIR=$(pwd)

cd ..

rm -rf result
echo Processing Test Data
python weather_oryza/datagen.py raw_data temp test_proc

echo Plotting Leaf Temp. Model
python weather_oryza/plot.py result temp LEAF

echo Plotting GPP Model
python weather_oryza/plot.py result temp GPP

echo Plotting RECO Model
python weather_oryza/plot.py result temp RECO

echo Training Process Finished Check Result Folder

cd $BASEDIR