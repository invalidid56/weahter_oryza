#!/bin/sh

BASEDIR=$(pwd)

cd ..

rm -rf result
echo Processing Test Data
python weather_oryza/datagen.py raw_data_test temp test_proc

echo Plotting Leaf Temp. Model
python weather_oryza/plot.py test_result temp LEAF

echo Plotting GPP Model
python weather_oryza/plot.py test_result temp GPP

echo Training Process Finished Check Result Folder

cd $BASEDIR