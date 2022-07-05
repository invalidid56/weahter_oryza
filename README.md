# Weather Oryza
Reproducible Research Package 
***

### Description
***
Weather Oryza is a Reproducible Research Package to Estimate Gross Primary Productivity(GPP), Respiration(RECO), Leaf Temperature(LT) of **Oryza Sativa** from Meteorological Datum.
We Used Temperature, Shortwave Input, Humidity to Train Model. When you run train.sh, Results will be produced in weather_oryza/result folder

### Environment
***
* None of GPUs are Needed, it can be reproduced at any environment.
### Files
***
* weahter_oryza/datagen,py
* weahter_oryza/train.py
* weahter_oryza/plot.py
* bin/train.sh
* raw_data

### Usage
***
1. Install Package using pip or setup.py
2. Run **train.sh**
3. Check your result in **weather_oryza/result**