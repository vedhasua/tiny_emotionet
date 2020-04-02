# TINY EMOTIONET

## Obtaining the datasets:

You can submit a request for

SEWA / The AVEC 2019 Cross-cultural Emotion Sub-Challenge (CES) Database [here](https://sites.google.com/view/avec2019/home#h.p_-h7OzDVf_pfv) 

UNBC Mc-Master Shoulder Pain Archive [here](https://www.pitt.edu/~emotion/um-spread.htm)

BIOVID Heat Pain Database [here](http://www.iikt.ovgu.de/BioVid.print)

## Primary scripts for running experiments:

### Training and testing the models:
1. Training the model to map the FAU features to the Affect labels (SEWA):
    
    `deepemotion_keras_DHC.py`:  Training done on German training split, testing on the entire Chinese and Hungarian splits + the German testing split
    
2. Training the model to map the FAU features to the PSPI labels (UNBC):

    `run_regression.py`

3. Training the model to map the FAU features to the Affect labels (BioVid):

    `run_regression.py`

### Feature attribution computation, and the time-series visualisation:

1. For the model mapping the FAU features to the Affect labels (SEWA):
    
    `filterplot_std.py`
    
2. Training the model to map the FAU features to the PSPI labels (UNBC):

    `filterplot_std.py`

3. Training the model to map the FAU features to the Affect labels (BioVid):

    `filterplot_std.py` (WIP: feature attribution score computation for the models with a batch normalisation layer).
