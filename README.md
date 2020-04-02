# TINY EMOTIONET

## Obtaining the datasets:

You can submit a request for

SEWA / The AVEC 2019 Cross-cultural Emotion Sub-Challenge (CES) Database [here](https://sites.google.com/view/avec2019/home#h.p_-h7OzDVf_pfv) 

UNBC Mc-Master Shoulder Pain Archive [here](https://www.pitt.edu/~emotion/um-spread.htm)

BIOVID Heat Pain Database [here](http://www.iikt.ovgu.de/BioVid.print)

## Primary scripts for running experiments:

### Training and testing the models:
1. Training the model to map the FAU features to the Affect labels (SEWA):
    
    `deepemotion_keras_DHC.py`
    
    Training split: 34 German subjects
    
    Validation split: 14 German subjects
    
    Testing splits: 16 German, 66 Hungarian, 70 Chinese subjects
    
2. Training the model to map the FAU features to the PSPI labels (UNBC):

    `run_regression.py`

    Training split: 9 subjects
    
    Validation split: 7 subjects
    
    Testing splits: 9 subjects



3. Training the model to map the FAU features to the Affect labels (BioVid):

    `run_regression.py`

    Training split: 29 subjects
    
    Validation split: 29 subjects
    
    Testing splits: 29 subjects



### Feature attribution computation, and the time-series visualisation:

1. For the model mapping the FAU features to the Affect labels (SEWA):
    
    `filterplot_std.py`
    
2. Training the model to map the FAU features to the PSPI labels (UNBC):

    `filterplot_std.py`

3. Training the model to map the FAU features to the Affect labels (BioVid):

    `filterplot_std.py` (WIP: feature attribution score computation for the models with a batch normalisation layer).
