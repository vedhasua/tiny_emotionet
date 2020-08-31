# TINY EMOTIONET

**What it is about:**

The shallowest-ever (and ever possible) CNN topology, an interpretable/white-box minimalist AI for time- and value-continuous affect and pain prediction. 

**Obtaining the datasets:**

You can submit a request for:

- SEWA / The AVEC 2019 Cross-cultural Emotion Sub-Challenge (CES) Database [here](https://sites.google.com/view/avec2019/home#h.p_-h7OzDVf_pfv) 
- UNBC Mc-Master Shoulder Pain Archive [here](https://www.pitt.edu/~emotion/um-spread.htm)
- BIOVID Heat Pain Database [here](http://www.iikt.ovgu.de/BioVid.print)

**Training and testing the models:**
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

**Feature attribution computation, and the time-series visualisation:**

1. For the model mapping the FAU features to the Affect labels (SEWA):
    
    `filterplot_std.py`
    
2. Training the model to map the FAU features to the PSPI labels (UNBC):

    `filterplot_std.py`

3. Training the model to map the FAU features to the Affect labels (BioVid):

    `filterplot_std.py` (WIP: feature attribution score computation for the models with a batch normalisation layer).

**Dependencies:** 

pandas, numpy, matplotlib, keras 

**Citation**:

If you use this code, please star the repository and cite the following paper:

Pandit, Vedhas, Maximilian Schmitt, Nicholas Cummins, and Björn Schuller. "[I see it in your eyes: Training the shallowest-possible CNN to recognise emotions and pain from muted web-assisted in-the-wild video-chats in real-time](https://authors.elsevier.com/a/1bPwq15hYdjpxA)" Information Processing & Management 57, no. 6 (2020): 102347.
```
@article{pandit2020see,
  title={I see it in your eyes: Training the shallowest-possible CNN to recognise emotions and pain from muted web-assisted in-the-wild video-chats in real-time},
  author={Pandit, Vedhas and Schmitt, Maximilian and Cummins, Nicholas and Schuller, Bj{\"o}rn},
  journal={Information Processing \& Management},
  volume={57},
  number={6},
  pages={102347},
  year={2020},
  publisher={Elsevier},
  doi = {10.1016/j.ipm.2020.102347}
}
```
**Additional Notes**: 

The code posted here should work fine. If you need some advice, tips, documentation, or just interested in building on top of this work in any way, please feel free to reach out. 

We are happy to help and/or collaborate. Stalk us to find our postal address to send us a snail-mail, or wait... Better yet, let's save you all that trouble! Here're my email addresses for this project. Happy coding! :)

**Contact:** 

Vedhas Pandit (panditvedhas@gmail.com, vedhas.pandit@informatik.uni-augsburg.de)

Chair: https://www.informatik.uni-augsburg.de/lehrstuehle/eihw/
