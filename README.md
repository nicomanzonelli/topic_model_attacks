# Membership Inference Attacks and Privacy in Topic Modeling

This repository contains the code and resources for "Membership Inference Attacks and Privacy in Topic Modeling." We investigate the effectiveness of membership inference attacks on topic models, and proposes differentially private (DP) vocabulary selection as a pre-processing step for topic modeling. The code in this repository allows reproducible results.

### Running Attack Simulations

After ensuring that all of the required packages in requirements.txt are installed, you can run a basic online LiRA against LDA using our simple command line interface.

```
python cli.py -d "./data/pheme_clean.py" -k 5
```

This experiment will simulate a LiRA against a model trained on the data by training $N = 128$ shadow models on samples from the data and fitting a LiRA. The code leverages python's multiprocessing library which speeds-up execution. On a M2 MacBook Air with 8GB RAM this simulation ~5 minutes to run.

### Code Organization

This repository is organized as follows:

- data - Directory containing data used in each experiment. Sources and descriptions of each data set is available in the paper.
- defense - Directory containing code for DP topic modeling.
  - init.py
  - dpsu_gw.py - Code from [Carvalho et al (2022)](https://github.com/ricardocarvalhods/diff-private-set-union) for DP Set Union (DPSU).
  - lda.py - Code for implementing DP LDA.
  - fdptm_helpers.py - Provides helper functions for Fully Differentially Private Topic Modeling.
- cli.py - Provides simple command line interface that allows for reproducible results.
- simulation_utils.py - Provides functions to help run attack simulations.
- topic_mia.py - Provides class for LiRA from [Carlini et al (2021)](https://arxiv.org/abs/2112.03570) adapted to attack topic models.
- topic_models.py - Provides general topic modeling functions.

**Note:** This repository is intended for educational and research purposes only. The code and resources should be used responsibly and ethically, with respect for privacy considerations.
