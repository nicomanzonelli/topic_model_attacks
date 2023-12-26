# Membership Inference Attacks and Privacy in Topic Modeling

This repository contains the code and resources for "Membership Inference Attacks and Privacy in Topic Modeling." We investigate the effectiveness of membership inference attacks on topic models, and proposes differentially private (DP) vocabulary selection as a pre-processing step for topic modeling. The code in this repository allows reproducable results.

### Code Organization

This repository is organized as follows:

- requirements.txt - Text file that specifies required python packages for running our code.
- data - Directory containing data used in each experiment. Sources and descriptions of each data set is available in the paper.
- dp_defense - Directory containing code for DP topic modeling.
  - init.py
  - dpsu_gw.py - Code from [Carvalho et al (2022)](https://github.com/ricardocarvalhods/diff-private-set-union) for DP Set Union (DPSU).
  - lda.py - Code for implementing DP LDA.
  - fdptm.py - Code that combines DP LDA and DPSU to implement Fully Differentially Private Topic Modeling.
- api.py - Wraper for implementing attack simulations.
- attack_simulations.py - Provides functions for attack simulations for the LiRA on topic models.
- basic_attacks.py - Code for replicating [Huang et al's (2022)](https://jcst.ict.ac.cn/EN/10.1007/s11390-022-2425-x) MIAs.
- online_LiRA.py - Provides class for online LiRA from [Carlini et al (2021)](https://arxiv.org/abs/2112.03570) adapted to attack topic models.
- offline_LiRA.py - Provides class for offline LiRA from [Carlini et al (2021)](https://arxiv.org/abs/2112.03570) adapted to attack topic models.
- utils.py - A series of helper functions to calculate statistics or evaluate performance.

### Running Attack Simulations

After ensuring that all of the required packages in requirements.txt are installed, you can run a basic online LiRA against LDA by executing the following code.

```
python api.py
```

Within the api.py file, we specify the input data path, output file path (pheme_test.pickle), and the attack simulation parameters. The simulation parameters for this simple attack train $N=64$ shadow models on the $TweetRumors$ data set. This API provides the basic functionality for running the LiRA.

On a M2 MacBook Air with 8GB RAM this simulation ~25 minutes to run. Although not provided, simple scripts can provide parrellization over different experements or simulation iterations.

To run [Huang et al's (2022)](https://jcst.ict.ac.cn/EN/10.1007/s11390-022-2425-x) simple attack execute the following code.

```
python basic_attacks.py
```

This simulation takes approxamatle 3 minutes to complete. Within the basic attacks file we specify input data path, output file path (pheme_basic_test.pickle).

**Note:** This repository is intended for educational and research purposes only. The code and resources should be used responsibly and ethically, with respect for privacy considerations.
