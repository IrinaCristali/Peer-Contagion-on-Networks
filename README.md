# README

This repository contains the software and data for the paper "Using Embeddings for Causal Estimation of Peer Influence in Social Networks". Homophily - the tendency of similar people to cluster together - is generally confounded with peer influence. For example, did Alice influence her friend Bob to get vaccinated, or did they both get vaccinated because they are Democrats (who tend to get vaccinated at higher rates), and that's why, actually, Alice and Bob are friends in the first place? The paper's main contributions are to mathematically formalize a causal peer influence effect which accounts for homophily, and to propose a nonparametric method of identifying and estimating this effect from social network data. 


This code builds on a method of using network embeddings to estimate averate treatment effects while adjusting for sources of unobserved confounding (https://github.com/vveitch/causal-network-embeddings) and on relational empirical risk minimization (https://github.com/wooden-spoon/relational-ERM/tree/tensorflow-v2). The contribution is extending these techniques to accomodate peer contagion effects, and providing a simple Keras implementation which could be easily applied to other data sources. 

# Requirements and Setup

First, within the 'src' directory, run the following command:

```
pip install -r requirements.txt
```

Notably, this code requires Python >= 3.8 and Tensorflow version **2.3.0**. The code is currently not compatible with other Tensorflow versions. 

Then, after making sure **gcc** is installed, run the following command, which builds efficient graph samplers to be used downstream, in minimizing an empirical risk over the network. 

```
python setup.py build_ext --inplace
```


# Data

We use a pre-processed subset of data from the Slovakian social media website Pokec. The original data can be downloaded from https://snap.stanford.edu/data/soc-Pokec.html. Our processed data files can be found in the folder path *src/dat/pokec/regional_subset*. The Python module containing the initial data cleaning code is located at *src/relational_erm/data_cleaning/pokec.py*. 

# Reproducing the Experiments


The default settings for the code match those used in the paper, and in previous work by Veitch et al. (https://arxiv.org/abs/1902.04114). 

1. To reproduce the experimental results in Section 6.1 of the paper - estimating peer influence for continuous outcomes - from 'src' run the command:

```
python -m relational_erm.rerm_model.keras_model --beta_1 BETA_1 --covariate COVARIATE --seed SEED
```

where the parameter BETA_1 - representing the strength of unobserved confounding - can take any of the values 0, 1, 10, the parameter COVARIATE - representing the variable taken as the hidden source of confounding - can take any of the values 'region', 'registration', 'age', and the SEED can take any of the values from 1 to 100. 

The above script will produce a csv file with outcome predictions for each node under hypothetical interventional treatments T = all 0's and T = all 1's. 

Having obtained all csv files for all the possible parameter settings for BETA_1, COVARIATE, and SEED (all of these are already included in the directory named 'cluster_simulations'), the average peer effect point estimates and confidence bands can be obtained by running the following inside 'src':

```
python -m relational_erm.data_cleaning.Confidence_Intervals 
```


2. To reproduce the experimental results in Section 6.2 of the paper - estimating peer influence for binary outcomes - from 'src' run the command:


```
python -m relational_erm.rerm_model.keras_model2 --covariate COVARIATE --seed SEED
```

where, similarly to the continuous case, COVARIATE can be anything from 'region', 'registration', 'age', and the s=SEED can take any value from 1 to 100. This script also produces a csv file with outcome predictions for each node under hypothetical interventional treatments T = all 0's and T = all 1's, and these results are also included in the 'cluster_simulations' folder. 

As above, the average peer contagion effects together with their error bands can be obtained by running 

```
python -m relational_erm.data_cleaning.Confidence_Intervals 
```

3. Finally, the way the treatment and outcome where simulated for the continuous and binary scenarios in this paper is shown in the Python module located at "src/relational_erm/data_cleaning/simulate_treatment_outcome.py". This file also computes the unadjusted, naive, peer contagion effects. The code corresponding to the parametric baseline method for peer contagion can be found at "src/relational_erm/data_cleaning/simulate_baseline_sbm.py". 

