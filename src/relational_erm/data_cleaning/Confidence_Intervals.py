import os
import numpy as np
import pandas as pd


def mean_confidence_interval(a):
    n = len(a)
    m, se = np.mean(a), np.std(a) / np.sqrt(n)
    return m, se


folder = 'src/cluster_simulations'

filelist = [file for file in os.listdir(folder) if file.startswith('outcome_beta1_10_cov_registration_seed')]

ates = np.array([])
database = {}

for file in filelist:
    database[file] = pd.read_csv('CLUSTER_RESULTS/' + file, '\t')
    adjusted_ate = database[file]['expected_outcome_st_all_treatment'].mean() - database[file][
        'expected_outcome_st_no_treatment'].mean()
    ates = np.append(ates, adjusted_ate)

print(mean_confidence_interval(ates))
