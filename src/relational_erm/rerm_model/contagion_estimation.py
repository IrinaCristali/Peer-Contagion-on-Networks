import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import statsmodels.api as sm


def ate_from_rerm_csv(csv_path):
    output = pd.read_csv(csv_path, '\t')
    y = output['y']
    X = output['v']
    X = sm.add_constant(X)
    model1 = sm.OLS(y, X).fit()
    unadjusted_ate = model1.params['v']  # reg.coef_
    unadjusted_ate_std = model1.bse['v']
    return unadjusted_ate, unadjusted_ate_std  # reg.bse #unadjusted_ate_std


def ate_binary_from_rerm_csv(csv_path):
    output = pd.read_csv(csv_path, '\t')
    y = output['y']
    v = output['v']
    y = list(y)
    v = list(v)
    log_reg = sm.Logit(y, v).fit()
    unadjusted_ate = log_reg.params[0]
    unadjusted_ate_std = log_reg.bse[0]

    return unadjusted_ate, unadjusted_ate_std


def main():
    ## Results for Experiment Section 6.1 - Continuous outcome
    print('Confounder = region, beta0 = 1, beta1 = 0')
    results1 = ate_from_rerm_csv('CLUSTER_RESULTS/outcome_beta1_0_cov_region_seed_10.csv')
    print(results1)
    print('Confounder = region, beta0 = 1, beta1 = 1')
    results2 = ate_from_rerm_csv('CLUSTER_RESULTS/outcome_beta1_1_cov_region_seed_10.csv')
    print(results2)
    print('Confounder = region, beta0 = 1, beta1 = 10')
    results3 = ate_from_rerm_csv('CLUSTER_RESULTS/outcome_beta1_10_cov_region_seed_11.csv')
    print(results3)
    print('Confounder = age, beta0 = 1, beta1 = 0')
    results4 = ate_from_rerm_csv('CLUSTER_RESULTS/outcome_beta1_0_cov_age_seed_10.csv')
    print(results4)
    print('Confounder = age, beta0 = 1, beta1 = 1')
    results5 = ate_from_rerm_csv('CLUSTER_RESULTS/outcome_beta1_1_cov_age_seed_10.csv')
    print(results5)
    print('Confounder = age, beta0 = 1, beta1 = 10')
    results6 = ate_from_rerm_csv('CLUSTER_RESULTS/outcome_beta1_10_cov_age_seed_10.csv')
    print(results6)

    print('Confounder = registration, beta0 = 1, beta1 = 0')
    results7 = ate_from_rerm_csv('CLUSTER_RESULTS/outcome_beta1_0_cov_registration_seed_11.csv')
    print(results7)
    print('Confounder = registration, beta0 = 1, beta1 = 1')
    results8 = ate_from_rerm_csv('CLUSTER_RESULTS/outcome_beta1_1_cov_registration_seed_10.csv')
    print(results8)
    print('Confounder = registration, beta0 = 1, beta1 = 10')
    results9 = ate_from_rerm_csv('CLUSTER_RESULTS/outcome_beta1_10_cov_registration_seed_10.csv')
    print(results9)

    ### Results for Experiment Section 6.2 - Binary outcome
    print('Confounder = region')
    results1 = ate_binary_from_rerm_csv('CLUSTER_RESULTS/binary_outcome_covariate_region_seed_10.csv')
    print(results1)

    print('Confounder = age')
    results2 = ate_binary_from_rerm_csv('CLUSTER_RESULTS/binary_outcome_covariate_age_seed_10.csv')
    print(results2)

    print('Confounder = registration')
    results3 = ate_binary_from_rerm_csv('CLUSTER_RESULTS/binary_outcome_covariate_registration_seed_10.csv')
    print(results3)



if __name__ == "__main__":
    main()
