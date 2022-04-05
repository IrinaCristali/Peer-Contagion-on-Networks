import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression


def ate_from_rerm_csv(csv_path):
    output = pd.read_csv(csv_path, '\t')
    y_t0 = output['outcome_no_treatment']
    y_t1 = output['outcome_all_treatment']
    expected_y_t0 = output['expected_outcome_st_no_treatment']
    expected_y_t1 = output['expected_outcome_st_all_treatment']
    y = output['y']
    X = output[['expected_outcome_st_no_treatment', 'expected_outcome_st_all_treatment', 'v']]
    X = sm.add_constant(X)
    model1 = sm.OLS(y, X).fit()
    adjusted_ate = expected_y_t1.mean() - expected_y_t0.mean()  # model1.params[3]
    adjusted_ate_std = expected_y_t0.std() / np.sqrt(len(expected_y_t0)) + expected_y_t1.std() / np.sqrt(
        len(expected_y_t1))
    ground_truth_ate = y_t1.mean() - y_t0.mean()
    # unadjusted_ate_std = y_t0.std()/np.sqrt(len(y_t0)) + y_t1.std()/np.sqrt(len(y_t1))

    unadjusted_ate = model1.params['v']  # reg.coef_
    unadjusted_ate_std = model1.bse['v']
    # unadjusted_ate_std =   #np.sqrt(np.cov(output['y'],output['v'])[0][0])*np.sqrt(np.cov(output['y'],output['v'])[1][1])
    return ground_truth_ate, adjusted_ate, adjusted_ate_std, unadjusted_ate, unadjusted_ate_std  # reg.bse#unadjusted_ate_std


def ate_binary_from_rerm_csv(csv_path):
    output = pd.read_csv(csv_path, '\t')
    y_t0 = output['outcome_no_treatment']
    y_t1 = output['outcome_all_treatment']
    expected_y_t0 = output['expected_outcome_st_no_treatment']
    expected_y_t1 = output['expected_outcome_st_all_treatment']
    ground_truth_ate = y_t1.mean() - y_t0.mean()
    n_obs = output['y'].shape[0]
    index_1 = np.random.choice(np.arange(output['y'].shape[0]), int(n_obs / 2), replace=False)
    y = output['y'][index_1]
    mask = np.ones(n_obs, dtype=bool)
    mask[index_1] = False
    v = output['v'][mask]
    y = list(y)
    v = list(v)
    log_reg = sm.Logit(y, v).fit()
    adjusted_ate = expected_y_t1.mean() - expected_y_t0.mean()
    adjusted_ate_std = np.std(expected_y_t1 - expected_y_t0)#expected_y_t0.std() / np.sqrt(len(expected_y_t0)) + expected_y_t1.std() / np.sqrt(
        #len(expected_y_t1))
    unadjusted_ate = log_reg.params[0]
    unadjusted_ate_std = log_reg.bse[0]

    return ground_truth_ate, adjusted_ate, adjusted_ate_std, unadjusted_ate, unadjusted_ate_std


def main():
    print('Confounder = region')
    results1 = ate_binary_from_rerm_csv('binary_outcome_region.csv')
    print(results1)

    print('Confounder = age')
    results2 = ate_binary_from_rerm_csv('binary_outcome_age.csv')
    print(results2)

    print('Confounder = registration')
    results3 = ate_binary_from_rerm_csv('binary_outcome_registration.csv')
    print(results3)

    # print('Confounder = region, beta0 = 1, beta1 = 0')
    # results1 = ate_from_rerm_csv('outcome_region_beta0_1_beta1_0.csv')
    # print(results1)
    # print('Confounder = region, beta0 = 1, beta1 = 1')
    # results2 = ate_from_rerm_csv('outcome_region_beta0_1_beta1_1.csv')
    # print(results2)
    # print('Confounder = region, beta0 = 1, beta1 = 10')
    # results3 = ate_from_rerm_csv('outcome_region_beta0_1_beta1_10.csv')
    # print(results3)
    # print('Confounder = age, beta0 = 1, beta1 = 0')
    # results4 = ate_from_rerm_csv('outcome_age_beta0_1_beta1_0.csv')
    # print(results4)
    # print('Confounder = age, beta0 = 1, beta1 = 1')
    # results5 = ate_from_rerm_csv('outcome_age_beta0_1_beta1_1.csv')
    # print(results5)
    # print('Confounder = age, beta0 = 1, beta1 = 10')
    # results6 = ate_from_rerm_csv('outcome_age_beta0_1_beta1_10.csv')
    # print(results6)
    #
    # print('Confounder = registration, beta0 = 1, beta1 = 0')
    # results7 = ate_from_rerm_csv('outcome_registration_beta0_1_beta1_0.csv')
    # print(results7)
    # print('Confounder = registration, beta0 = 1, beta1 = 1')
    # results8 = ate_from_rerm_csv('outcome_registration_beta0_1_beta1_1.csv')
    # print(results8)
    # print('Confounder = registration, beta0 = 1, beta1 = 10')
    # results9 = ate_from_rerm_csv('outcome_registration_beta0_1_beta1_10.csv')
    # print(results9)
    breakpoint()


if __name__ == "__main__":
    main()
