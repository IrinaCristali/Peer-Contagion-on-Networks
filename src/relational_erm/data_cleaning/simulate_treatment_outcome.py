import random

import numpy as np
import statsmodels.api as sm
import tensorflow as tf

from relational_erm.data_cleaning.pokec import load_data_pokec, process_pokec_attributes


### SIMULATING TREATMENT/OUTCOME VARIABLES:


def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig


def simulate_y(propensities, treatment, beta0=1.0, beta1=1.0, gamma=1.0, set_seed=42):
    confounding = (propensities - 0.5).astype(np.float32)
    np.random.seed(set_seed)
    noise = np.random.normal(0., 1., size=propensities.shape[0]).astype(np.float32)

    y0 = beta1 * confounding
    y1 = beta0 * treatment + y0
    y = y1 + gamma * noise

    return y, y0, y1


def simulate_y_binary(propensities, treatment, beta0=1.0, beta1=1.0, gamma=1.0, set_seed=42):
    confounding = (propensities - 0.5).astype(np.float32)
    np.random.seed(set_seed)
    noise = np.random.normal(0., 1., size=propensities.shape[0]).astype(np.float32)
    y0 = beta1 * confounding
    y1 = beta0 * treatment + y0
    y = y1 + gamma * noise
    y = np.random.binomial(1, sigmoid(y))
    return y


def simulate_from_pokec_covariate(data_dir, covariate='region', beta0=1.0, beta1=1.0, gamma=1.0, set_seed=42):
    graph_data, profiles = load_data_pokec(data_dir)
    pokec_features = process_pokec_attributes(profiles)
    np.random.seed(set_seed)
    # predictable covariates
    covs = ['scaled_registration',
            'scaled_age',
            'region']
    # 'sign_in_zodiac',
    # 'relation_to_casual_sex']

    # reindex region to 0, 1, 2
    region = pokec_features['region']
    region = np.searchsorted(np.unique(region), region) - 1.

    age = pokec_features['scaled_age']
    age_cat = np.where(age < 0., -1., 1.)
    age_cat[np.isnan(age)] = 0

    registration = pokec_features['scaled_registration']
    registration_cat = np.where(registration < -0.5, -1., 0.)
    registration_cat[registration > 0.5] = 1.

    if covariate == 'region':
        confounder = region
    elif covariate == 'age':
        confounder = age_cat
    elif covariate == 'registration':
        confounder = registration_cat
    else:
        raise Exception("covariate name not recognized")

    # simulate treatments and outcomes
    propensities = 0.5 + 0.35 * confounder
    treatment = np.random.binomial(1, propensities)

    treatment_agg = np.empty(shape=(len(treatment)), dtype=np.float32)
    for i in range(len(treatment)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        # lst = np.append(i, neighbours)
        treatment_agg[i] = np.mean(treatment[neighbours], dtype=np.float32)
    treatment = treatment_agg

    y, y0, y1 = simulate_y(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma, set_seed=set_seed)
    t = treatment.astype(np.float32)
    y = y.astype(np.float32)
    y0 = y0.astype(np.float32)
    y1 = y1.astype(np.float32)

    return t, y, y0, y1, propensities


def simulate_from_pokec_covariate_y_binary(data_dir, covariate='region', beta0=1.0, beta1=1.0, gamma=1.0, set_seed=42):
    graph_data, profiles = load_data_pokec(data_dir)
    pokec_features = process_pokec_attributes(profiles)
    np.random.seed(set_seed)
    # predictable covariates
    covs = ['scaled_registration',
            'scaled_age',
            'region']
    # 'sign_in_zodiac',
    # 'relation_to_casual_sex']

    # reindex region to 0, 1, 2
    region = pokec_features['region']
    region = np.searchsorted(np.unique(region), region) - 1.

    age = pokec_features['scaled_age']
    age_cat = np.where(age < 0., -1., 1.)
    age_cat[np.isnan(age)] = 0

    registration = pokec_features['scaled_registration']
    registration_cat = np.where(registration < -0.5, -1., 0.)
    registration_cat[registration > 0.5] = 1.

    if covariate == 'region':
        confounder = region
    elif covariate == 'age':
        confounder = age_cat
    elif covariate == 'registration':
        confounder = registration_cat
    else:
        raise Exception("covariate name not recognized")

    # simulate treatments and outcomes
    propensities = 0.5 + 0.35 * confounder
    treatment = np.random.binomial(1, propensities)

    treatment_agg = np.empty(shape=(len(treatment)), dtype=np.float32)
    for i in range(len(treatment)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        # lst = np.append(i, neighbours)
        treatment_agg[i] = np.mean(treatment[neighbours], dtype=np.float32)
    treatment = treatment_agg

    y = simulate_y_binary(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma, set_seed=set_seed)
    t = treatment.astype(np.float32)
    y = y.astype(np.float32)

    return t, y


def simulate_from_pokec_covariate_treatment_all0(data_dir, covariate='region', beta0=1.0, beta1=1.0, gamma=1.0,
                                                 set_seed=42):
    graph_data, profiles = load_data_pokec(data_dir)
    pokec_features = process_pokec_attributes(profiles)
    np.random.seed(set_seed)
    # predictable covariates
    covs = ['scaled_registration',
            'scaled_age',
            'region']
    # 'sign_in_zodiac',
    # 'relation_to_casual_sex']

    # reindex region to 0, 1, 2
    region = pokec_features['region']
    region = np.searchsorted(np.unique(region), region) - 1.

    age = pokec_features['scaled_age']
    age_cat = np.where(age < 0., -1., 1.)
    age_cat[np.isnan(age)] = 0

    registration = pokec_features['scaled_registration']
    registration_cat = np.where(registration < -0.5, -1., 0.)
    registration_cat[registration > 0.5] = 1.

    if covariate == 'region':
        confounder = region
    elif covariate == 'age':
        confounder = age_cat
    elif covariate == 'registration':
        confounder = registration_cat
    else:
        raise Exception("covariate name not recognized")

    # simulate treatments and outcomes
    propensities = 0.5 + 0.35 * confounder
    treatment = np.zeros(shape=len(propensities), dtype=np.float32)

    treatment_agg = np.empty(shape=(len(treatment)), dtype=np.float32)
    for i in range(len(treatment)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        # lst = np.append(i, neighbours)
        treatment_agg[i] = np.mean(treatment[neighbours], dtype=np.float32)
    treatment = treatment_agg

    y, y0, y1 = simulate_y(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma, set_seed=set_seed)
    t = treatment.astype(np.float32)
    y = y.astype(np.float32)
    y0 = y0.astype(np.float32)
    y1 = y1.astype(np.float32)

    return t, y, y0, y1


def simulate_from_pokec_covariate_treatment_all1(data_dir, covariate='region', beta0=1.0, beta1=1.0, gamma=1.0,
                                                 set_seed=42):
    graph_data, profiles = load_data_pokec(data_dir)
    pokec_features = process_pokec_attributes(profiles)
    np.random.seed(set_seed)
    # predictable covariates
    covs = ['scaled_registration',
            'scaled_age',
            'region']
    # 'sign_in_zodiac',
    # 'relation_to_casual_sex']

    # reindex region to 0, 1, 2
    region = pokec_features['region']
    region = np.searchsorted(np.unique(region), region) - 1.

    age = pokec_features['scaled_age']
    age_cat = np.where(age < 0., -1., 1.)
    age_cat[np.isnan(age)] = 0

    registration = pokec_features['scaled_registration']
    registration_cat = np.where(registration < -0.5, -1., 0.)
    registration_cat[registration > 0.5] = 1.

    if covariate == 'region':
        confounder = region
    elif covariate == 'age':
        confounder = age_cat
    elif covariate == 'registration':
        confounder = registration_cat
    else:
        raise Exception("covariate name not recognized")

    # simulate treatments and outcomes
    propensities = 0.5 + 0.35 * confounder
    treatment = np.ones(shape=len(propensities), dtype=np.float32)

    treatment_agg = np.empty(shape=(len(treatment)), dtype=np.float32)
    for i in range(len(treatment)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        # lst = np.append(i, neighbours)
        treatment_agg[i] = np.mean(treatment[neighbours], dtype=np.float32)
    treatment = treatment_agg

    y, y0, y1 = simulate_y(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma, set_seed=set_seed)
    t = treatment.astype(np.float32)
    y = y.astype(np.float32)
    y0 = y0.astype(np.float32)
    y1 = y1.astype(np.float32)

    return t, y, y0, y1


# def simulate_exogeneity_experiment(base_propensity_scores, exogeneous_con=0.,
#                                    beta0=1.0, beta1=1.0, gamma=1.0):
#     extra_confounding = np.random.normal(0, 1, base_propensity_scores.shape[0]).astype(np.float32)
#
#     propensities = expit((1. - exogeneous_con) * logit(base_propensity_scores) +
#                          exogeneous_con * extra_confounding).astype(np.float32)
#
#     treatment = np.random.binomial(1, propensities)
#     y, y0, y1 = simulate_y(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma)
#
#     t = treatment.astype(np.int32)
#     y = y.astype(np.float32)
#     y0 = y0.astype(np.float32)
#     y1 = y1.astype(np.float32)
#
#     return t, y, y0, y1, propensities


def simulate_from_pokec_covariate_treatment_label(data_dir, covariate='region', set_seed=2):
    graph_data, profiles = load_data_pokec(data_dir)
    pokec_features = process_pokec_attributes(profiles)

    # predictable covariates
    covs = ['scaled_registration',
            'scaled_age',
            'region']
    # 'sign_in_zodiac',
    # 'relation_to_casual_sex']

    # reindex region to 0, 1, 2
    region = pokec_features['region']
    region = np.searchsorted(np.unique(region), region) - 1.

    age = pokec_features['scaled_age']
    age_cat = np.where(age < 0., -1., 1.)
    age_cat[np.isnan(age)] = 0

    registration = pokec_features['scaled_registration']
    registration_cat = np.where(registration < -0.5, -1., 0.)
    registration_cat[registration > 0.5] = 1.

    if covariate == 'region':
        confounder = region
    elif covariate == 'age':
        confounder = age_cat
    elif covariate == 'registration':
        confounder = registration_cat
    else:
        raise Exception("covariate name not recognized")

    # simulate treatments and outcomes
    np.random.seed(set_seed)
    propensities = 0.5 + 0.35 * confounder
    treatment = np.random.binomial(1, propensities)
    y = treatment
    treatment_new = treatment[:].copy()
    indices = np.where(np.in1d(treatment_new, [1]))[0]
    n_obs = indices.shape[0]
    subset = random.sample(indices.tolist(), int(n_obs / 2))
    treatment_new[subset] = 0

    treatment_agg = np.empty(shape=(len(treatment_new)), dtype=np.float32)
    for i in range(len(treatment_new)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        treatment_agg[i] = np.mean(treatment_new[neighbours], dtype=np.float32)

    t = treatment_agg.astype(np.float32)
    y = y.astype(np.float32)

    return t, y


def simulate_from_pokec_covariate_treatment_all0_treatment_label(data_dir, covariate='region'):
    graph_data, profiles = load_data_pokec(data_dir)
    pokec_features = process_pokec_attributes(profiles)
    # predictable covariates
    covs = ['scaled_registration',
            'scaled_age',
            'region']
    # 'sign_in_zodiac',
    # 'relation_to_casual_sex']

    # reindex region to 0, 1, 2
    region = pokec_features['region']
    region = np.searchsorted(np.unique(region), region) - 1.

    age = pokec_features['scaled_age']
    age_cat = np.where(age < 0., -1., 1.)
    age_cat[np.isnan(age)] = 0

    registration = pokec_features['scaled_registration']
    registration_cat = np.where(registration < -0.5, -1., 0.)
    registration_cat[registration > 0.5] = 1.

    if covariate == 'region':
        confounder = region
    elif covariate == 'age':
        confounder = age_cat
    elif covariate == 'registration':
        confounder = registration_cat
    else:
        raise Exception("covariate name not recognized")

    # simulate treatments and outcomes
    propensities = 0.5 + 0.35 * confounder
    treatment = np.zeros(shape=len(propensities), dtype=np.float32)
    y = treatment
    treatment_agg = np.empty(shape=(len(treatment)), dtype=np.float32)
    for i in range(len(treatment)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        # lst = np.append(i, neighbours)
        treatment_agg[i] = np.mean(treatment[neighbours], dtype=np.float32)
    treatment = treatment_agg
    # y = simulate_y_binary(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma)
    t = treatment.astype(np.float32)
    y = y.astype(np.float32)

    return t, y


def simulate_from_pokec_covariate_treatment_all1_treatment_label(data_dir, covariate='region'):
    graph_data, profiles = load_data_pokec(data_dir)
    pokec_features = process_pokec_attributes(profiles)
    # predictable covariates
    covs = ['scaled_registration',
            'scaled_age',
            'region']
    # 'sign_in_zodiac',
    # 'relation_to_casual_sex']

    # reindex region to 0, 1, 2
    region = pokec_features['region']
    region = np.searchsorted(np.unique(region), region) - 1.

    age = pokec_features['scaled_age']
    age_cat = np.where(age < 0., -1., 1.)
    age_cat[np.isnan(age)] = 0

    registration = pokec_features['scaled_registration']
    registration_cat = np.where(registration < -0.5, -1., 0.)
    registration_cat[registration > 0.5] = 1.

    if covariate == 'region':
        confounder = region
    elif covariate == 'age':
        confounder = age_cat
    elif covariate == 'registration':
        confounder = registration_cat
    else:
        raise Exception("covariate name not recognized")

    # simulate treatments and outcomes
    propensities = 0.5 + 0.35 * confounder
    treatment = np.ones(shape=len(propensities), dtype=np.float32)
    y = treatment

    treatment_agg = np.empty(shape=(len(treatment)), dtype=np.float32)
    for i in range(len(treatment)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        # lst = np.append(i, neighbours)
        treatment_agg[i] = np.mean(treatment[neighbours], dtype=np.float32)
    treatment = treatment_agg
    # y = simulate_y_binary(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma)
    t = treatment.astype(np.float32)
    y = y.astype(np.float32)

    return t, y


def simulate_from_pokec_covariate_binary_region(data_dir, covariate='region', set_seed=2):
    graph_data, profiles = load_data_pokec(data_dir)
    pokec_features = process_pokec_attributes(profiles)

    # predictable covariates
    covs = ['scaled_registration',
            'scaled_age',
            'region']
    # 'sign_in_zodiac',
    # 'relation_to_casual_sex']

    # reindex region to 0, 1, 2
    region = pokec_features['region']
    region = np.searchsorted(np.unique(region), region) - 1.
    region[region == -1] = 1

    age = pokec_features['scaled_age']
    age_cat = np.where(age < 0., -1., 1.)
    age_cat[np.isnan(age)] = 0
    age_cat[age_cat == -1] = 1

    registration = pokec_features['scaled_registration']
    registration_cat = np.where(registration < -0.5, -1., 0.)
    registration_cat[registration > 0.5] = 1.
    registration_cat[registration_cat == -1] = 1

    if covariate == 'region':
        confounder = region
    elif covariate == 'age':
        confounder = age_cat
    elif covariate == 'registration':
        confounder = registration_cat
    else:
        raise Exception("covariate name not recognized")

    # simulate treatments and outcomes
    np.random.seed(set_seed)
    propensities = 0.5 + 0.35 * confounder
    treatment = np.random.binomial(1, propensities)
    y = treatment
    treatment_new = treatment[:].copy()
    indices = np.where(np.in1d(treatment_new, [1]))[0]
    n_obs = indices.shape[0]
    subset = random.sample(indices.tolist(), int(n_obs / 2))
    treatment_new[subset] = 0

    treatment_agg = np.empty(shape=(len(treatment_new)), dtype=np.float32)
    for i in range(len(treatment_new)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        treatment_agg[i] = np.mean(treatment_new[neighbours], dtype=np.float32)

    t = treatment_agg.astype(np.float32)
    y = y.astype(np.float32)
    confounder = confounder.astype(np.float32)
    return t, confounder


def main():
    tf.compat.v1.enable_eager_execution()
    data_dir = 'dat/pokec/regional_subset'
    t, y = simulate_from_pokec_covariate_treatment_label(data_dir, covariate='registration', set_seed=2)
    y = list(y)
    t = list(t)
    log_reg = sm.Logit(y, t).fit()
    unadjusted_ate = log_reg.params[0]
    print(unadjusted_ate)
    #t, y = simulate_from_pokec_covariate_binary_region(data_dir, covariate='region', set_seed=42)
    # t, y_all1, y0, y1 = simulate_from_pokec_covariate_treatment_all1(data_dir, covariate='region', beta0=1.0, beta1=1,
    #                                                                  gamma=1.0, set_seed=42)
    # t, y_all0, y0, y1 = simulate_from_pokec_covariate_treatment_all0(data_dir, covariate='region', beta0=1.0, beta1=1,
    #                                                                  gamma=1.0, set_seed=42)
    #
    # t, y, y0, y1, prop = simulate_from_pokec_covariate(data_dir, covariate='region', beta0=1.0, beta1=1,
    #                                                    gamma=1.0, set_seed=42)

if __name__ == '__main__':
    main()
