import argparse
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from relational_erm.data_cleaning.pokec import load_data_pokec, process_pokec_attributes
from relational_erm.sampling import adapters, factories

from scipy.special import expit
import numpy as np
import numpy.random as npr
import pandas as pd
import scipy.stats as stats
import os
from fnmatch import fnmatch
from sklearn.linear_model import LogisticRegression, LinearRegression
import statsmodels.api as sm

def add_parser_sampling_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--proportion-censored', type=float, default=0.5,
                        help='proportion of censored vertex labels at train time.')
    parser.add_argument('--batch-size', type=int, default=100, help='minibatch size')
    parser.add_argument('--dataset-shards', type=int, default=None, help='dataset parallelism')
    parser.add_argument('--sampler', type=str, default=None, choices=factories.dataset_names(),
                        help='The sampler to use. biased-walk gives a skipgram random-walk with unigram negative '
                             'sampling; p-sampling gives p-sampling with unigram negative sampling; uniform-edge '
                             'gives uniform edge sampling with unigram negative sampling; biased-walk-induced-uniform '
                             'gives induced random-walk with unigram negative-sampling; p-sampling-induced gives '
                             'p-sampling with induced non-edges.')
    parser.add_argument('--sampler-test', type=str, default=None,
                        choices=factories.dataset_names(),
                        help='if not None, the sampler to use for testing')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--indef-ip', default=False, action='store_true',
                        help='Uses a krein inner product instead of the regular inner product.')
    parser.add_argument('--num-edges', type=int, default=800,
                        help='Number of edges per sample.')
    parser.add_argument('--p-sample-prob', type=float, default=None,
                        help='Probability of sampling a vertex for p-sampling. Only used if the sampling scheme is a '
                             'p-sampling scheme, in which case this is used to override the num-edges argument.')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')
    parser.add_argument('--num-negative', type=int, default=5,
                        help='negative examples per vertex for negative sampling')
    parser.add_argument('--num-negative-total', type=int, default=None,
                        help='total number of negative vertices sampled')
    parser.add_argument('--embedding_learning_rate', type=float, default=0.025,
                        help='SGD learning rate for embedding updates.')
    parser.add_argument('--global_learning_rate', type=float, default=1.,
                        help='SGD learning rate for global updates.')
    parser.add_argument('--global_regularization', type=float, default=1.,
                        help='Regularization scale for global variables.')
    return parser


###Several functions for getting the dataset in the right tf.data.Dataset format
def compose(*fns):
    """ Composes the given functions in reverse order.
    Parameters
    ----------
    fns: the functions to compose
    Returns
    -------
    comp: a function that represents the composition of the given functions.
    """
    import functools

    def _apply(x, f):
        if isinstance(x, tuple):
            return f(*x)
        else:
            return f(x)

    def comp(*args):
        return functools.reduce(_apply, fns, args)

    return comp


def get_dataset_fn(sampler, args):
    if sampler is None:
        sampler = 'biased-walk'

    return factories.make_dataset(sampler, args)


def make_test_fn(graph_data, args, treatments, outcomes, dataset_fn=None, num_samples=None, is_test=False):
    def input_fn():

        dataset = dataset_fn(graph_data, args.seed)

        data_processing = adapters.compose(
            adapters.relabel_subgraph(),
            adapters.append_vertex_labels(treatments, 'treatment'),
            adapters.append_vertex_labels(outcomes, 'outcome'),
            adapters.make_split_vertex_labels(
                graph_data.num_vertices, args.proportion_censored,
                np.random.RandomState(args.seed), is_test, is_pred=False),
            adapters.add_sample_size_info(),
            adapters.format_features_labels1())

        dataset = dataset.map(data_processing, 8)
        if num_samples is not None:
            dataset = dataset.take(num_samples)

        batch_size = args.batch_size
        num_edges = args.num_edges

        if batch_size is not None:
            dataset = dataset.apply(
                adapters.padded_batch_samples_supervised(batch_size, n_edges_max=num_edges * 2,
                                                         n_vertices_max=num_edges,
                                                         t_dtype=treatments.dtype, o_dtype=outcomes.dtype))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    return input_fn


def make_no_graph_input_fn0(graph_data, args, treatments, outcomes, filter_test=False):
    """
    A dataset w/ all the label processing, but no graph structure.
    Used at evaluation and prediction time

    """

    def input_fn():
        metadata = {'edge_list': tf.expand_dims(np.zeros((graph_data.num_vertices, 2)), 1),
                    'vertex_index': tf.expand_dims(np.array(range(graph_data.num_vertices)), 1),
                    'treatment': tf.expand_dims(np.ones(graph_data.num_vertices), 1),
                    'weights': graph_data.weights,
                    'is_positive': tf.expand_dims(np.ones(graph_data.num_vertices), 1)}

        num_samples = graph_data.num_vertices

        def gen():
            for i in range(num_samples):
                ls = {}
                for key, val in metadata.items():
                    ls[key] = val[i]
                yield ls

        dataset = tf.data.Dataset.from_generator(gen, output_types={k: tf.int64 for k in metadata})
        # dataset = dataset.shuffle(buffer_size=num_samples, reshuffle_each_iteration=True)
        data_processing = adapters.compose(
            adapters.append_vertex_labels(treatments, 'treatment'),
            adapters.append_vertex_labels(metadata['weights'], 'weights'),
            adapters.append_vertex_labels(outcomes, 'outcome'),
            adapters.make_split_vertex_labels(
                graph_data.num_vertices, args.proportion_censored,
                np.random.RandomState(args.seed), is_test=filter_test, is_pred=True),
            adapters.format_features_labels())

        dataset = dataset.map(data_processing, 8)

        if filter_test:
            def filter_test_fn(features, labels):
                return tf.equal(tf.squeeze(features['in_test']), 1)

            dataset = dataset.filter(filter_test_fn)
        batch_size = 78982 #args.num_edges
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

        return dataset

    return input_fn


### SIMULATING TREATMENT/OUTCOME VARIABLES:

def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig


def simulate_y(propensities, treatment, beta0=1.0, beta1=1.0, gamma=1.0):
    confounding = (propensities - 0.5).astype(np.float32)

    noise = np.random.normal(0., 1., size=propensities.shape[0]).astype(np.float32)

    y0 = beta1 * confounding
    y1 = beta0 * treatment + y0
    y = y1 + gamma * noise
    # y = (1. - treatment) * y0 + treatment * y1 + gamma * noise

    return y, y0, y1

def simulate_y_binary(propensities, treatment, beta0=1.0, beta1=1.0, gamma=1.0):
    confounding = (propensities - 0.5).astype(np.float32)

    noise = np.random.normal(0., 1., size=propensities.shape[0]).astype(np.float32)

    y0 = beta1 * confounding
    y1 = beta0 * treatment + y0
    y = y1 + gamma * noise
    y = np.random.binomial(1, sigmoid(y))


    # y = (1. - treatment) * y0 + treatment * y1 + gamma * noise

    return y


def simulate_from_pokec_covariate(data_dir, covariate='region', beta0=1.0, beta1=1.0, gamma=1.0):
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
    treatment = np.random.binomial(1, propensities)

    treatment_agg = np.empty(shape=(len(treatment)), dtype=np.float32)
    for i in range(len(treatment)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        # lst = np.append(i, neighbours)
        treatment_agg[i] = np.mean(treatment[neighbours], dtype=np.float32)
    treatment = treatment_agg

    y, y0, y1 = simulate_y(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma)
    t = treatment.astype(np.float32)
    y = y.astype(np.float32)
    y0 = y0.astype(np.float32)
    y1 = y1.astype(np.float32)

    return t, y, y0, y1, propensities


def simulate_from_pokec_covariate_treatment_all0(data_dir, covariate='region', beta0=1.0, beta1=1.0, gamma=1.0):
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

    treatment_agg = np.empty(shape=(len(treatment)), dtype=np.float32)
    for i in range(len(treatment)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        # lst = np.append(i, neighbours)
        treatment_agg[i] = np.mean(treatment[neighbours], dtype=np.float32)
    treatment = treatment_agg

    y, y0, y1 = simulate_y(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma)
    t = treatment.astype(np.float32)
    y = y.astype(np.float32)
    y0 = y0.astype(np.float32)
    y1 = y1.astype(np.float32)

    return t, y, y0, y1


def simulate_from_pokec_covariate_treatment_all1(data_dir, covariate='region', beta0=1.0, beta1=1.0, gamma=1.0):
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

    treatment_agg = np.empty(shape=(len(treatment)), dtype=np.float32)
    for i in range(len(treatment)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        # lst = np.append(i, neighbours)
        treatment_agg[i] = np.mean(treatment[neighbours], dtype=np.float32)
    treatment = treatment_agg

    y, y0, y1 = simulate_y(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma)
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

def simulate_from_pokec_covariate_treatment_label(data_dir, covariate='region', beta0=1.0, beta1=1.0, gamma=1.0):
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
    treatment = np.random.binomial(1, propensities)
    y = treatment
    treatment_agg = np.empty(shape=(len(treatment)), dtype=np.float32)
    for i in range(len(treatment)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        # lst = np.append(i, neighbours)
        treatment_agg[i] = np.mean(treatment[neighbours], dtype=np.float32)
    treatment = treatment_agg
    #y = simulate_y_binary(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma)
    t = treatment.astype(np.float32)
    y = y.astype(np.float32)

    return t, y


def simulate_from_pokec_covariate_treatment_all0_treatment_label(data_dir, covariate='region', beta0=1.0, beta1=1.0, gamma=1.0):
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
    treatment_agg = np.empty(shape=(len(treatment)), dtype=np.float32)
    for i in range(len(treatment)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        # lst = np.append(i, neighbours)
        treatment_agg[i] = np.mean(treatment[neighbours], dtype=np.float32)
    treatment = treatment_agg
    y = simulate_y_binary(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma)
    t = treatment.astype(np.float32)
    y = y.astype(np.float32)

    return t, y


def simulate_from_pokec_covariate_treatment_all1_treatment_label(data_dir, covariate='region', beta0=1.0, beta1=1.0, gamma=1.0):
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

    treatment_agg = np.empty(shape=(len(treatment)), dtype=np.float32)
    for i in range(len(treatment)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        # lst = np.append(i, neighbours)
        treatment_agg[i] = np.mean(treatment[neighbours], dtype=np.float32)
    treatment = treatment_agg
    y = simulate_y_binary(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma)
    t = treatment.astype(np.float32)
    y = y.astype(np.float32)

    return t, y



###CREATING AND GETTING A DATA SAMPLE OF MULTIPLE SUBGRAPHS INTO THE RIGHT KERAS FORMAT FOR MODEL FITTING (UNSUPERVISED):


def create_dataset1(sampler, args, is_test_value):
    graph_data, profiles = load_data_pokec('dat/pokec/regional_subset')
    pokec_features = process_pokec_attributes(profiles)
    #treatments = pokec_features['I_like_books']

    data_dir = 'dat/pokec/regional_subset'
    t, y, y0, y1, propensities = simulate_from_pokec_covariate(data_dir, covariate='region', beta0=1.0,
                                                               beta1=1.0, gamma=1.0)

    treatments = t #pokec_features['I_like_books']
    outcomes = y  # pokec_features['relation_to_casual_sex']
    dataset_fn = get_dataset_fn(sampler, args)
    make_sample_generator = make_test_fn(graph_data, args, treatments, outcomes, dataset_fn, is_test=is_test_value)
    sample_generator = make_sample_generator()
    return sample_generator


def main():
    parser = add_parser_sampling_arguments()
    args = parser.parse_args()
    graph_data, profiles = load_data_pokec('dat/pokec/regional_subset')
    data_dir = 'dat/pokec/regional_subset'
    t, y = simulate_from_pokec_covariate_treatment_label(data_dir, covariate='age', beta0=1.0,
                                                               beta1=0.0, gamma=1.0)

    treatments = t
    outcomes = y
    make_prediction_generator = make_no_graph_input_fn0(graph_data, args, treatments, outcomes,
                                                        filter_test=False)  # make_no_graph_input_fn0(graph_data, args)
    prediction_generator = make_prediction_generator()
    itr = iter(prediction_generator)
    sample = next(itr)
    sbm_embedding = np.loadtxt('groups.txt')
    sbm_embedding = sbm_embedding[:, 1:]  # drop the first column of embedding
    sbm_embedding = sbm_embedding[sbm_embedding[:, 0].argsort()]
    sbm_embedding = sbm_embedding[:, 1:]
    outcomes = sample[1]['outcome']
    treatments = sample[0]['treatment']
    X = np.column_stack([treatments, sbm_embedding])
    Y = tf.squeeze(outcomes).numpy()
    n_obs = X.shape[0]
    index_1 = np.random.choice(np.arange(X.shape[0]), int(n_obs / 2), replace=False)
    Y = Y[index_1]
    mask = np.ones(n_obs, dtype=bool)
    mask[index_1] = False
    X = X[mask]
    Y = list(Y)
    X = list(X)
    log_reg = sm.Logit(Y, X).fit()

    print('sbm_estimate')
    print(log_reg.params[0])
    print('sbm_std')
    print(log_reg.bse[0])

    # n = X.shape[0]
    # train_prop = 0.5
    # train_idx = npr.choice(np.arange(n), int(train_prop * n), replace=False)
    # index_1 = train_idx
    # Y = Y[index_1]
    # X_tr = X[train_idx]
    # Y_tr = Y[train_idx]
    # X_tr_coef = sm.add_constant(X_tr)
    # model1 = sm.OLS(Y_tr, X_tr_coef).fit()
    # print('sbm_estimate')
    # print(model1.params[1])
    # print('sbm_std')
    # print(model1.bse[1])

    # parser = add_parser_sampling_arguments()
    # args = parser.parse_args()
    # graph_data, profiles = load_data_pokec('dat/pokec/regional_subset')
    # data_dir = 'dat/pokec/regional_subset'
    # t, y, y0, y1, propensities = simulate_from_pokec_covariate(data_dir, covariate='registration', beta0=1.0,
    #                                                            beta1=10.0, gamma=1.0)
    #
    # treatments = t
    # outcomes = y
    # make_prediction_generator = make_no_graph_input_fn0(graph_data, args, treatments, outcomes,
    #                                                     filter_test=False)  # make_no_graph_input_fn0(graph_data, args)
    # prediction_generator = make_prediction_generator()
    # itr = iter(prediction_generator)
    # sample = next(itr)
    # sbm_embedding = np.loadtxt('groups.txt')
    # sbm_embedding = sbm_embedding[:, 1:]  # drop the first column of embedding
    # sbm_embedding = sbm_embedding[sbm_embedding[:, 0].argsort()]
    # sbm_embedding = sbm_embedding[:, 1:]
    # outcomes = sample[1]['outcome']
    # treatments = sample[0]['treatment']
    # X = np.column_stack([treatments, sbm_embedding])
    # Y = tf.squeeze(outcomes).numpy()
    # n = X.shape[0]
    # train_prop = 0.5
    # train_idx = npr.choice(np.arange(n), int(train_prop * n), replace=False)
    # X_tr = X[train_idx]
    # Y_tr = Y[train_idx]
    # X_tr_coef = sm.add_constant(X_tr)
    # model1 = sm.OLS(Y_tr, X_tr_coef).fit()
    # print('sbm_estimate')
    # print(model1.params[1])
    # print('sbm_std')
    # print(model1.bse[1])

    #breakpoint()

if __name__ == "__main__":
    main()
