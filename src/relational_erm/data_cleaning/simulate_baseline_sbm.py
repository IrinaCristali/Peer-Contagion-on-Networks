import argparse
import random

import numpy as np
import numpy.random as npr
import statsmodels.api as sm
import tensorflow as tf

from relational_erm.data_cleaning.pokec import load_data_pokec, process_pokec_attributes
from relational_erm.sampling import adapters, factories


def add_parser_sampling_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--proportion-censored', type=float, default=0,
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

        # if filter_test:
        #     def filter_test_fn(features, labels):
        #         return tf.equal(tf.squeeze(features['in_test']), 1)
        #
        #     dataset = dataset.filter(filter_test_fn)
        batch_size = 78982  # args.num_edges
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

        return dataset

    return input_fn


### SIMULATING TREATMENT/OUTCOME VARIABLES:

def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig


def simulate_y(propensities, treatment, beta0=1.0, beta1=1.0, gamma=1.0, set_seed=0):
    confounding = (propensities - 0.5).astype(np.float32)
    np.random.seed(set_seed)
    noise = np.random.normal(0., 1., size=propensities.shape[0]).astype(np.float32)

    y0 = beta1 * confounding
    y1 = beta0 * treatment + y0
    y = y1 + gamma * noise

    return y, y0, y1


def simulate_y_binary(propensities, treatment, beta0=1.0, beta1=1.0, gamma=1.0):
    confounding = (propensities - 0.5).astype(np.float32)

    noise = np.random.normal(0., 1., size=propensities.shape[0]).astype(np.float32)

    y0 = beta1 * confounding
    y1 = beta0 * treatment + y0
    y = y1 + gamma * noise
    y = np.random.binomial(1, sigmoid(y))
    return y


def simulate_from_pokec_covariate(data_dir, covariate='region', beta0=1.0, beta1=1.0, gamma=1.0, set_seed=0):
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


def simulate_from_pokec_covariate_treatment_label(data_dir, covariate='region', set_seed=2):
    np.random.seed(set_seed)
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
    np.random.seed(set_seed)
    treatment = np.random.binomial(1, propensities)
    y = treatment
    treatment_new = treatment[:].copy()
    indices = np.where(np.in1d(treatment_new, [1]))[0]
    n_obs = indices.shape[0]
    np.random.seed(set_seed)
    subset = random.sample(indices.tolist(), int(n_obs / 2))
    treatment_new[subset] = 0
    treatment_agg = np.empty(shape=(len(treatment_new)), dtype=np.float32)
    np.random.seed(set_seed)
    for i in range(len(treatment_new)):
        neighbours = graph_data.adjacency_list.get_neighbours(i)
        treatment_agg[i] = np.mean(treatment_new[neighbours], dtype=np.float32)

    t = treatment_agg
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


###CREATING AND GETTING A DATA SAMPLE OF MULTIPLE SUBGRAPHS INTO THE RIGHT KERAS FORMAT FOR MODEL FITTING (UNSUPERVISED):


def create_dataset1(sampler, args, is_test_value):
    graph_data, profiles = load_data_pokec('dat/pokec/regional_subset')
    pokec_features = process_pokec_attributes(profiles)
    # treatments = pokec_features['I_like_books']

    data_dir = 'dat/pokec/regional_subset'
    t, y, y0, y1, propensities = simulate_from_pokec_covariate(data_dir, covariate='region', beta0=1.0,
                                                               beta1=1.0, gamma=1.0)

    treatments = t  # pokec_features['I_like_books']
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
    #t, y, y0, y1, propensities = simulate_from_pokec_covariate(data_dir, covariate='registration', beta0=1.0, beta1=10.0, gamma=1.0, set_seed=0)
    t, y = simulate_from_pokec_covariate_treatment_label(data_dir, covariate='registration', set_seed=2)
    treatments = t
    outcomes = y
    sbm_embedding = np.loadtxt('groups.txt')
    sbm_embedding = sbm_embedding[:, 1:]  # drop the first column of embedding
    sbm_embedding = sbm_embedding[sbm_embedding[:, 0].argsort()]
    sbm_embedding = sbm_embedding[:, 1:]

    X = np.column_stack([treatments, sbm_embedding])
    Y = tf.squeeze(outcomes).numpy()

    model1 = sm.Logit(Y, X).fit()
    print('sbm_estimate')
    print(model1.params[0])
    print('sbm_std')
    print(model1.bse[0])


if __name__ == "__main__":
    main()
