### Model Training and Prediction Code for Experiments Section 6.1 - "Continuous outcome"
import argparse
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from relational_erm.data_cleaning.pokec import load_data_pokec
from relational_erm.data_cleaning.simulate_treatment_outcome import simulate_from_pokec_covariate, \
    simulate_from_pokec_covariate_treatment_all0, simulate_from_pokec_covariate_treatment_all1
from relational_erm.sampling import adapters, factories


def add_parser_sampling_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=50)
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
                        help='Probability of samping a vertex for p-sampling. Only used if the sampling scheme is a '
                             'p-sampling scheme, in which case this is used to override the num-edges argument.')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')
    parser.add_argument('--num-negative', type=int, default=5,
                        help='negative examples per vertex for negative sampling')
    parser.add_argument('--num-negative-total', type=int, default=None,
                        help='total number of negative vertices sampled')
    parser.add_argument('--beta_1', type=float, default=10,
                        help='beta_1 parameter')
    parser.add_argument('--covariate', type=str, default='registration',
                        help='covariate to use as "hidden" confounder')
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


def make_no_graph_input_fn0(graph_data, args, treatments, outcomes, filter_test=True):
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
        batch_size = args.num_edges
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

        return dataset

    return input_fn


###CREATING AND GETTING A DATA SAMPLE OF MULTIPLE SUBGRAPHS INTO THE RIGHT KERAS FORMAT FOR MODEL FITTING (UNSUPERVISED):

def create_train_dataset(sampler, args, is_test_value, graph_data, treatments, outcomes):
    dataset_fn = get_dataset_fn(sampler, args)
    make_sample_generator = make_test_fn(graph_data, args, treatments, outcomes, dataset_fn, is_test=is_test_value)
    sample_generator = make_sample_generator()
    return sample_generator


def create_predict_dataset(args, graph_data, treatments, outcomes):
    make_prediction_generator = make_no_graph_input_fn0(graph_data, args, treatments, outcomes,
                                                        filter_test=True)
    prediction_generator = make_prediction_generator()
    return prediction_generator


def create_predict_dataset_t0(args, graph_data, treatments, outcomes):
    make_prediction_generator = make_no_graph_input_fn0(graph_data, args, treatments, outcomes,
                                                        filter_test=True)
    prediction_generator = make_prediction_generator()
    return prediction_generator


def create_predict_dataset_t1(args, graph_data, treatments, outcomes):
    make_prediction_generator = make_no_graph_input_fn0(graph_data, args, treatments, outcomes,
                                                        filter_test=True)  # make_no_graph_input_fn0(graph_data, args)
    prediction_generator = make_prediction_generator()
    return prediction_generator


#
# ###UNSUPERVISED LEARNING MODEL
# def make_edge_model(num_vertices=80000, embedding_dim=128):
#     edge_list = tf.keras.Input(shape=[None, 2], dtype=tf.float32, name="canonical_edge_list")  # n_edges_max
#     embedding_fn = tf.keras.layers.Embedding(num_vertices, embedding_dim, input_length=None, mask_zero=True)
#     edge_list_start = edge_list[:, :, 0]  # batch, n_edges_max
#     edge_list_end = edge_list[:, :, 1]  # batch, n_edges_max
#     embeddings0 = embedding_fn(edge_list_start)  # batch, n_edges_max, 128
#     embeddings1 = embedding_fn(edge_list_end)  # batch, n_edges_max, 128
#     half1 = embeddings1[:, :, 0:64]
#     half2 = embeddings1[:, :, 64:128]
#     embed1 = tf.concat([half1, half2], 2)
#     product = embeddings0 * embed1  # batch, n_edges_max, 128
#     edge_predictions = tf.math.reduce_sum(product, axis=-1)  # batch, n_edges_max
#     return tf.keras.Model(
#         inputs=[{'edge_list': edge_list}],
#         outputs=[{'weights': edge_predictions}])


###SUPERVISED LEARNING MODEL:
def make_outcome_model(num_vertices=80000, embedding_dim=128):
    edge_list = tf.keras.Input(shape=[None, 2], dtype=tf.float32, name="canonical_edge_list")
    vertex_index = tf.keras.Input(shape=[None], dtype=tf.float32, name='vertex_list')
    treatment = tf.keras.Input(shape=[None], dtype=tf.float32, name='treatment')
    vert_mask = tf.keras.Input(shape=[None], dtype=tf.float32, name='vert_mask')
    parser = add_parser_sampling_arguments()
    args = parser.parse_args()
    embedding_fn = tf.keras.layers.Embedding(num_vertices, embedding_dim, input_length=None, mask_zero=True)

    # edge stuff
    # make predictions for each edge
    edge_list_start = edge_list[:, :, 0]  # batch, n_edges_max
    edge_list_end = edge_list[:, :, 1]  # batch, n_edges_max
    embeddings0 = embedding_fn(edge_list_start)  # batch, n_edges_max, 128
    embeddings1 = embedding_fn(edge_list_end)  # batch, n_edges_max, 128
    # parser = add_parser_sampling_arguments()
    # args = parser.parse_args()
    half1 = embeddings1[:, :, 0:64]
    half2 = embeddings1[:, :, 64:128]  # multiply by -1 if you want to add krein IP
    embed1 = tf.concat([half1, half2], 2)
    product = embeddings0 * embed1  # batch, n_edges_max, 128
    edge_predictions = tf.math.reduce_sum(product, axis=-1)  # batch, n_edges_max

    # vertex stuff
    vertex_ind = vertex_index * vert_mask
    vertex_embed = embedding_fn(vertex_ind)  # batch, n_vertices_max, 128

    # lin_layer0 = tf.keras.layers.Dense(units=1, name='forcing_treatment')
    # treatment_hat = tf.squeeze(lin_layer0(vertex_embed))

    treatment_embed_concatenate = tf.keras.layers.Concatenate(axis=2)([tf.expand_dims(treatment, 2), vertex_embed])

    lin_layer = tf.keras.layers.Dense(units=1, name='vertex_prediction')
    y_hat = tf.squeeze(lin_layer(treatment_embed_concatenate))

    return tf.keras.Model(
        inputs={'edge_list': edge_list, 'vertex_index': vertex_index, 'treatment': treatment, 'vert_mask': vert_mask},
        outputs={'weights': edge_predictions, 'outcome': y_hat})


###FITTING AND EVALUATING THE MODELS
def main():
    ### INITIALIZING ARGS
    session_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=4)
    parser = add_parser_sampling_arguments()
    args = parser.parse_args()
    graph_data, profiles = load_data_pokec('dat/pokec/regional_subset')

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Override num of edges if scheme is p-sampling
    if args.sampler is not None:
        if ("p-sampling" in args.sampler) and args.p_sample_prob is not None:
            args.num_edges = int((args.p_sample_prob ** 2)
                                 * np.size(graph_data.adjacency_list))

        if "induced" in args.sampler:
            args.num_negative = None

    #### MAKING THE MODEL

    m = make_outcome_model(num_vertices=80000, embedding_dim=128)
    m.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0045 * 100),
        # 0.005 is good gives 0.85 causal ef #0.0024 ->0.87
        loss=
        {'weights': tf.keras.losses.BinaryCrossentropy(from_logits=True),
         'outcome': 'mse'},
        loss_weights={'weights': 1, 'outcome': 0.005},  # 0.005
        metrics={'weights': tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None),
                 'outcome': [tf.keras.metrics.MeanSquaredError()]}
    )

    ### LOOKING AT A SUMMARY OF THE MODEL
    m.summary()

    beta_1 = args.beta_1
    cov = args.covariate

    ### FITTING THE MODEL AND PREDICTING:

    graph_data, profiles = load_data_pokec('dat/pokec/regional_subset')
    data_dir = 'dat/pokec/regional_subset'
    t, y, y0, y1, propensities = simulate_from_pokec_covariate(data_dir, covariate=cov, beta0=1.0,
                                                               beta1=beta_1, gamma=0.01, set_seed=0)
    train_data = create_train_dataset(sampler=args.sampler, args=args, is_test_value=False, graph_data=graph_data,
                                      treatments=t, outcomes=y)
    prediction_generator = create_predict_dataset(args=args, graph_data=graph_data, treatments=t, outcomes=y)
    t_0, y_0, y0, y1 = simulate_from_pokec_covariate_treatment_all0(data_dir, covariate=cov,
                                                                    beta0=1,
                                                                    beta1=beta_1, gamma=0.01, set_seed=0)
    prediction_generator_treatment_all0 = create_predict_dataset_t0(args=args, graph_data=graph_data, treatments=t_0,
                                                                    outcomes=y_0)
    t_1, y_1, y0, y1 = simulate_from_pokec_covariate_treatment_all1(data_dir, covariate=cov,
                                                                    beta0=1,
                                                                    beta1=beta_1, gamma=0.01, set_seed=0)
    prediction_generator_treatment_all1 = create_predict_dataset_t1(args=args, graph_data=graph_data, treatments=t_1,
                                                                    outcomes=y_1)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    m.fit(train_data,
          epochs=1,  # 13
          steps_per_epoch=1000,  # 100
          batch_size=100,
          callbacks=[tensorboard_callback],
          # validation_data=prediction_generator,
          # validation_steps=50,
          # shuffle=True
          )

    ### EVALUATE AND MAKE PREDICTIONS

    results = m.evaluate(prediction_generator, steps=10)
    predictions = m.predict(prediction_generator, steps=10)

    predictions0 = m.predict(prediction_generator_treatment_all0, steps=10)
    predictions1 = m.predict(prediction_generator_treatment_all1, steps=10)
    print(predictions1['outcome'].mean() - predictions0['outcome'].mean())

    out_dict = {}
    out_dict['expected_outcome_st_no_treatment'] = predictions0['outcome'].squeeze()
    out_dict['expected_outcome_st_all_treatment'] = predictions1['outcome'].squeeze()
    out_dict['outcome_no_treatment'] = []
    out_dict['outcome_all_treatment'] = []
    out_dict['y'] = []
    out_dict['v'] = []
    itr0 = iter(prediction_generator_treatment_all0)
    itr1 = iter(prediction_generator_treatment_all1)
    itr = iter(prediction_generator)

    print('Storing Simulated Outcome Values')

    for _ in range(10):
        sample0 = next(itr0)
        outcome0 = tf.squeeze(sample0[1]['outcome'])
        out_dict['outcome_no_treatment'] = np.append(out_dict['outcome_no_treatment'], outcome0)
        sample1 = next(itr1)
        outcome1 = tf.squeeze(sample1[1]['outcome'])
        out_dict['outcome_all_treatment'] = np.append(out_dict['outcome_all_treatment'], outcome1)
        sample = next(itr)
        y = tf.squeeze(sample[1]['outcome'])
        out_dict['y'] = np.append(out_dict['y'], y)
        v = tf.squeeze(sample[0]['treatment'])
        out_dict['v'] = np.append(out_dict['v'], v)

    predictions = pd.DataFrame(out_dict)
    predictions.to_csv(f'outcome_beta1_{beta_1}_cov_{cov}_seed_{args.seed}.csv', sep='\t', header='true')


if __name__ == "__main__":
    main()
