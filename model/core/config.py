##############################################################################
#                                                                            #
#  Code for the USENIX Security '24 paper:                                   #
#  Code is not Natural Language: Unlock the Power of Semantics-Oriented      #
#             Graph Representation for Binary Code Similarity Detection      #
#                                                                            #
#  MIT License                                                               #
#                                                                            #
#  Copyright (c) 2023 SJTU NSSL Lab                                     #
#                                                                            #
#----------------------------------------------------------------------------#
#  This implementation contains code from:                                   #
#  https://github.com/Cisco-Talos/binary_function_similarity/tree/main       #
#                                                                            #
#  Licensed under MIT License                                                #
#                                                                            #
#  Copyright (c) 2019-2022 Cisco Talos                                       #
#                                                                            #
#  Permission is hereby granted, free of charge, to any person obtaining     #
#  a copy of this software and associated documentation files (the           #
#  "Software"), to deal in the Software without restriction, including       #
#  without limitation the rights to use, copy, modify, merge, publish,       #
#  distribute, sublicense, and/or sell copies of the Software, and to        #
#  permit persons to whom the Software is furnished to do so, subject to     #
#  the following conditions:                                                 #
#                                                                            #
#  The above copyright notice and this permission notice shall be            #
#  included in all copies or substantial portions of the Software.           #
#                                                                            #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,           #
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF        #
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                     #
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE    #
#  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION    #
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION     #
#  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.           #
#----------------------------------------------------------------------------#
#  This implementation contains code from:                                   #
#  https://github.com/deepmind/deepmind-research/blob/master/                #
#    graph_matching_networks/graph_matching_networks.ipynb                   #
#    licensed under Apache License 2.0                                       #
##############################################################################


from os.path import join, basename
import json
import os
import random
import string
import coloredlogs
import logging

LOG_NAME = 'gnn'
log = logging.getLogger(LOG_NAME)


def random_str(size):
    chars = string.ascii_letters + string.digits
    return ''.join(random.sample(chars, size))


def set_logger_filehandler(log_name, debug, outputdir, mode="train"):
    """
    Set logger level, syntax, and logfile

    Args:
        debug: if True, set the log level to DEBUG
        outputdir: path of the output directory for the logfile
    """
    log = logging.getLogger(log_name)

    fh = logging.FileHandler(os.path.join(
        outputdir, '{}_{}_{}.log'.format(log_name, mode, random_str(4))))
    fh.setLevel(logging.DEBUG)

    fmt = '%(asctime)s %(levelname)s:: %(message)s'
    formatter = coloredlogs.ColoredFormatter(fmt)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    if debug:
        loglevel = 'DEBUG'
    else:
        loglevel = 'INFO'
    coloredlogs.install(fmt=fmt,
                        datefmt='%H:%M:%S',
                        level=loglevel,
                        logger=log)
    return log, fh


def dump_config_to_json(config, outputdir):
    """
    Dump the configuration file to JSON

    Args:
        config: a dictionary with model configuration
        outputdir: path of the output directory
    """
    with open(os.path.join(outputdir, "config.json"), "w") as f_out:
        json.dump(config, f_out)
    return


def load_config_from_json(workingdir):
    """
    Load the configuration file of current workdir. 

    Args:
        workingdir: path of the workdir
    """
    config_fp = os.path.join(workingdir, "config.json")
    with open(config_fp, "r") as f:
        config = json.load(f)
    return config


def update_config_datasetone(config_dict, inputdir, outputdir, featuresdir, feature_json_name):
    """Config for Dataset-1."""
    inputdir = os.path.join(inputdir, "Dataset-1")

    # Training
    func_csv_fn = "training_Dataset-1.csv"
    config_dict['training']['df_train_path'] = \
        os.path.join(inputdir, func_csv_fn)
    config_dict['training']['features_train_path'] = \
        os.path.join(
            featuresdir, 'Dataset-1_training',
            feature_json_name)

    # Validation
    valdir = os.path.join(inputdir, "pairs", "validation")
    config_dict['validation'] = dict(
        func_info_csv_path=os.path.join(valdir, "validation_functions.csv"),
        features_validation_path=os.path.join(
            featuresdir, 'Dataset-1_validation',
            feature_json_name)
    )

    # Testing
    testdir = os.path.join(inputdir, "pairs", "testing")
    config_dict['testing'] = dict(
        infer_tasks=[
            (
                os.path.join(inputdir, "testing_Dataset-1.csv"),
                os.path.join(outputdir, "testing_Dataset-1.pkl"),
            )
        ],
        features_testing_path=os.path.join(
            featuresdir, 'Dataset-1_testing',
            feature_json_name)
    )


def update_config_datasetrtos(config_dict, inputdir, outputdir, featuresdir, feature_json_name):
    """Config for Dataset-RTOS."""
    testdir = os.path.join(inputdir, "Dataset-RTOS-525")
    config_dict['testing'] = dict(
        infer_tasks=[
            (
                os.path.join(testdir, "testing_Dataset-RTOS-525.csv"),
                os.path.join(outputdir, "testing_Dataset-RTOS-525.pkl"),
            )
        ],
        features_testing_path=os.path.join(
            featuresdir, 'Dataset-RTOS_testing',
            feature_json_name)
    )


def get_config(args):
    """The default configs."""
    NODE_STATE_DIM = 32
    EDGE_STATE_DIM = 8
    GRAPH_REP_DIM = 128

    config_dict = dict(
        ggnn_net=dict(
            n_node_feat_dim=NODE_STATE_DIM,
            n_edge_feat_dim=EDGE_STATE_DIM,
            layer_groups=[1, 1, 1, 1, 1, 1],
            n_message_net_layers=3,
            skip_mode=0, # 0<-no skip, 1<-trival add, 2<-adap skip
            output_mode=0, # 0<-plain, 1<-concat hiddens, 2<-attention
            concat_skip=0,
            num_query=4, # used when output_mode==2
            n_atte_layers=0, # used when output_mode==2
            layer_aggr="add",
            layer_aggr_kwargs=dict()
        ),
        encoder=dict(
            name='embed',
            hbmp=dict(
                hbmp_config=dict(
                    embed_size=500+2,
                    embed_dim=32,
                    hidden_dim=16,
                    seq_limit=48,
                    layers=1,
                    dropout=0.05,
                    proj_size=0,
                )
            ),
            gru=dict(
                c=dict(
                    embed_size=500+2,
                    embed_dim=32,
                    hidden_dim=16,
                    seq_limit=48,
                    layers=1,
                    proj_size=0,
                )
            ),
            mlp=dict(
                n_node_feat_dim=NODE_STATE_DIM,
                n_edge_feat_dim=EDGE_STATE_DIM,
            ),
            embed=dict(
                n_node_feat_dim=NODE_STATE_DIM,
                n_edge_feat_dim=EDGE_STATE_DIM,
                n_node_attr=195,
                n_edge_attr=4,
                n_pos_enc=4,
            ),
        ),
        aggr=dict(
            name='msoftv2',
            gated=dict(
                n_node_trans=1,
                n_hidden_channels=GRAPH_REP_DIM * 2,
                n_out_channels=GRAPH_REP_DIM,
                gated=True,
            ),
            softmax=dict(
                out_channels=GRAPH_REP_DIM,
                n_graph_trans=0,
            ),
            msoftv2=dict(
                num_querys=6,
                hidden_channels=GRAPH_REP_DIM//2,
                n_node_trans=1,
                n_agg_trans=1,
                q_scale=1.,
                out_method='lin',
            ),
            set2set=dict(
                out_channels=GRAPH_REP_DIM,
                processing_steps=5,
                num_layers=1,
            ),
            ## Settrans cost too much memory ...
            ## Needing modification.
            settrans=dict(
                channels=NODE_STATE_DIM,
                num_seed_points=4,
                num_encoder_blocks=1,
                num_decoder_blocks=0,
                heads=1,
                concat=True,
                layer_norm=True,
                dropout=0.,
            ),
        ),

        used_subgraphs=[1, 2, 3, 4],
        max_vertices=-1,
        edge_feature_dim=EDGE_STATE_DIM,

        training=dict(
            # Choices: ['pair', 'triplet', 'batch', 'batch_pair']
            mode="batch_pair",
            # Alternative is 'hamming' ('margin' == -euclidean)
            loss='cosine',
            opt='circle',
            gama=2.,
            margin=0.40,
            norm_neg_sampling_s=1.0,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in
            # the model we can add `snt.LayerNorm` to the outputs of each layer
            # , the aggregated messages and aggregated node representations to
            # keep the network activation scale in  reasonable range.
            graph_vec_regularizer_weight=1e-8,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            learning_rate=1e-3,
            weight_decay=0.0,
            # decay_steps=10000,
            # decay_rate=1.0,
            num_epochs=args.num_epochs,
            batch_size=40,
            max_num_nodes=220000,
            max_num_edges=400000,
            n_sim_funcs=4,
            epoh_tolerate=1e+7,
            clean_cache_after=1e+7,
            evaluate_after=1,
            print_after=1250),
        validation=dict(),
        testing=dict(),
        tunning=dict(
            run_test=["last"],
        ),
        outputdir=args.outputdir,

        device=args.device,
        batch_size=200,
        checkpoint_dir=args.outputdir,
        seed=11
    )
    update_config(config_dict, args)

    return config_dict


def update_config(config_dict, args):
    if args.dataset == 'one':
        update_config_datasetone(
            config_dict, args.inputdir, args.outputdir, args.featuresdir, args.feature_json_name)
    elif args.dataset == 'rtos':
        update_config_datasetrtos(
            config_dict, args.inputdir, args.outputdir, args.featuresdir, args.feature_json_name)

    if args.device is not None:
        config_dict['device'] = args.device
    return config_dict


def stat_model(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel()
                        for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

# t, tr = stat_model(self._model)
# print(f"Total num of parameters: {t}")
# print(f"Total trainable num of parameters: {tr}")
