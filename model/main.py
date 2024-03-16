#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
##############################################################################

import argparse
import collections
import os
import json

from datetime import datetime

from os.path import join, basename, isdir
from core import (
    GNNModel,
    dump_config_to_json, load_config_from_json,
    get_config, update_config, set_logger_filehandler, LOG_NAME
)

log = None


def generate_output_dir(args, base_output_dir="outputs", custom_desc=None):
    feature_name = (basename(args.featuresdir) +
                    "-") if args.featuresdir is not None else ""
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%m%d_%H%M%S")
    custom_desc = custom_desc + '-' if custom_desc is not None else ''
    outdirname = f"Output-{custom_desc}{feature_name}{currentTime}"
    outputdir = join(base_output_dir, outdirname)
    return outputdir


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def iter_configs(args):
    COMMAN = "common"
    with open(args.config, "r") as f:
        tunning_params = json.load(f)
    base_output_dir = args.outputdir if args.outputdir is not None else "Results"
    common_config = {}
    if COMMAN in tunning_params:
        common_config = tunning_params[COMMAN]
        del tunning_params[COMMAN]
    log_fh = None
    global log
    for desc, params in tunning_params.items():
        repeating = params.get('repeating', 1)
        args.outputdir = None
        if "args" in common_config:
            for key, val in common_config['args'].items():
                setattr(args, key, val)
        if "args" in params:
            for key, val in params['args'].items():
                setattr(args, key, val)

        if args.featuresdir is None or not isdir(args.featuresdir):
            print("[!] Non existing featuresdir: {}".format(args.featuresdir))
            return

        if args.outputdir is None:
            args.outputdir = generate_output_dir(args, base_output_dir, desc)

        # Create the output directory
        if not isdir(args.outputdir):
            os.mkdir(args.outputdir)
            print("Created outputdir: {}".format(args.outputdir))

        testing = params.get('is_testing', False)
        item_outputdir = args.outputdir
        for idx in range(repeating):
            if repeating > 1:
                args.outputdir = join(item_outputdir, idx)
                os.mkdir(args.outputdir)
            
            # Create logger
            if log_fh is not None:
                log.removeHandler(log_fh)
            log, log_fh = set_logger_filehandler(LOG_NAME, args.debug,
                                                 args.outputdir, "tunning")

            if testing:
                config = load_config_from_json(args.outputdir)
                config = update_config(config, args)
            else:
                config = get_config(args)
            update(config, common_config)
            update(config, params)

            yield config, desc, testing


def model_train(gnn_model, config, desc):
    log.info("Running model training")
    dump_config_to_json(config, config['outputdir'])
    best_val_auc, best_val_ckpt, last_ckpt = gnn_model.model_train()
    log.info(f"{desc}:\t{best_val_auc}")

    # Reload checkpoints for testing.
    for test_desc in config['tunning']['run_test']:
        if 'best' == test_desc:
            gnn_model.restore_model(best_val_ckpt)
            gnn_model.model_test(test_desc)
        elif 'last' == test_desc:
            gnn_model.restore_model(last_ckpt)
            gnn_model.model_test(test_desc)
        elif isinstance(test_desc, int):
            gnn_model.restore_model_at_epoch(test_desc)
            gnn_model.model_test(f"ckpt_{test_desc}")


def model_test(gnn_model, test_outdir):
    log.info("Running model testing")
    gnn_model.model_test(test_outdir)


def main():
    parser = argparse.ArgumentParser(
        prog='HermesSim-Network',
        description='The GGNN models for HermesSim. ',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-d', '--debug', action='store_true',
                        help='Log level debug')

    parser.add_argument("--inputdir", default="/dbs",
                        help="Path to the Input dir (DBs)")

    # featuresdir and feature_json_name can be specified in \
    #       either commandline arguments or configure files.
    parser.add_argument("--featuresdir", required=False, default=None,
                        help="Path to the Preprocessing dir")

    parser.add_argument("--feature_json_name", required=False, default=None,
                        help="Name of the feature dict json file. ")

    parser.add_argument("--device", default="cuda",
                        help="Device used. ")

    parser.add_argument('--num_epochs', type=int,
                        required=False, default=20,
                        help='Number of training epochs')

    parser.add_argument('--dataset', required=True,
                        choices=['one', 'rtos'],
                        help='Choose the dataset to use for the train or test')

    parser.add_argument('--config', required=True,
                        help='Tunning params to be loaded. Given as a json file path. ')

    parser.add_argument('--test_outdir', required=False, default=None,
                        help='Subdir to save the test output')

    parser.add_argument('--label', required=False, default=None,
                        help='label are part of the output dir name. ')

    parser.add_argument('-o', '--outputdir', required=False, default=None,
                        help='Output dir')

    args = parser.parse_args()

    if args.outputdir is None:
        args.outputdir = generate_output_dir(args, custom_desc=args.label)

    # Create the output directory
    if not isdir(args.outputdir):
        os.mkdir(args.outputdir)
        print("Created outputdir: {}".format(args.outputdir))

    # Iter config items
    for config, desc, testing in iter_configs(args):
        # Setup GNNModel.
        gnn_model = GNNModel(config)

        if testing:
            model_test(gnn_model, args.test_outdir)
        else:
            model_train(gnn_model, config, desc)


if __name__ == '__main__':
    main()
