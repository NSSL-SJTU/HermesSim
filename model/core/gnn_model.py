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

import collections
from typing import Dict
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import scipy
import pickle

import torch as tf
# from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
from torch.nn.functional import relu

from os.path import basename, dirname, join
from scipy.stats import rankdata

from .build_dataset import *
from .ggnn import *
from .encoders import HBMPEncoder, GruEncoder
from .aggrs import *

from .loss import pairwise_loss
from .loss import triplet_loss
from .similarities import (
    pairwise_cosine_similarity,
    compute_similarity,
)

import logging
log = logging.getLogger('gnn')


def _it_check_condition(it_num, threshold):
    """
    Utility function to make the code cleaner.

    Args:
        it_num: the iteration number.
        threshold: threshold at which the condition must be verfied.

    Return:
        True if it_num +1 is a multiple of the threshold.
    """
    return (it_num + 1) % threshold == 0


def reshape_and_split_tensor(tensor, n_splits: int):
    """Reshape and split a 2D tensor along the last dimension.

    Args:
      tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
        multiple of `n_splits`.
      n_splits: int, number of splits to split the tensor into.

    Returns:
      splits: a list of `n_splits` tensors.  The first split is [tensor[0],
        tensor[n_splits], tensor[n_splits * 2], ...], the second split is
        [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
    """
    feature_dim = tensor.shape[-1]
    # feature dim must be known, otherwise you can provide that as an input
    assert isinstance(feature_dim, int)
    tensor = tensor.view(-1, feature_dim * n_splits)
    return torch.split(tensor, feature_dim, dim=-1)


def batch_to(batch_inputs, device, non_blocking=False):
    return [
        t.to(device, non_blocking=non_blocking)
        if isinstance(t, torch.Tensor) else t
        for t in batch_inputs]


class TrainHelper(torch.nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self._model = model
        self._train_mode = config['training']['mode']
        self._loss_type = config['training']['loss']
        self._loss_opt = config['training']['opt']
        self._loss_margin = config['training']['margin']
        self._loss_gama = config['training']['gama']
        self._gv_regular_w = config['training']['graph_vec_regularizer_weight']
        self._s = config['training']['norm_neg_sampling_s']
        self._k = config['training']['n_sim_funcs']
        assert not self._loss_type.startswith('batch') or self._k >= 2

    def forward(self, labels, x_dict, edge_index_dict, edge_feat, batch_dict, batch_size: int) -> Dict[str, torch.Tensor]:
        model = self._model
        loss_type = self._loss_type
        graph_vectors = model(
            x_dict, edge_index_dict, edge_feat, batch_dict, batch_size
        )
        rloss_mean, rloss_max = torch.tensor(0.), torch.tensor(0.)
        if self._train_mode == 'pair':
            x, y = reshape_and_split_tensor(graph_vectors, 2)
            loss = pairwise_loss(x, y, labels,
                                 loss_type=loss_type,
                                 margin=self._loss_margin)
            # torch.no_grad() is buggy under torch.jit.script
            # Replace it with detachs is ok
            # with torch.no_grad():
            # optionally monitor the similarity between positive and negative pairs
            is_pos = (labels == 1).to(dtype=torch.float32, device="cpu")
            is_neg = 1 - is_pos
            n_pos = tf.sum(is_pos)
            n_neg = tf.sum(is_neg)
            sim = compute_similarity(loss_type, x.detach(), y.detach()).cpu()
            sim_pos = tf.sum(sim * is_pos) / (n_pos + 1e-8)
            sim_neg = tf.sum(sim * is_neg) / (n_neg + 1e-8)
        elif self._train_mode == 'triplet':
            x_1, y, z = reshape_and_split_tensor(graph_vectors, 3)
            loss = triplet_loss(x_1, y, x_1, z,
                                loss_type=loss_type,
                                margin=self._loss_margin)
            rloss_mean = loss.detach().mean().cpu()
            rloss_max = loss.detach().max().cpu()
            if self._loss_gama < 1e+3:
                delta = 1. if self._loss_type.endswith('cosine') else 0.
                loss = torch.log(
                    (torch.exp(self._loss_gama * loss)-delta).sum() + 1.) / self._loss_gama
            sim_pos = tf.mean(compute_similarity(
                loss_type, x_1.detach(), y.detach())).cpu()
            sim_neg = tf.mean(compute_similarity(
                loss_type, x_1.detach(), z.detach())).cpu()
        elif self._train_mode.startswith('batch'):
            is_pair = self._train_mode.endswith('pair')
            is_cosine = self._loss_type.endswith('cosine')
            k = self._k
            embeds = graph_vectors
            m, n = graph_vectors.shape
            m_range = list(range(m))
            # dists: m x m matrix
            if self._loss_type == 'margin':
                dists = torch.cdist(embeds, embeds, p=2.)
            elif self._loss_type == 'cosine':
                dists = 1 - pairwise_cosine_similarity(embeds, embeds)
            else:
                raise Exception(
                    f"{self._loss_type} is not supported with batch loss. ")
            ss = [i & (-k) for i in m_range]
            se = [(i & (-k)) + k for i in m_range]
            # sim_dists: m x (k - 1) matrix
            # ifs inside [] are not supported by torch script
            sim_dists = dists[
                [[i] for i in m_range], # dists[m x 1, m x (k - 1)]
                [[j for j in range(ss[i], se[i]) if j != i] for i in m_range]]
            # diff_dists: m x (m - k)
            diff_dists = torch.stack([
                torch.cat((dists[i, :ss[i]], dists[i, se[i]:])) for i in m_range])
            if self._loss_opt == 'circle':
                if self._s is not None:
                    w = -(n-2)*diff_dists - ((n-3)/2) * \
                        (1-diff_dists*diff_dists/4)
                    w = torch.softmax(w * self._s, dim=1)
                    ## int(j[0].item()): the convertion to int
                    ##      is a workaround for the type check of the torch jit.
                    _i = torch.tensor(m_range).view(m, 1)
                    _j = torch.multinomial(w, k-1)
                    ## Using _i, _j as indices rather than [_i, _j] is
                    ##  a workaround of a pytorch jit bug, indexing tensor with [_i, _j]
                    ##  behaves differently under pytorch and torch jit.
                    ### loss: m x k matrix
                    if is_pair:
                        loss = relu(sim_dists - self._loss_margin) + \
                            relu(1 - self._loss_margin - diff_dists[_i, _j])
                        # loss = torch.concat((
                        #     relu(sim_dists - self._loss_margin),
                        #     relu(1 - self._loss_margin - diff_dists[_i, _j])), dim=1)
                    else:
                        loss = relu(
                            self._loss_margin + sim_dists - diff_dists[_i, _j])
                else:
                    ## Totally m x (k - 1) x (m - k) items
                    if is_pair:
                        loss = relu(sim_dists - self._loss_margin).unsqueeze(2) + \
                            relu(1 - self._loss_margin -
                                 diff_dists).unsqueeze(1)
                    else:
                        loss = relu(self._loss_margin +
                                    sim_dists.unsqueeze(2) - diff_dists.unsqueeze(1))
                rloss_mean = loss.detach().mean().cpu()
                rloss_max = loss.detach().max().cpu()
                if self._loss_gama < 1e+3:
                    loss = torch.log(
                        (torch.exp(self._loss_gama * loss)-1.).sum() + 1.) / self._loss_gama
            elif self._loss_opt == 'log':
                loss = -torch.log(1/(1+sim_dists)+1e-10) - \
                    torch.log(1-1/(1+diff_dists)+1e-10)
                rloss_mean = loss.detach().mean().cpu()
                rloss_max = loss.detach().max().cpu()
            sim_pos = -sim_dists.detach().mean().cpu()
            sim_neg = -diff_dists.detach().mean().cpu()
        else:
            raise Exception(f'Unkown train mode {self._train_mode}')

        loss = loss.mean()
        if self._gv_regular_w > 0:
            graph_vec_scale = tf.mean(graph_vectors**2)
            grw = self._gv_regular_w
            loss += (grw * 0.5 * graph_vec_scale)
        else:
            # with torch.no_grad():
            graph_vec_scale = tf.mean(graph_vectors.detach()**2)

        loss.backward()

        loss = loss.detach().cpu()
        graph_vec_scale = graph_vec_scale.detach().cpu()

        return {
            'loss': loss,
            'rloss_mean': rloss_mean,
            'rloss_max': rloss_max,
            'graph_vec_scale': graph_vec_scale,
            'sim_pos': sim_pos,
            'sim_neg': sim_neg,
            'sim_diff': sim_pos - sim_neg
        }


class GNNModel:

    def __init__(self, config):
        """
        GNNModel initialization

        Args:
            config: global configuration
        """
        self._config = config
        self._model_name = self._get_model_name()

        checkpoint_dir = self._config['checkpoint_dir']
        self._checkpoint_path = os.path.join(checkpoint_dir, self._model_name)
        if not os.path.exists(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)

        # Set random seeds
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed + 1)

        self._device = torch.device(config['device'])

        self._inited = False
        return

    def get_model_name(self):
        return self._model_name

    def _get_model_name(self):
        """Return the name of the model based to the configuration."""
        training_mode = self._config['training']['mode']
        feature_name = self._config['testing']['features_testing_path']
        feature_name = basename(dirname(dirname(feature_name)))
        model_name = "graph-{}-{}-{}".format("ggnn",
                                             training_mode, feature_name)
        return model_name

    def _get_debug_str(self, mean_accumulated_metrics):
        """Return a string with the mean of the input values"""
        info_str = ', '.join([' %s %.6f' % (k, v)
                              for k, v in mean_accumulated_metrics.items()])
        return info_str

    def _create_encoder(self):
        config = self._config
        encoder_map = {
            'mlp': MLPEncoder,
            'hbmp': HBMPEncoder,
            'gru': GruEncoder,
            'embed': EmbeddingEncoder,
        }
        name = config['encoder']['name']
        aggr = encoder_map[name](**config['encoder'][name])
        return aggr

    def _create_aggr(self, name=None):
        config = self._config
        aggr_map = {
            'gated': GatedAggr,
            'set2set': Set2Set,
            'settrans': SetTransformerAggr,
            'softmax': SoftmaxAggr,
            'msoftv2': AdapMultiSoftmaxAggrV2,
        }
        aggr_net = config['aggr']['name'] if name is None else name
        aggr = aggr_map[aggr_net](**config['aggr'][aggr_net])
        return aggr

    def _model_initialize(self):
        # Create the TF NN
        self._model = GGNN(
            self._create_encoder(),
            self._create_aggr(),
            **self._config['ggnn_net'],
        ).to(self._device)
        self._inited = True
        return

    # optimizer=opt, net=net, iterator=iterator
    def _save_checkpoint(self, epoch):
        """save checkpoints"""
        path = os.path.join(self._checkpoint_path, f"checkpoint_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
        }, path)
        return path

    def _latest_checkpoint(self):
        cp_prefix = "checkpoint_"
        cp_postfix = ".pt"
        all_cps = sorted([
            (cp, int(cp[len(cp_prefix):cp.find(cp_postfix)]))
            for cp in os.listdir(self._checkpoint_path)
            if cp.endswith(cp_postfix) and cp.startswith(cp_prefix)
        ], key=lambda x: x[1], reverse=True)
        if len(all_cps) == 0:
            return None
        return os.path.join(self._checkpoint_path, all_cps[0][0])

    def _restore_model(self, ckpt=None):
        checkpoint = torch.load(ckpt)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        if getattr(self, '_optimizer', None) is not None:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return {'epoch': epoch}

    def restore_model(self, ckpt=None):
        """Restore the model from the latest checkpoint"""
        if ckpt is not None:
            ckpt_to_restore = ckpt
        elif 'checkpoint_name' in self._config and self._config['checkpoint_name'] is not None:
            ckpt_to_restore = os.path.join(
                self._checkpoint_path, self._config['checkpoint_name'])
        else:
            ckpt_to_restore = self._latest_checkpoint()
        if ckpt_to_restore:
            log.info("Loading trained model from: {}".format(ckpt_to_restore))
            state = self._restore_model(ckpt_to_restore)
        else:
            log.info("No checkpoint to restore, initializing from scratch.")
            state = {}
        return state

    def restore_model_at_epoch(self, epoch: int):
        """Restore the model from the latest checkpoint"""
        ckpt_to_restore = os.path.join(
            self._checkpoint_path, f"checkpoint_{epoch}.pt")
        log.info(f"Loading trained model from: {ckpt_to_restore}")
        state = self._restore_model(ckpt_to_restore)
        return state

    def _embed_one_batch(self, *args):
        args = batch_to(args, self._device)
        self._model.eval()
        with torch.no_grad():
            eval_pairs = self._model(*args)
        return eval_pairs.detach()

    def _infer_one_batch(self, *args):
        loss = self._config['training']['loss']
        eval_pairs = self._embed_one_batch(*args)
        x, y = reshape_and_split_tensor(eval_pairs, 2)
        similarity = compute_similarity(loss, x, y)
        return similarity.cpu()

    def _run_evaluation(self, batch_generator):
        """
        Common operations for the dataset evaluation.
        Used with validation and testing.

        Args:
            batch_generator: provides batches of input data.
        """
        embeds_list = list()

        for batch_inputs in batch_generator.pairs():
            embeds = self._embed_one_batch(*batch_inputs)
            embeds_list.append(embeds)

        embeds = torch.cat(embeds_list)
        gids = batch_generator.get_group_ids()
        assert len(embeds) == len(gids)

        val_size = len(embeds)

        mrr1 = 0.
        recalls = [0, 0, 0, 0]
        recalls_rankf_raw = [0.01, 0.05, 0.10]
        recalls_rank = [1] + [int(r*val_size) for r in recalls_rankf_raw]
        pair_sizes = []
        skip_nexts, nfunc = 0, 0
        for anchor, gid in zip(embeds, gids):
            if skip_nexts > 0:
                skip_nexts -= 1
                continue
            dists = - \
                compute_similarity(
                    self._config['training']['loss'], embeds, anchor)
            ranks = rankdata(dists.cpu(), method='ordinal', nan_policy='raise')
            group = gids == gid
            group_size = sum(group)
            worest_rank = max(ranks[group]) - group_size + 1
            g_ranks = np.sort(ranks[group]) - np.array(list(range(sum(group))))
            g_ranks[g_ranks > recalls_rank[1]] = 1e+7
            mrr1 += (1 / g_ranks).mean()
            for j, it_rank in enumerate(recalls_rank):
                if it_rank >= worest_rank:
                    recalls[j] += 1
            skip_nexts = group_size
            pair_sizes.append(group_size)
            nfunc += 1
        mrr1 /= nfunc
        for i, recall in enumerate(recalls):
            recalls[i] = recall / nfunc

        log.info("AVG pair size: %.2f" % (sum(pair_sizes)/len(pair_sizes)))
        log.info("Pool size: %.2f. No. func: %d. " % (val_size, nfunc))
        log.info("mrr.01: %.4f" % mrr1)

        # Print the Recalls
        log.info("\t\tRecall@(%d): %.4f", 1, recalls[0])
        for limit, item in zip(recalls_rankf_raw, recalls[1:]):
            log.info("\t\tRecall@(%.2f): %.4f", limit, item)

        return mrr1, recalls

    def trace_model(self, batch_input):
        _ = self._model(*batch_input)
        self._model = torch.jit.script(self._model, example_inputs={
                                       self._model: list(batch_input)})
        # self._model = torch.jit.trace(self._model, batch_input)

    # XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib
    def model_train(self, restore=None):
        """Run model training"""

        # Create a training and validation dataset
        training_set, validation_set = \
            build_train_validation_generators(self._config)

        # Model initialization
        self._model_initialize()
        batch_inputs, label = next(iter(training_set))
        self.trace_model(batch_to(batch_inputs, self._device))
        trainHelper = TrainHelper(self._model, self._config)

        for name, parameters in self._model.named_parameters():
            print(name, ':', parameters.size())

        lr = self._config['training']['learning_rate']
        wd = self._config['training']['weight_decay']
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=lr, weight_decay=wd)

        # Model restoring
        epoch_counter = 0
        if restore:
            state = self.restore_model()
            epoch_counter = state['epoch'] + \
                1 if 'epoch' in state else epoch_counter

        # Logging
        print_after = self._config['training']['print_after']
        evaluate_after = self._config['training']['evaluate_after']
        epoh_tolerate = self._config['training']['epoh_tolerate']
        clean_after = self._config['training']['clean_cache_after']

        clip_value = self._config['training']['clip_value']

        log.info("Starting model training!")

        t_start = time.time()

        best_mrr = 0
        best_mrr_ckpt = None
        cnt_from_last_best = 0
        accumulated_metrics = collections.defaultdict(list)

        # Iterates over the training data.
        it_num = 0

        # Init Data Loader
        # def worker_init(worker_id):
        #     training_set.reset_seed(worker_id)
        training_batch_generator = data.DataLoader(
            training_set, batch_size=None, persistent_workers=True,
            num_workers=1, prefetch_factor=3, pin_memory=True)
            # worker_init_fn=worker_init)

        # Let's check the starting values
        self._run_evaluation(validation_set)
        self._save_checkpoint(-1)

        batch_inputs = None
        max_node_size, max_edge_size = 0, 0
        total_num_epochs = self._config['training']['num_epochs']
        while epoch_counter < self._config['training']['num_epochs']:
            log.info("Epoch %d", epoch_counter)

            self._model.train()
            training_set.step()

            for next_batch_inputs, labels in training_batch_generator:
                # Training step
                self._optimizer.zero_grad()
                if _it_check_condition(it_num, clean_after):
                    torch.cuda.empty_cache()

                # Overlay the data transfer with computation
                next_batch_inputs = batch_to(
                    next_batch_inputs, self._device, True)
                if batch_inputs is None:
                    batch_inputs = next_batch_inputs
                    continue

                try:
                    train_metrics = trainHelper(labels, *batch_inputs)
                except RuntimeError as e:
                    log.warning(e)
                    torch.cuda.empty_cache()
                    train_metrics = trainHelper(labels, *batch_inputs)

                batch_inputs = next_batch_inputs

                params = list(self._model.parameters())
                if clip_value > 0:
                    grad_scale = torch.nn.utils.clip_grad_norm_(
                        params, max_norm=clip_value).detach().cpu()
                else:
                    grad_scale = 0
                self._optimizer.step()
                param_scale = torch.sqrt(sum(
                    [torch.norm(p.detach().double()).cpu()**2 for p in params]))

                train_metrics['grad_scale'] = grad_scale
                train_metrics['param_scale'] = param_scale

                # Accumulate over minibatches to reduce variance
                for k, v in train_metrics.items():
                    accumulated_metrics[k].append(v)
                accumulated_metrics["npairs"].append(labels.shape[0])

                # Logging
                if _it_check_condition(it_num, print_after):

                    # Print the AVG for each metric
                    metrics_to_print = {k: np.mean(v)
                                        for k, v in accumulated_metrics.items()}
                    info_str = self._get_debug_str(metrics_to_print)
                    elapsed_time = time.time() - t_start
                    log.info('Iter %d, %s, time %.2fs' %
                             (it_num + 1, info_str, elapsed_time))

                    # Reset
                    accumulated_metrics = collections.defaultdict(list)

                it_num += 1

                if batch_inputs[0].shape[0] > max_node_size:
                    max_node_size = batch_inputs[0].shape[0]
                    log.debug(f"max batch node size: {max_node_size}")
                if batch_inputs[1].shape[1] > max_edge_size:
                    max_edge_size = batch_inputs[1].shape[1]
                    log.debug(f"max batch edge size: {max_edge_size}")

            elapsed_time = time.time() - t_start
            log.info("End of Epoch %d (elapsed_time %.2fs)",
                     epoch_counter, elapsed_time)

            # Run the evaluation every `evaluate_after` epoch:
            if _it_check_condition(epoch_counter, evaluate_after):
                log.info("Validation set")
                mrr, type_aucs = self._run_evaluation(validation_set)
                if mrr >= best_mrr:
                    cnt_from_last_best = 0
                    best_mrr = mrr
                    log.warning("best_mrr: %.4f", best_mrr)
                    # Save the model
                    best_mrr_ckpt = self._save_checkpoint(epoch_counter)
                    log.info("Model saved: {}".format(best_mrr_ckpt))
                    saved_ckpt = best_mrr_ckpt
                else:
                    cnt_from_last_best += evaluate_after
                    if _it_check_condition(epoch_counter, 10) or \
                            epoch_counter == total_num_epochs - 1:
                        saved_ckpt = self._save_checkpoint(epoch_counter)
                        log.info("Model saved: {}".format(saved_ckpt))

            if cnt_from_last_best > epoh_tolerate:
                break
            epoch_counter += 1

        return best_mrr, best_mrr_ckpt, saved_ckpt

    def model_validate(self):
        """Run model validation"""

        # Create a validation dataset
        _, validation_set = build_train_validation_generators(self._config)

        # Model initialization
        if not self._inited:
            self._model_initialize()
            self.restore_model()

        # Evaluate the validation set
        self._run_evaluation(validation_set)

        return

    def model_test(self, subname="test_out"):
        """Testing the GNN model on a single CSV with function pairs"""

        if subname is not None:
            out_dir = join(self._config['outputdir'], subname)
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = self._config['outputdir']

        infer_tasks = self._config['testing'].get('infer_tasks', [])
        for infer_inp in infer_tasks:
            output_name = None
            if isinstance(infer_inp, tuple):
                output_name = basename(infer_inp[1])
                infer_inp = infer_inp[0]
            else:
                output_name = basename(infer_inp)[:-4] + ".pkl"

            batch_generator = build_testing_generator(
                self._config, infer_inp)

            # Model initialization
            if not self._inited:
                self._model_initialize()
                self.restore_model()
                print(
                    f"Tot. Param. {sum(p.numel() for p in self._model.parameters())}")

            batch_generator = data.DataLoader(
                batch_generator, batch_size=None, persistent_workers=False,
                num_workers=1, prefetch_factor=3, pin_memory=True)

            embeddings = []
            for batch_inputs in batch_generator:
                batch_inputs = batch_to(batch_inputs, self._device, True)
                batch_embeds = self._embed_one_batch(*batch_inputs).cpu()
                embeddings.append(batch_embeds)
            embeddings = torch.concat(embeddings, dim=0)

            infer_out = join(out_dir, output_name)
            with open(infer_out, 'wb') as f:
                pickle.dump(embeddings, f)
            log.info("Result embeddings saved to {}".format(infer_out))

        return
