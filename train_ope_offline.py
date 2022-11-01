import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction
from metrics import ndcg_score, ap_score, ndcg_score_replayed, entropy_score
from model.tgn import TGN
from modules.decision_module import get_decision_maker
from modules.ope_loss_module import get_ope_loss_function
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics, Data

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--rws_weight', type=float, default=0.1, help='Weights of exploration in rw strategy')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--tune_backbone', action='store_true',
                    help='Whether to use finetune backbone model in online fashion')
parser.add_argument('--slated', action='store_true',
                    help='Whether data is slated')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                   'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--topk', type=int, default=5, help='Top k items to show')
parser.add_argument('-m', '--decison_maker', type=str, default="eps", help='Type of decision maker')
parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon for epsilon-greedy strategy')

parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--num_clusters', type=int, default=128, help='Number of clusters for diff-group like arch')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--pretrain', action='store_true',
                    help='Whether to pretrain model or train in online fashion')
parser.add_argument("--best_checkpoint_path", type=str, help="Path to the pretrained model")
parser.add_argument('--pretrain_predictor', action='store_true',
                    help='Whether to pretrain model or train in online fashion')

parser.add_argument('--freeze_all', action='store_true',
                    help='Whether to pretrain model or train in online fashion')

parser.add_argument('--use_neg', action='store_true',
                    help='Whether to use negative history to embed')
parser.add_argument('--use_filter', action='store_true',
                    help='Whether to filter user history')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes,
                              randomize_features=args.randomize_features, pretrain=args.pretrain)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

MIN_ITEM_IDX = full_data.destinations[full_data.destinations > 0].min()

USER_LOSS_WGT = 0.1
ITEM_LOSS_WGT = 0.1
ADD_LOSS_WGT = 0.1

pre_test_data = Data(
    np.vstack([train_data.sources.reshape(-1, 1), val_data.sources.reshape(-1, 1)]).flatten(),
    np.vstack([train_data.destinations.reshape(train_data.sources.shape[0], -1),
               val_data.destinations.reshape(val_data.sources.shape[0], -1)]),
    np.vstack([train_data.timestamps.reshape(-1, 1), val_data.timestamps.reshape(-1, 1)]).flatten(),
    np.vstack([train_data.edge_idxs.reshape(train_data.sources.shape[0], -1),
               val_data.edge_idxs.reshape(val_data.sources.shape[0], -1)]),
    np.vstack([train_data.labels.reshape(train_data.sources.shape[0], -1),
               val_data.labels.reshape(val_data.sources.shape[0], -1)]),
)

order, weights = np.unique(pre_test_data.destinations.flatten(), return_counts=True)
INVERSE_WEIGHTS = np.zeros(full_data.destinations.max() + 1)
INVERSE_WEIGHTS[order] = weights
INVERSE_WEIGHTS = 1 / np.clip(INVERSE_WEIGHTS / INVERSE_WEIGHTS.sum(), 1e-2, 1 - 1e-2)

if len(train_data.destinations.shape) > 1:
    mask = (train_data.labels > 0)
    reps = mask.sum(axis=1)
    train_data = Data(
        np.repeat(train_data.sources, reps),
        train_data.destinations[mask],
        np.repeat(train_data.timestamps, reps),
        train_data.edge_idxs[mask],
        train_data.labels[mask],
    )

    mask = (val_data.labels > 0)
    reps = mask.sum(axis=1)
    val_data = Data(
        np.repeat(val_data.sources, reps),
        val_data.destinations[mask],
        np.repeat(val_data.timestamps, reps),
        val_data.edge_idxs[mask],
        val_data.labels[mask],
    )

    mask = (new_node_val_data.labels > 0)
    reps = mask.sum(axis=1)
    new_node_val_data = Data(
        np.repeat(new_node_val_data.sources, reps),
        new_node_val_data.destinations[mask],
        np.repeat(new_node_val_data.timestamps, reps),
        new_node_val_data.edge_idxs[mask],
        new_node_val_data.labels[mask],
    )

for i in range(args.n_runs):
    results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    decision_maker = get_decision_maker(args, MIN_ITEM_IDX, full_data.destinations.max() - MIN_ITEM_IDX + 1,
                                        train_ngh_finder)
    tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
              edge_features=edge_features, device=device,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
              memory_update_at_start=not args.memory_update_at_end,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              memory_updater_type=args.memory_updater,
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message,
              dyrep=args.dyrep, min_item_idx=MIN_ITEM_IDX, decision_maker=decision_maker, use_neg=args.use_neg)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
    if not args.pretrain_predictor:
        for param in tgn.decision_maker.parameters():
            param.requires_grad = False
    tgn = tgn.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    new_nodes_val_aps = []
    val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []
    best_model_path = None

    early_stopper = EarlyStopMonitor(max_round=args.patience)
    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()
        ### Training

        # Reinitialize memory of the model at the start of each epoch
        if USE_MEMORY:
            tgn.memory.__init_memory__()

        # Train using only training graph
        tgn.set_neighbor_finder(train_ngh_finder)
        m_loss = []

        logger.info('start {} epoch'.format(epoch))
        for k in range(0, num_batch, args.backprop_every):
            loss = 0
            optimizer.zero_grad()

            # Custom loop to allow to perform backpropagation only every a certain number of batches
            for j in range(args.backprop_every):
                batch_idx = k + j

                if batch_idx >= num_batch:
                    continue

                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)
                sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                    train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]

                size = len(sources_batch)
                _, negatives_batch = train_rand_sampler.sample(size)

                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device=device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=device)

                tgn = tgn.train()
                pos_prob, neg_prob, add_loss = tgn.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                              negatives_batch,
                                                                              timestamps_batch, edge_idxs_batch,
                                                                              NUM_NEIGHBORS)

                loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
                loss += ADD_LOSS_WGT * add_loss

            loss /= args.backprop_every

            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())

            # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
            # the start of time
            if USE_MEMORY:
                tgn.memory.detach_memory()

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        ### Validation
        # Validation uses the full graph
        tgn.set_neighbor_finder(full_ngh_finder)

        if USE_MEMORY:
            # Backup memory at the end of training, so later we can restore it and use it for the
            # validation on unseen nodes
            train_memory_backup = tgn.memory.backup_memory()

        val_ap, val_auc = eval_edge_prediction(model=tgn,
                                               negative_edge_sampler=val_rand_sampler,
                                               data=val_data,
                                               n_neighbors=NUM_NEIGHBORS)
        if USE_MEMORY:
            val_memory_backup = tgn.memory.backup_memory()
            # Restore memory we had at the end of training to be used when validating on new nodes.
            # Also backup memory after validation so it can be used for testing (since test edges are
            # strictly later in time than validation edges)
            tgn.memory.restore_memory(train_memory_backup)

        # Validate on unseen nodes
        nn_val_ap, nn_val_auc = eval_edge_prediction(model=tgn,
                                                     negative_edge_sampler=val_rand_sampler,
                                                     data=new_node_val_data,
                                                     n_neighbors=NUM_NEIGHBORS)

        if USE_MEMORY:
            # Restore memory we had at the end of validation
            tgn.memory.restore_memory(val_memory_backup)

        new_nodes_val_aps.append(nn_val_ap)
        val_aps.append(val_ap)
        train_losses.append(np.mean(m_loss))

        # Save temporary results to disk
        pickle.dump({
            "val_aps": val_aps,
            "new_nodes_val_aps": new_nodes_val_aps,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info(
            'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
        logger.info(
            'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

        # Early stopping
        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            tgn.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            tgn.eval()
            break
        else:
            torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen
    # nodes
    if USE_MEMORY:
        val_memory_backup = tgn.memory.backup_memory()

    if best_model_path is not None:
        tgn.load_state_dict(torch.load(best_model_path))
    elif args.best_checkpoint_path is not None:
        tgn.load_state_dict(torch.load(args.best_checkpoint_path))

    ### Test
    test_ngh_finder = get_neighbor_finder(pre_test_data, args.uniform)
    test_ngh_finder.is_online = True
    tgn.set_neighbor_finder(test_ngh_finder)
    tgn = tgn.train()
    if not args.tune_backbone:
        for param in tgn.parameters():
            param.requires_grad = False
    if not args.freeze_all:
        for param in tgn.decision_maker.parameters():
            param.requires_grad = True
    else:
        for param in tgn.decision_maker.parameters():
            param.requires_grad = False
    decision_maker.graph = test_ngh_finder

    num_instance = len(test_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    replay_all = 0
    replay_positive = 0
    ndcg_nom = 0
    ndcg_rep_nom = 0
    ap_nom = 0
    replay_ctrs = []
    replay_ndcgs = []
    replay_maps = []
    replay_ndcgs_rep = []
    batch_mean_entropy = []
    index_align = []
    criterion = get_ope_loss_function(args)

    for k in range(0, num_batch, args.backprop_every):
        optimizer.zero_grad()
        # Custom loop to allow to perform backpropagation only every a certain number of batches
        j = 0
        batch_idx = k + j

        if batch_idx >= num_batch:
            continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch = test_data.sources[start_idx:end_idx]
        destinations_batch = test_data.destinations[start_idx:end_idx]
        destinations_targets = test_data.labels[start_idx:end_idx]
        edge_idxs_batch = test_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = test_data.timestamps[start_idx:end_idx]

        source_node_embedding, source_memory, item_embedding = tgn.compute_interaction_embedding(
            sources_batch, timestamps_batch, NUM_NEIGHBORS
        )
        topk, user_losses, item_losses, add_loss = decision_maker(source_node_embedding, item_embedding, sources_batch,
                                                                  timestamps_batch, edge_idxs_batch)
        cur_inverse_weights = INVERSE_WEIGHTS[topk.indices.detach().cpu().numpy()]
        loss, probs, f = criterion(topk, destinations_batch, destinations_targets, cur_inverse_weights)
        f = f.detach().cpu().numpy()
        f = f * (destinations_batch.reshape(destinations_batch.shape[0], -1) > 0)
        mask = f
        if not args.slated:
            f = f.sum(1) > 0
        if f.sum() == 0:
            continue
        if len(f.shape) == 2:
            ff = f.sum(axis=1)
        else:
            ff = f.astype(int)
        loss += USER_LOSS_WGT * user_losses[ff > 0].mean()

        if item_losses is not None:
            loss += ITEM_LOSS_WGT * item_losses.mean()

        if add_loss is not None:
            loss += ADD_LOSS_WGT * add_loss

        replay_all += ((f > 0) * INVERSE_WEIGHTS[destinations_batch]).sum()
        replay_positive += (destinations_targets[f] * INVERSE_WEIGHTS[destinations_batch[f]]).sum()
        ndcg_nom += ndcg_score(mask, destinations_targets, INVERSE_WEIGHTS[destinations_batch])
        ap_nom += ap_score(mask, destinations_targets, INVERSE_WEIGHTS[destinations_batch])
        ndcg_rep_nom += ndcg_score_replayed(mask, destinations_targets, INVERSE_WEIGHTS[destinations_batch])

        index_align.append(np.arange(start_idx, end_idx)[ff > 0])

        replay_ctrs.append(replay_positive / (replay_all + 1))
        replay_ndcgs.append(ndcg_nom / (replay_all + 1))
        replay_ndcgs_rep.append(ndcg_rep_nom / (replay_all + 1))
        replay_maps.append(ap_nom / (replay_all + 1))

        batch_mean_entropy.append(entropy_score(topk.probabilities).item())

        loss += USER_LOSS_WGT * user_losses[ff > 0].mean()
        sources = np.repeat(sources_batch, (f.reshape(f.shape[0], -1) * destinations_targets.reshape(f.shape[0], -1)).sum(1).astype(int), axis=0)
        tses = np.repeat(timestamps_batch, (f.reshape(f.shape[0], -1) * destinations_targets.reshape(f.shape[0], -1)).sum(1).astype(int), axis=0)
        cur_f = (f > 0) & (destinations_targets > 0)
        if cur_f.sum() != 0:
            result_memory = torch.cat([source_memory[sources], item_embedding[destinations_batch[cur_f]]], dim=0)
            tgn.save_memory_update(
                sources,
                destinations_batch[cur_f],
                result_memory,
                source_node_embedding,
                item_embedding[destinations_batch[cur_f]],
                tses,
                edge_idxs_batch[cur_f]
            )

            test_ngh_finder.update_graphs(
                sources,
                destinations_batch[cur_f],
                edge_idxs_batch[cur_f],
                tses,
            )

            sources = np.repeat(sources_batch,
                                (f.reshape(f.shape[0], -1) * (1 - destinations_targets).reshape(f.shape[0], -1)).sum(
                                    1).astype(int), axis=0)
            tses = np.repeat(timestamps_batch,
                             (f.reshape(f.shape[0], -1) * (1 - destinations_targets).reshape(f.shape[0], -1)).sum(1).astype(
                                 int), axis=0)
            cur_f = (f > 0) & (destinations_targets == 0)
            if cur_f.sum() > 0:
                test_ngh_finder.update_negative_graphs(
                    sources,
                    destinations_batch[cur_f],
                    edge_idxs_batch[cur_f],
                    tses,
                )

        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
        # the start of time
        if USE_MEMORY:
            tgn.memory.detach_memory()

    # Save results for this run
    pickle.dump({
        "replay_ctrs": replay_ctrs,
        "replay_ndcgs": replay_ndcgs,
        "replay_maps": replay_maps,
        "replay_ndcgs_rep": replay_ndcgs_rep,
        "batch_mean_entropy": batch_mean_entropy,
        "index_align": index_align,
        "val_aps": val_aps,
        "new_nodes_val_aps": new_nodes_val_aps,
        "epoch_times": epoch_times,
        "train_losses": train_losses,
        "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    logger.info('Saving TGN model')
    if USE_MEMORY:
        # Restore memory at the end of validation (save a model which is ready for testing)
        tgn.memory.restore_memory(val_memory_backup)
    torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
    logger.info('TGN model saved')
