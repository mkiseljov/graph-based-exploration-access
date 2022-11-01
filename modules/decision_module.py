import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

from modules.olr import OnlineLinearRegression


class TopKWrapper:
    def __init__(self, indices, values, probabilities):
        self.indices = indices
        self.values = values
        self.probabilities = probabilities


class DecisionMaker(nn.Module):
    def __init__(self, k, min_idx, num_items, input_size, hidden_size, dropout_prob=0.1, use_filter=True):
        super(DecisionMaker, self).__init__()
        self.k = k
        self.min_idx = min_idx
        self.num_items = num_items
        self.item_idx = np.arange(self.min_idx, num_items + self.min_idx)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        self.node_embeddings_encoder = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        ).to(self.device)
        self.graph = None
        self.use_filter = use_filter

    def encode_nodes(self, embedding, node_idx, timestamps=None, edge_idx=None):
        return self.node_embeddings_encoder(embedding), torch.zeros(embedding.shape[0])

    def affinity_score(self, user_embeddings, item_embeddings, user_idx=None, item_idx=None, timestamps=None, edge_idx=None):
        user, ul = self.encode_nodes(user_embeddings, user_idx, timestamps, edge_idx)
        item, il = self.encode_nodes(item_embeddings, item_idx, timestamps, edge_idx)
        return (user * item).sum(dim=1).sigmoid(), ul.mean() / 2 + il.mean() / 2

    def calc_scores(self, user_embedding, item_embedding, user_idx=None, item_idx=None, timestamps=None, edge_idx=None):
        return torch.mm(user_embedding, item_embedding.T), torch.FloatTensor([0]).mean().to(user_embedding.device)

    def filter_history(self, scores, user_idx, timestamps):
        for i in range(scores.shape[0]):
            for j in self.graph.neg_node_to_neighbors[user_idx[i]]:
                if j > 0:
                    scores[i][j - self.min_idx] = 0
            for j in self.graph.node_to_neighbors[user_idx[i]]:
                if j > 0:
                    scores[i][j - self.min_idx] = 0
        return scores

    def decide(self, scores):
        topk = torch.topk(scores, self.k, dim=1)
        return TopKWrapper(topk.indices + self.min_idx, topk.values, scores.sigmoid())

    def forward(self, user_embeddings, item_embeddings, user_idx=None, timestamps=None, edge_idx=None):
        ### TODO introduce previous historical filter here
        item_embeddings = item_embeddings[self.min_idx:]
        user_embeddings, user_loss = self.encode_nodes(user_embeddings, user_idx, timestamps, edge_idx)
        item_embeddings, item_loss = self.encode_nodes(item_embeddings, self.item_idx, timestamps, edge_idx)
        scores, add_loss = self.calc_scores(user_embeddings, item_embeddings, user_idx, self.item_idx, timestamps, edge_idx)
        if self.use_filter:
            scores = self.filter_history(scores, user_idx, timestamps)
        return self.decide(scores), user_loss, item_loss, add_loss


class DotDecisionMaker(DecisionMaker):
    def __init__(self, k, min_idx, num_items, input_size, hidden_size, dropout_prob, use_filter):
        super(DotDecisionMaker, self).__init__(k, min_idx, num_items, input_size, hidden_size, dropout_prob, use_filter)


class GreedyDecisionMaker(DecisionMaker):
    def __init__(self, k, min_idx, num_items, input_size, hidden_size, dropout_prob, use_filter, epsilon=0.1):
        super(GreedyDecisionMaker, self).__init__(k, min_idx, num_items, input_size, hidden_size, dropout_prob, use_filter)
        self.epsilon = epsilon

    def decide(self, scores):
        if np.random.random() < 0.1:
            col_idxs = torch.randint(scores.shape[1], size=(scores.shape[0] * self.k,),
                                     device=scores.device)
            row_idxs = torch.repeat_interleave(torch.arange(scores.shape[0], device=scores.device), self.k)
            idxs = col_idxs.reshape(-1, self.k) + self.min_idx
            return TopKWrapper(idxs, scores[row_idxs, col_idxs].reshape(-1, self.k), scores.sigmoid())
        else:
            return super().decide(scores)


class VAEDecisionMaker(DecisionMaker):
    def __init__(self, k, min_idx, num_items, input_size, hidden_size, dropout_prob, use_filter, out_size):
        super(VAEDecisionMaker, self).__init__(k, min_idx, num_items, input_size, hidden_size, dropout_prob, use_filter)

        self.encoder = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, out_size * 2),
        ).to(self.device)

        self.decoder = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(out_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, input_size),
            # nn.LayerNorm(hidden_size),
        ).to(self.device)
        self.out_size = out_size
        self.mse = nn.MSELoss(reduction="none")

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        return mu + std * eps

    def encode_nodes(self, x, node_idx, timestamps=None, edge_idx=None):
        h = self.encoder(x).reshape(-1, self.out_size, 2)
        mu = h[:, :, 0]
        log_var = h[:, :, 1]
        h = self.reparametrize(mu, log_var)
        decoded = self.decoder(h)
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        l = kl + ((decoded - x) ** 2).mean(dim=1)
        return h, l


class RwsCountDecisionMaker(DecisionMaker):
    def __init__(self, k, min_idx, num_items, input_size, hidden_size, dropout_prob, use_filter, graph, rws_len=3, rws_num=100, weight=0.1):
        super(RwsCountDecisionMaker, self).__init__(k, min_idx, num_items, input_size, hidden_size, dropout_prob, use_filter)
        self.graph = graph
        self.rws_len = rws_len
        self.rws_num = rws_num
        self.weight = weight

    def calc_scores(self, user_embedding, item_embedding, user_idx=None, item_idx=None, timestamps=None, edge_idx=None):
        scores, _ = super().calc_scores(user_embedding, item_embedding, user_idx, item_idx, timestamps, edge_idx)
        for idx, user_id in enumerate(user_idx):
            walks_n_idx, _, _ = self.graph.get_random_walks(user_id, timestamps[idx], self.rws_num, self.rws_len)
            walks_n_idx = walks_n_idx.flatten()
            walks_n_idx = walks_n_idx[walks_n_idx >= self.min_idx]
            indices, counts = np.unique(walks_n_idx, return_counts=True)
            indices -= self.min_idx
            add = np.zeros(item_embedding.shape[0])
            add[indices] = counts
            # TODO decaying weight
            # TODO add some function
            scores[idx] += self.weight / torch.FloatTensor(add + 1).to(user_embedding.device)
        return scores, torch.FloatTensor([0]).mean().to(user_embedding.device)


class RNDDecisionMaker(DecisionMaker):
    def __init__(self, k, min_idx, num_items, input_size, hidden_size, dropout_prob, use_filter, weight):
        super(RNDDecisionMaker, self).__init__(k, min_idx, num_items, input_size, hidden_size, dropout_prob, use_filter)
        self.weight = weight
        self.random_net = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        for param in self.random_net.parameters():
            param.requires_grad = False

        self.learner = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def affinity_score(self, user_embeddings, item_embeddings, user_idx=None, item_idx=None, timestamps=None, edge_idx=None):
        user, ul = self.encode_nodes(user_embeddings, user_idx, timestamps, edge_idx)
        item, il = self.encode_nodes(item_embeddings, item_idx, timestamps, edge_idx)
        x = torch.cat([user, item], dim=1)
        rnd = self.random_net(x)
        pred = self.learner(x)
        mse = ((pred - rnd).flatten() ** 2) / 4
        # aff, _ = super().affinity_score(user_embeddings, item_embeddings, user_idx, item_idx, timestamps, edge_idx)
        return ((user * item).sum(dim=1) + self.weight * mse).sigmoid(), mse.mean()

    def calc_scores(self, user_embedding, item_embedding, user_idx=None, item_idx=None, timestamps=None, edge_idx=None):
        # TODO FIX DIMENSIONAL ISSUE
        # HERE ENCODINGS ARE AFTER NODE ENCODINGS, BUT IN AFFINITY WE YSE RAW ONES
        scores, _ = super().calc_scores(user_embedding, item_embedding, user_idx, item_idx, timestamps, edge_idx)
        # user_reps = torch.tile(user_embedding, (1, item_embedding.shape[0])).reshape(-1, user_embedding.shape[1])
        user_reps = torch.repeat_interleave(user_embedding, item_embedding.shape[0], dim=0)
        # item_reps = torch.tile(item_embedding, (user_embedding.shape[0], 1))
        item_reps = item_embedding.view(-1, 1).repeat(user_embedding.shape[0], 1).view(-1, user_embedding.shape[1])
        x = torch.cat([user_reps, item_reps], dim=1)
        rnd = self.random_net(x)
        pred = self.learner(x)
        mse = ((pred - rnd) ** 2) / 4
        mse = mse.reshape(user_embedding.shape[0], -1)
        return scores + self.weight * mse, mse.mean()


def get_decision_maker(args, min_idx, num_items, graph=None):
    if args.decison_maker == "eps": #ok
        return GreedyDecisionMaker(args.topk, min_idx, num_items, args.memory_dim, args.memory_dim // 2, args.drop_out, args.use_filter, args.epsilon)
    elif args.decison_maker == "vae": #ok
        return VAEDecisionMaker(args.topk, min_idx, num_items, args.memory_dim, args.memory_dim // 2, args.drop_out, args.use_filter, args.memory_dim // 2)
    elif args.decison_maker == "ssl-rws-count":
        return RwsCountDecisionMaker(args.topk, min_idx, num_items, args.memory_dim, args.memory_dim // 2, args.drop_out, args.use_filter, graph, weight=args.rws_weight)
    elif args.decison_maker == "ssl-rnd":
        return RNDDecisionMaker(args.topk, min_idx, num_items, args.memory_dim, args.memory_dim // 2, args.drop_out, args.use_filter, weight=args.rws_weight)
    return DotDecisionMaker(args.topk, min_idx, num_items, args.memory_dim, args.memory_dim // 2, args.drop_out, args.use_filter)
