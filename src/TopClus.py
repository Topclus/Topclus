
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from nltk.corpus import stopwords
import pickle
import matplotlib as mpl


class AutoEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dims):
        super(AutoEncoder, self).__init__()

        self.encoder_layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            else:
                layer = nn.Linear(dims[i], dims[i+1])
            self.encoder_layers.append(layer)
        self.encoder = nn.Sequential(*self.encoder_layers)

        self.decoder_layers = []
        hidden_dims.reverse()
        dims = hidden_dims + [input_dim]
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            else:
                layer = nn.Linear(dims[i], dims[i+1])
            self.decoder_layers.append(layer)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def forward(self, x):

        z = self.encoder(x)
        x_bar = self.decoder(z)
        x_bar = F.normalize(x_bar, dim=-1)

        return x_bar, z


class TopicCluster(nn.Module):

    def __init__(self, args):
        super(TopicCluster, self).__init__()
        self.alpha = 1.0
        self.dataset = args.dataset
        self.args = args
        self.device = args.device
        self.temperature = args.temperature
        self.distribution = args.distribution

        input_dim = args.input_dim
        hidden_dims = eval(args.hidden_dims)
        self.model = AutoEncoder(input_dim, hidden_dims)
        self.topic_emb = Parameter(torch.Tensor(args.n_clusters, hidden_dims[-1]))
        torch.nn.init.xavier_normal_(self.topic_emb.data)

    def pretrain(self, input_data, pretrain_epoch=200):
        pretrained_path = os.path.join(self.dataset, "pretrained.pt")
        if os.path.exists(pretrained_path):
            # load pretrain weights
            print(f"loading pretrained model from {pretrained_path}")
            self.model.load_state_dict(torch.load(pretrained_path))
        else:
            train_loader = DataLoader(input_data, batch_size=self.args.batch_size, shuffle=True)
            optimizer = Adam(self.model.parameters(), lr=self.args.lr)
            for epoch in range(pretrain_epoch):
                total_loss = 0
                for batch_idx, (x, _) in enumerate(train_loader):
                    x = x.to(self.device)
                    optimizer.zero_grad()
                    x_bar, z = self.model(x)
                    loss = F.mse_loss(x_bar, x)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                print(f"epoch {epoch}: loss = {total_loss / (batch_idx+1):.4f}")
            torch.save(self.model.state_dict(), pretrained_path)
            print(f"model saved to {pretrained_path}")

    def cluster_assign(self, z):
        if self.distribution == 'student':
            q = 1.0 / (1.0 + torch.sum(
                torch.pow(z.unsqueeze(1) - self.topic_emb, 2), 2) / self.alpha)
            q = q.pow((self.alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
        else:
            self.topic_emb.data = F.normalize(self.topic_emb.data, dim=-1)
            z = F.normalize(z, dim=-1)
            sim = torch.matmul(z, self.topic_emb.t()) / self.temperature
            q = F.softmax(sim, dim=-1)
        return q
    
    def forward(self, x):
        x_bar, z = self.model(x)
        q = self.cluster_assign(z)
        return x_bar, z, q

    def target_distribution(self, x, method='all', top_num=0):
        _, z = self.model(x)
        q = self.cluster_assign(z).detach()
        if method == 'all':
            p = q**2 / q.sum(0)
            p = (p.t() / p.sum(1)).t()
        elif method == 'top':
            assert top_num > 0
            p = q.clone()
            sim = torch.matmul(self.topic_emb, z.t())
            _, selected_idx = sim.topk(k=top_num, dim=-1)
            for i, topic_idx in enumerate(selected_idx):
                p[topic_idx] = 0
                p[topic_idx, i] = 1
        return q, p


def train(args, emb_dict, seed=None, plot=False):

    inv_vocab = emb_dict["inv_vocab"]
    vocab = emb_dict["vocab"]
    embs = F.normalize(torch.tensor(emb_dict["word_emb"]), dim=-1)
    input_data = TensorDataset(embs, torch.arange(embs.size(0)))
    topic_cluster = TopicCluster(args).to(args.device)
    topic_cluster.pretrain(input_data, args.pretrain_epoch)
    train_loader = DataLoader(input_data, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(topic_cluster.parameters(), lr=args.lr)

    # topic embedding initialization
    embs = embs.to(args.device)
    x_bar, z = topic_cluster.model(embs)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())

    z = None
    x_bar = None

    y_pred_last = y_pred
    topic_cluster.topic_emb.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    topic_cluster.train()
    i = 0
    for epoch in range(20):
        if epoch % 5 == 0:
            _, z, q = topic_cluster(embs)
            if not os.path.exists(os.path.join(args.dataset, "clusters")):
                os.makedirs(os.path.join(args.dataset, "clusters"))
            f = open(os.path.join(args.dataset, f"clusters/{epoch}.txt"), 'w')
            pred_cluster = q.argmax(-1)
            if seed is not None:
                seed_idx = vocab[seed]
                topic_sim = torch.matmul(z[seed_idx], topic_cluster.topic_emb.t())
                _, topic_idx = topic_sim.topk(k=1)
            else:
                topic_idx = []
            for j in range(args.n_clusters):
                if args.sort_method == 'discriminative':
                    word_idx = torch.arange(embs.size(0))[pred_cluster == j]
                    sorted_idx = torch.argsort(q[pred_cluster == j][:, j], descending=True)
                    word_idx = word_idx[sorted_idx]
                else:
                    sim = torch.matmul(topic_cluster.topic_emb[j], z.t())
                    _, word_idx = sim.topk(k=10, dim=-1)
                word_cluster = [inv_vocab[idx.item()] for idx in word_idx]
                f.write(f"Topic {j}: " + ', '.join(word_cluster)+'\n\n')
                if j in topic_idx:
                    relevant_idx = torch.arange(embs.size(0))[pred_cluster == j]
                    print(f"Relevent topic: " + ','.join(word_cluster))
                    print(f"all words: {[inv_vocab[idx.item()] for idx in relevant_idx]}")

            if plot:
                random_idx = torch.multinomial(torch.ones(z.size(0)), min(10000, z.size(0)))
                z = z[random_idx]
                q = q[random_idx]
                pred_cluster = pred_cluster[random_idx].detach().cpu().numpy()
                fig = plt.figure(figsize=(8, 6))
                centers = F.normalize(topic_cluster.topic_emb.data, dim=-1).detach().cpu().numpy()
                rand_rgb = np.random.rand(args.n_clusters, 3)
                z_plot = z.detach().cpu().numpy()
                plot_data = np.concatenate((centers, z_plot), axis=0)
                projected_emb = TSNE(n_components=2, n_iter=2000, perplexity=30, init='pca', metric='cosine', random_state=0, n_jobs=20).fit_transform(plot_data)
                centers = projected_emb[:args.n_clusters, :]
                projected_emb = projected_emb[args.n_clusters:, :]
                with open(f'{epoch}.pickle', 'wb') as handle:
                    pickle.dump({'projected_emb': projected_emb, 'pred_cluster': pred_cluster, 'centers': centers}, handle, protocol=pickle.HIGHEST_PROTOCOL)
                for j in range(projected_emb.shape[0]):
                    plt.plot(projected_emb[j, 0], projected_emb[j, 1], c=rand_rgb[pred_cluster[j]], 
                            marker='o', markersize=5, alpha=.2)
                for j in range(args.n_clusters):    
                    plt.plot(centers[j, 0], centers[j, 1], c=rand_rgb[j], 
                            marker='*', markersize=8, alpha=1)
                if not os.path.exists(os.path.join(args.dataset, "vis")):
                    os.makedirs(os.path.join(args.dataset, "vis"))
                fig.savefig(os.path.join(args.dataset, f'vis/{epoch}.png'))
            
        for x, idx in train_loader:
            
            if i % args.update_interval == 0:
                q, p = topic_cluster.target_distribution(embs, method='all', top_num=epoch+1)

                y_pred = q.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred

                print(f"Iter {i}: Delta: {delta_label}")

                if i > 0 and delta_label < args.tol:
                    print(f'delta_label {delta_label:.4f} < tol ({args.tol})')
                    print('Reached tolerance threshold. Stopping training.')
                    return

            i += 1
            x = x.to(args.device)
            idx = idx.to(args.device)

            x_bar, _, q = topic_cluster(x)

            reconstr_loss = F.mse_loss(x_bar, x)
            kl_loss = F.kl_div(q.log(), p[idx], reduction='batchmean')
            loss = args.gamma * kl_loss + reconstr_loss
            print(f"KL loss: {kl_loss}; Reconstruction loss: {reconstr_loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return relevant_idx if seed is not None else None


def filter_vocab(emb_dict):
    stop_words = set(stopwords.words('english'))
    new_inv_vocab = [w for w, _ in emb_dict['vocab'].items() if w not in stop_words and not w.startswith('##')]
    new_vocab = {w:i for i, w in enumerate(new_inv_vocab)}
    new_word_emb = emb_dict['word_emb'][[emb_dict['vocab'][w] for w in new_inv_vocab]]
    return {"word_emb": new_word_emb, "vocab": new_vocab, "inv_vocab": new_inv_vocab}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', default=None)
    parser.add_argument('--n_clusters', default=100, type=int)
    parser.add_argument('--input_dim', default=768, type=int)
    parser.add_argument('--pretrain_epoch', default=200, type=int)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--sort_method', default='generative', choices=['generative', 'discriminative'])
    parser.add_argument('--distribution', default='softmax', choices=['softmax', 'student'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_dims', default='[500, 500, 1000, 50]', type=str)
    parser.add_argument('--dataset', type=str, default='../datasets/nyt')
    parser.add_argument('--gamma', default=0.02, type=float, help='weight of clustering loss')
    parser.add_argument('--update_interval', default=100, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")

    pretrained_lm = 'bert-base-uncased'
    model = BertModel.from_pretrained(pretrained_lm,
                                      output_attentions=False,
                                      output_hidden_states=False)
    tokenizer = BertTokenizer.from_pretrained(pretrained_lm, do_lower_case=True)
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    inv_vocab = {k:v for v, k in vocab.items()}
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    save_file = os.path.join("../datasets/nyt/", "word_emb.pt")
    assert os.path.exists(save_file):
    emb_dict = torch.load(save_file)
    emb_dict = filter_vocab(emb_dict)
    
    print(args)
    
    train(args, emb_dict, seed=args.seed, plot=True)
