import argparse
import logging
import time
import random
import datetime
import tqdm
from os.path import exists
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.optim as optim

import pyro
import pyro.contrib.examples.polyphonic_data_loader as poly
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceEnum_ELBO, TraceTMC_ELBO, config_enumerate
from pyro.optim import ClippedAdam

import ot
import torch.nn.functional as F

import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage

from midi2audio import FluidSynth

import pretty_midi
from scipy.io import wavfile

import musics


class Emitter(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`
    """

    def __init__(self, input_dim, z_dim, emission_dim, use_cuda=False):
        super().__init__()
        self.input_dim = input_dim
        self.use_cuda = use_cuda
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        if self.input_dim == 1:
            x = self.lin_hidden_to_input(h2)
        else:
            ps = torch.sigmoid(self.lin_hidden_to_input(h2))

            #Reparameterization Trick
            if self.use_cuda: 
                eps = torch.rand(self.input_dim).cuda()
            else : eps = torch.rand(self.input_dim)
            # assert len(emission_probs_t) == 88
            appxm = torch.log(eps + 1e-20) - torch.log(1-eps + 1e-20) + torch.log(ps + 1e-20) - torch.log(1-ps + 1e-20)
            # appxm = torch.log(eps) - torch.log(1-eps) + torch.log(x) - torch.log(1-x)
            x = torch.sigmoid(appxm)
        return x

class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_dim, transition_dim):
        super().__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return loc, scale

class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """

    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        # return loc, scale which can be fed into Normal
        return loc, scale

class Encoder(nn.Module):
    def __init__(self, input_dim=88, z_dim=100,rnn_dim=60, num_layers=1, rnn_dropout_rate=0.1,num_iafs=0, iaf_dim=50, N_z0 = 10, use_cuda=False, var_clip = 0.5, rnn_check=False):
        super().__init__()
        # instantiate PyTorch modules used in the model and guide below
        self.combiner = Combiner(z_dim, rnn_dim)
        # dropout just takes effect on inner layers of rnn
        rnn_dropout_rate = 0. if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                          batch_first=True, bidirectional=False, num_layers=num_layers,
                          dropout=rnn_dropout_rate)

        # if we're using normalizing flows, instantiate those too
        # self.iafs = [affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        # self.iafs_modules = nn.ModuleList(self.iafs)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))

        # randnでやっても失敗した。やっぱりいけた。学習率が問題？
        # onesでやってみる 訓練データ1つならOK
        # self.z_0 = nn.Parameter(torch.randn(z_dim))
        # self.z_q_0 = nn.Parameter(torch.randn(z_dim))
        
        # self.z_0 = nn.Parameter(torch.randn(N_z0, z_dim))
        # self.z_q_0 = nn.Parameter(torch.randn(N_z0, z_dim))

        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.randn(1, 1, rnn_dim))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()

        self.rnn_check = rnn_check
        self.var_clip = var_clip

    def forward(self, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, annealing_factor=1.0):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)

        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
        # if any(torch.isnan(h_0_contig.reshape(-1))):
        #     for param in self.rnn.parameters():
        #         print(param)
        #     assert False
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        # if True:
        #     if any(torch.isnan(rnn_output.data.reshape(-1))):
        #         assert False

        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)

        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))
        # z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))
        # z_prev = self.z_q_0

        # if any(torch.isnan(z_prev.reshape(-1))):
        #     print("z_prev")

        z_container = []
        z_loc_container = []
        z_scale_container = []
        for t in range(1,T_max+1):
            # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
            z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
            z_scale = torch.clamp(z_scale, min = self.var_clip)
            # Reparameterization Trick
            if self.use_cuda:
                eps = torch.randn(z_loc.size()).cuda()
            else: eps = torch.randn(z_loc.size())
            z_t = z_loc + z_scale * eps

            # the latent sampled at this time step will be conditioned upon
            # in the next time step so keep track of it
            z_prev = z_t
            z_container.append(z_t)
            z_loc_container.append(z_loc)
            z_scale_container.append(z_scale)
        
        z_container = torch.stack(z_container)
        z_loc_container = torch.stack(z_loc_container)
        z_scale_container = torch.stack(z_scale_container)
        return z_container.transpose(0,1), z_loc_container.transpose(0,1), z_scale_container.transpose(0,1)

class Prior(nn.Module):
    def __init__(self, z_dim=100, transition_dim=200,  N_z0 = 10, use_cuda=False, var_clip = 0.5):
        super().__init__()
        # instantiate PyTorch modules used in the model and guide below
        self.trans = GatedTransition(z_dim, transition_dim)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))

        # self.z_0 = nn.Parameter(torch.randn(z_dim))
        # self.z_q_0 = nn.Parameter(torch.randn(z_dim))
        
        # self.z_0 = nn.Parameter(torch.randn(N_z0, z_dim))
        # self.z_q_0 = nn.Parameter(torch.randn(N_z0, z_dim))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        self.var_clip = var_clip

    def forward(self, length, N_generate):

        # this is the number of time steps we need to process in the mini-batch
        T_max = length

        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(N_generate, self.z_q_0.size(0))
        # z_prev = self.z_q_0.expand(N_generate, self.z_q_0.size(0))
        # z_prev = self.z_q_0

        # if any(torch.isnan(z_prev.reshape(-1))):
        #     print("z_prev")

        z_container = []
        z_loc_container = []
        z_scale_container = []
        for t in range(1,T_max+1):
            # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
            z_loc, z_scale = self.trans(z_prev)
            z_scale = torch.clamp(z_scale, min = self.var_clip)

            # Reparameterization Trick
            if self.use_cuda:
                eps = torch.randn(z_loc.size()).cuda()
            else: eps = torch.randn(z_loc.size())
            z_t = z_loc + z_scale * eps

            z_prev = z_t
            z_container.append(z_t)
            z_loc_container.append(z_loc)
            z_scale_container.append(z_scale)
        
        z_container = torch.stack(z_container)
        z_loc_container = torch.stack(z_loc_container)
        z_scale_container = torch.stack(z_scale_container)
        return z_container.transpose(0,1), z_loc_container.transpose(0,1), z_scale_container.transpose(0,1)

def multi_normal_prob(loc,scale,x):
    # locは平均, scaleは共分散行列, xはサンプルとする。
    pi_2_d = torch.tensor(np.sqrt((np.pi * 2) ** loc.size(-1)))
    det = torch.sum(scale, dim=2)
    mom = torch.sqrt(det)
    exp_arg = -0.5 * torch.sum((x-loc) * (x-loc) / scale, dim=2)
    return torch.exp(exp_arg) / mom

def D_KL(p_loc, q_loc, p_scale, q_scale):
    # locは平均, scaleは共分散行列とする。
    # size = [曲数、拍数、88鍵]
    # Determinant of Covariance Matrix
    det_p_scale = torch.prod(p_scale, dim=2)
    det_q_scale = torch.prod(q_scale, dim=2)
    # Dimension of Maltivariate Normal Distribution
    dim = p_loc.size(-1)
    beats = p_loc.size(1)
    songs = p_loc.size(0)
    # Trace of (\Sigma_q^{-1} \Sigma_p)
    trace = torch.sum(p_scale/q_scale, dim = 2)
    # (loc_q - loc_p)^T \Sigma_q (loc_q - loc_p)
    niji = torch.sum((q_loc-p_loc)*(q_loc-p_loc)/q_scale, dim=2)
    # KL-divergence
    # KL = 0.5 *(torch.log(det_q_scale+1e-45) - torch.log(det_p_scale+1e-45) - dim + trace + niji)
    KL = 0.5 *(torch.log(det_q_scale) - torch.log(det_p_scale) - dim + trace + niji)
    return KL.sum() / (beats * songs)

def D_Wass(p_loc, q_loc, p_scale, q_scale):
    # locは平均, scaleは共分散行列とする。
    # size = [曲数、拍数、88鍵]
    beats = p_loc.size(1)
    songs = p_loc.size(0)
    # Determinant of Covariance Matrix
    norm_2 = torch.sum((p_loc - q_loc)*(p_loc - q_loc), dim=2)
    # Dimension of Maltivariate Normal Distribution
    trace_p_scale = torch.sum(p_scale, dim=2)
    trace_q_scale = torch.sum(q_scale, dim=2)
    trace = torch.sum(torch.sqrt(p_scale*q_scale), dim=2)
    # KL-divergence
    Wass = torch.sqrt(norm_2 + trace_p_scale + trace_q_scale - 2 * trace)
    return Wass.sum() / (beats * songs)

def multi_normal_prob(loc,scale,x):
    # locは平均, scaleは共分散行列, xはサンプルとする。
    pi_2_d = torch.tensor(np.sqrt((np.pi * 2) ** loc.size(-1)))
    det = torch.sum(scale, dim=2)
    # mom = torch.sqrt(det) * pi_2_d
    mom = torch.sqrt(det)
    exp_arg = -0.5 * torch.sum((x-loc) * (x-loc) / scale, dim=2)
    return torch.exp(exp_arg) / mom

def D_JS_Monte(p_loc, q_loc, p_scale, q_scale, x_p, x_q):
    N_data_inverse = 1/(x_p.size(0) + x_q.size(0))
    length = x_p.size(1)
    eps = 1e-40
    p_prob_x_p = multi_normal_prob(p_loc,p_scale,x_p)
    q_prob_x_p = multi_normal_prob(q_loc,q_scale,x_p)
    p_prob_x_q = multi_normal_prob(p_loc,p_scale,x_q)
    q_prob_x_q = multi_normal_prob(q_loc,q_scale,x_q)
    m_prob_x_p = 0.5 * (p_prob_x_p + q_prob_x_p)
    m_prob_x_q = 0.5 * (p_prob_x_q + q_prob_x_q)
    KL_p_m = N_data_inverse * (torch.log(p_prob_x_p + eps) - torch.log(m_prob_x_p + eps))
    KL_q_m = N_data_inverse * (torch.log(q_prob_x_q + eps) - torch.log(m_prob_x_q + eps))
    return torch.sum(0.5 * (KL_p_m + KL_q_m) / length)

def calc_how_close_to_mini_batch(samples, mini_batch):
    sum = 0
    for i in range(samples.size(0)):
        sample = samples[i].expand(mini_batch.size(0), samples.size(1), samples.size(2))
        candidate = torch.sum(torch.norm(mini_batch - sample, p=2, dim = 2), dim=1)
        sum += torch.min(candidate)
    return sum / samples.size(0)

def saveReconSinGraph(train_data, recon_data, length, path, number, id):
    FS = 15
    fig = plt.figure(figsize=(5, 7))
    plt.rcParams["font.size"] = FS
    # x = np.linspace(0, 2*np.pi, length)
    x = np.linspace(0, len(recon_data), length)
    plt.plot(x, train_data, label="Training data", color="black")
    plt.plot(x, recon_data.detach().numpy(), label="Reconstructed data", color="orange")
    plt.grid()
    # plt.title("Sin Curves")
    plt.ylim(top=10, bottom=-10)
    plt.xlim(0, length)
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    # plt.xlabel("t", fontsize=15)
    # plt.ylabel("y", fontsize=FS)
    plt.legend()
    fig.savefig(os.path.join(path, "Reconstruction"+str(number)+id+".png"))

def saveGeneSinGraph(gene_data, length, path, number, id):
    FS = 15
    fig = plt.figure(figsize=(5, 8))
    # plt.figure(figsize=(5, 8))
    plt.rcParams["font.size"] = FS
    # x = np.linspace(0, 2*np.pi, length)
    x = np.linspace(0, length, length)
    for i in range(len(gene_data)):
        plt.plot(x, gene_data[i].detach().numpy(), label="Generated data" + str(i+1))
    plt.grid()
    # plt.ylim(bottom=0)
    # plt.title("Sin Curves")
    plt.ylim(top=10, bottom=-10)
    plt.xlim(0, length)
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    # plt.xlabel("t", fontsize=15)
    # plt.ylabel("y", fontsize=FS)
    plt.legend()
    fig.savefig(os.path.join(path, "Generation"+str(number)+id+".png"))

def saveTrainSinGraph(gene_data, length, path, number, id):
    FS = 10
    fig = plt.figure(figsize=(6,3))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95)
    plt.rcParams["font.size"] = FS
    # x = np.linspace(0, 2*np.pi, length)
    x = np.linspace(0, length, length)
    for i, data in enumerate(gene_data):
        plt.plot(x, data.detach().numpy(), label="Training data"+ str(i))
    plt.grid()
    # plt.ylim(bottom=0)
    # plt.title("Sin Curves")
    plt.ylim(top=10, bottom=-10)
    plt.xlim(0, length)
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel("t", fontsize=12)
    # plt.ylabel("y", fontsize=FS)
    # plt.legend()
    # fig.savefig(os.path.join(path, "Training"+id+".pdf"))
    fig.savefig(os.path.join(path, "Training"+".pdf"))


def main(args):
    # make directory for Data
    now = datetime.datetime.now().strftime('%Y%m%d_%H_%M')
    os.makedirs(os.path.join("saveEstimate", now), exist_ok=True)

    # save args as TEXT
    f = open(os.path.join("saveEstimate", now, 'args.txt'), 'w') # 書き込みモードで開く
    for name, site in vars(args).items():
        f.write(name +" = ") 
        f.write(str(site) + "\n")
    f.close() # ファイルを閉じる

    #Arguments
    N_songs = 10
    length = 20
    mini_batch_size = 20
    z_dim = 100
    rnn_dim = 200
    transition_dim = 200
    emission_dim =100

    reconstructions = []
    generations = []
    trainings = []
    for i in range(args.N_estimate):
        encoder = Encoder(input_dim=1, z_dim=z_dim, rnn_dim=rnn_dim, N_z0=N_songs, var_clip=args.clips[i])
        prior = Prior(z_dim=z_dim, transition_dim=transition_dim,  N_z0=N_songs, var_clip=args.clips[i])
        decoder = Emitter(input_dim=1, z_dim=z_dim, emission_dim=emission_dim)
        date = "2021" + args.dates[i]
        path = os.path.join("saveData", date)
        DMM_dics = torch.load(os.path.join(path,"DMM_dic"))
        training_data_sequences = DMM_dics["mini_batch"]

        encoder.load_state_dict(DMM_dics["Encoder_dic"]())
        prior.load_state_dict(DMM_dics["Prior_dic"]())
        decoder.load_state_dict(DMM_dics["Emitter_dic"]())

        which_mini_batch = 0
        training_seq_lengths = torch.tensor([length]*N_songs)
        data_dim = training_data_sequences.size(-1)

        N_train_data = len(training_seq_lengths)
        N_train_time_slices = float(torch.sum(training_seq_lengths))
        N_mini_batches = int(N_train_data / mini_batch_size +
                        int(N_train_data % mini_batch_size > 0))
        shuffled_indices = torch.randperm(N_train_data)
        mini_batch_start = (which_mini_batch * mini_batch_size)
        mini_batch_end = np.min([(which_mini_batch + 1) * mini_batch_size, N_train_data])
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]

        # grab a fully prepped mini-batch using the helper function in the data loader
        mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
            = poly.get_mini_batch(mini_batch_indices, training_data_sequences,
                                    training_seq_lengths)

        pri_z, pri_z_loc, pri_z_scale = prior(length = mini_batch.size(1), N_generate= args.N_generate)
        pos_z,a,b = encoder(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths)
        reconed_x = decoder(pos_z)
        reconstructions.append(reconed_x)
        generations.append(decoder(pri_z))
        trainings.append(mini_batch)
        entropy = 0
        entropy -= torch.sum(torch.log(multi_normal_prob(pri_z_loc, pri_z_scale, pri_z))) / args.N_generate
        entropy += 0.5 * pri_z.size(1) * pri_z.size(2) * torch.log(torch.tensor(2*math.pi))
        how_close = calc_how_close_to_mini_batch(decoder(pri_z), mini_batch)

        ppath = os.path.join("saveEstimate", now, date)
        os.makedirs(ppath, exist_ok=True)
        os.makedirs(os.path.join(ppath,"Reconstruction"), exist_ok=True)
        os.makedirs(os.path.join(ppath,"Generation"), exist_ok=True)
        os.makedirs(os.path.join(ppath,"Training"), exist_ok=True)
        for j in range(args.N_pics):
            saveReconSinGraph(mini_batch[j], reconed_x[j], pri_z.size(1), os.path.join(ppath,"Reconstruction"), j, date)
        saveGeneSinGraph(decoder(pri_z)[:3], pri_z.size(1), os.path.join(ppath,"Generation"), j, date)
        #saveTrainSinGraph(mini_batch, pri_z.size(1), os.path.join(ppath,"Training"), 1, date)

        # save args as TEXT
        f = open(os.path.join("saveEstimate", now, 'estimate.txt'), 'a') # 書き込みモードで開く
        f.write("2021" + args.dates[i] + "\n") 
        f.write("Entorpy = " + str(entropy) + "\n")
        f.write("How close to mini_batch = " + str(how_close) + "\n\n")
        f.close() # ファイルを閉じる

    ppath = os.path.join("saveEstimate", now)
    saveTrainSinGraph(mini_batch, pri_z.size(1), os.path.join(ppath), 1, date)
    ## PAINT PICTURE ##
    Ver = 2 * args.N_estimate
    Ver = 8
    Hor = 6
    fig = plt.figure(figsize=(Hor, Ver))
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95)
    FS = 6
    plt.rcParams["font.size"] = FS

    x = np.linspace(0, args.length, args.length)
    for i in range(args.N_estimate):
        plt.subplot(args.N_estimate, 2, 2*i+1)
        index = np.random.randint(0, args.N_songs)
        plt.plot(x, trainings[i][index], label="Training data", color="black")
        plt.plot(x, reconstructions[i][index].detach().numpy(), label="Reconstructed data", color="orange")
        plt.grid()
        plt.ylim(top=10, bottom=-10)
        plt.xlim(0, length)
        if i == args.N_estimate-1:
            plt.xlabel("t", fontsize=10)
        plt.legend()
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))

        plt.subplot(args.N_estimate, 2, 2*i+2)
        for j in range(3):
            plt.plot(x, generations[i][j].detach().numpy(), label="Generated data" + str(j+1))
        plt.grid()
        plt.ylim(top=10, bottom=-10)
        plt.xlim(0, args.length)
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        if i == args.N_estimate-1:
            plt.xlabel("t", fontsize=10)
        plt.legend()
    fig.savefig(os.path.join(ppath, "Results"+".pdf"))




# parse command-line arguments and execute the main method
if __name__ == '__main__':
    # assert pyro.__version__.startswith('1.5.1')

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-nes', '--N-estimate', type=int, default=5)
    parser.add_argument('--dates', type=str, nargs="*", required=True)
    parser.add_argument('--clips', type=float, nargs="*")
    parser.add_argument('-nsongs', '--N-songs', type=int, default=10)
    parser.add_argument('-length', '--length', type=int, default=20)
    parser.add_argument('-npic', '--N-pics', type=int, default=10)
    parser.add_argument('-ngen', '--N-generate', type=int, default=100)
    args = parser.parse_args()

    main(args)



