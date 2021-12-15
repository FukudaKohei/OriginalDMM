import argparse
import logging
import time
import random
import datetime
import tqdm
from os.path import exists
import os

import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, input_dim=88, z_dim=100,rnn_dim=60, num_layers=1, rnn_dropout_rate=0.1,num_iafs=0, iaf_dim=50, N_z0 = 10, use_cuda=False, rnn_check=False):
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
            if args.clip != None:
                z_scale = torch.clamp(z_scale, min = args.clip)
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
    def __init__(self, z_dim=100, transition_dim=200,  N_z0 = 10, use_cuda=False):
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
            if args.clip != None:
                z_scale = torch.clamp(z_scale, min = args.clip)

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
    mom = torch.sqrt(det) * pi_2_d
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


def main(args):

    ## This func is for saving losses at final
    def saveGraph(loss_list, sub_error_list,now):
        FS = 10
        fig = plt.figure()
        # plt.rcParams["font.size"] = FS
        plt.plot(loss_list, label="LOSS")
        plt.plot(sub_error_list, label="Reconstruction Error")
        plt.ylim(bottom=0)
        plt.title("Loss")
        plt.xlabel("epoch", fontsize=FS)
        plt.ylabel("loss", fontsize=FS)
        plt.legend()
        fig.savefig(os.path.join("saveData", now, "LOSS.png"))
        plt.close()


    def saveReconSinGraph(train_data, recon_data, length, path, number):
        FS = 10
        fig = plt.figure()
        # plt.rcParams["font.size"] = FS
        x = np.linspace(0, 2*np.pi, length)
        plt.plot(x, train_data, label="Training data")
        plt.plot(x, recon_data.detach().numpy(), label="Reconstructed data")
        # plt.ylim(bottom=0)
        plt.title("Sin Curves")
        # plt.ylim(top=10, bottom=-10)
        plt.xlabel("time", fontsize=FS)
        plt.ylabel("y", fontsize=FS)
        plt.legend()
        fig.savefig(os.path.join(path, "Reconstruction"+str(number)+".png"))
        plt.close()


    def saveGeneSinGraph(gene_data, length, path, number):
        FS = 10
        fig = plt.figure()
        # plt.rcParams["font.size"] = FS
        x = np.linspace(0, 2*np.pi, length)
        plt.plot(x, gene_data.detach().numpy(), label="Generated data")
        # plt.ylim(bottom=0)
        plt.title("Sin Curves")
        # plt.ylim(top=10, bottom=-10)
        plt.xlabel("time", fontsize=FS)
        plt.ylabel("y", fontsize=FS)
        plt.legend()
        fig.savefig(os.path.join(path, "Generation"+str(number)+".png"))
        plt.close()


    ## This func is for save generatedTones and trainingTones as MIDI
    def save_as_midi(song, path="", name="default.mid", BPM = 120, velocity = 100):
        pm = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=BPM) #pretty_midiオブジェクトを作ります
        instrument = pretty_midi.Instrument(0) #instrumentはトラックみたいなものです。
        for i,tones in enumerate(song):
            which_tone = torch.nonzero((tones == 1), as_tuple=False).reshape(-1)
            if len(which_tone) == 0:
                note = pretty_midi.Note(velocity=0, pitch=0, start=i, end=i+1) #noteはNoteOnEventとNoteOffEventに相当します。
                instrument.notes.append(note)
            else:
                for which in which_tone:
                    note = pretty_midi.Note(velocity=velocity, pitch=int(which), start=i, end=i+1) #noteはNoteOnEventとNoteOffEventに相当します。
                    instrument.notes.append(note)
        pm.instruments.append(instrument)
        pm.write(os.path.join(path, name)) #midiファイルを書き込みます。

    # generate Xs
    def generate_Xs(batch_data):
        songs_list = []
        for i, song in enumerate(batch_data):
            tones_container = []
            for time in range(batch_data.size(1)):
                p = dist.Bernoulli(probs=song[time])
                tone = p.sample()
                tones_container.append(tone)
            tones_container = torch.stack(tones_container)
            songs_list.append(tones_container)
        return songs_list

    def saveSongs(songs_list, mini_batch, N_songs, path):
        # print(len(songs_list[0][0]))
        if len(songs_list) != len(mini_batch):
            assert False
        if N_songs <= len(songs_list):
            song_No = random.sample(range(len(songs_list)), k=N_songs)
        else :
            song_No = random.sample(range(len(songs_list)), k=len(songs_list))
        for i, Number in enumerate(song_No):
            save_as_midi(song=songs_list[Number], path=path, name="No%d_Gene.midi"%i)
            save_as_midi(song=mini_batch[Number], path=path, name="No%d_Tran.midi"%i)

    FS = 10
    plt.rcParams["font.size"] = FS

    ## 長さ最長129、例えば長さが60のやつは61~129はすべて0データ
    data = poly.load_data(poly.JSB_CHORALES)
    training_seq_lengths = data['train']['sequence_lengths'][:args.N_songs]
    training_data_sequences = data['train']['sequences'][:args.N_songs,:8]
    training_seq_lengths = torch.tensor([8]*args.N_songs)

    if args.doremifasorasido:
        # ## ドドド、レレレ、ミミミ、ドレミ
        training_seq_lengths, training_data_sequences = musics.doremifasorasido(args.N_songs)
    if args.dododorerere:
        # ドドド、レレレ
        training_seq_lengths, training_data_sequences = musics.dododorerere(args.N_songs)
    if args.dodododododo:
        ## ドドドのみ
        training_seq_lengths, training_data_sequences = musics.dodododododo(args.N_songs)

    if args.sin:
        training_data_sequences = musics.createSin_allChanged(args.N_songs, args.length)
        training_seq_lengths = torch.tensor([args.length]*args.N_songs)

    training_data_sequences = musics.createNewTrainingData(args.N_songs, args.length)
    # traing_data =[]
    # for i in range(args.N_songs):
    #     traing_data.append(musics.Nonlinear(torch.randn(1), args.length, T = args.T))
    # training_data_sequences = torch.stack(traing_data)
    training_seq_lengths = torch.tensor([args.length]*args.N_songs)
    data_dim = training_data_sequences.size(-1)

    N_train_data = len(training_seq_lengths)
    N_train_time_slices = float(torch.sum(training_seq_lengths))
    N_mini_batches = int(N_train_data / args.mini_batch_size +
                        int(N_train_data % args.mini_batch_size > 0))

    z_dim = args.z_dim #100 #2
    rnn_dim = args.rnn_dim #200 #30
    transition_dim = args.transition_dim #200 #30
    emission_dim = args.emission_dim #100 #30
    encoder = Encoder(input_dim=training_data_sequences.size(-1), z_dim=z_dim, rnn_dim=rnn_dim, N_z0=args.N_songs)
    # encoder = Encoder(z_dim=100, rnn_dim=200)
    prior = Prior(z_dim=z_dim, transition_dim=transition_dim, N_z0=args.N_songs, use_cuda=args.cuda)
    decoder = Emitter(input_dim=training_data_sequences.size(-1), z_dim=z_dim, emission_dim=emission_dim, use_cuda=args.cuda)

    # Create optimizer algorithm
    # optimizer = optim.SGD(dmm.parameters(), lr=args.learning_rate)
    # optimizer = optim.Adam(dmm.parameters(), lr=args.learning_rate, betas=(0.96, 0.999), weight_decay=2.0)
    params = list(encoder.parameters()) + list(prior.parameters()) + list(decoder.parameters()) 
    optimizer = optim.Adam(params, lr=args.learning_rate)
    # Add learning rate scheduler
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9999) 
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 1.) 
    #1でもやってみる

    # make directory for Data
    now = datetime.datetime.now().strftime('%Y%m%d_%H_%M')
    os.makedirs(os.path.join("saveData", now), exist_ok=True)

    # save args as TEXT
    f = open(os.path.join("saveData", now, 'args.txt'), 'w') # 書き込みモードで開く
    for name, site in vars(args).items():
        f.write(name +" = ") 
        f.write(str(site) + "\n")
    f.close() # ファイルを閉じる

    #######################
    #### TRAINING LOOP ####
    #######################
    times = [time.time()]
    losses = []
    recon_errors = []
    pbar = tqdm.tqdm(range(args.num_epochs))
    for epoch in pbar:
        epoch_nll = 0
        epoch_recon_error = 0
        shuffled_indices = torch.randperm(N_train_data)
        # print("Proceeding: %.2f " % (epoch*100/5000) + "%")

        # process each mini-batch; this is where we take gradient steps
        for which_mini_batch in range(N_mini_batches):

            # compute which sequences in the training set we should grab
            mini_batch_start = (which_mini_batch * args.mini_batch_size)
            mini_batch_end = np.min([(which_mini_batch + 1) * args.mini_batch_size, N_train_data])
            mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]

            # grab a fully prepped mini-batch using the helper function in the data loader
            mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
                = poly.get_mini_batch(mini_batch_indices, training_data_sequences,
                                        training_seq_lengths, cuda=args.cuda)

            # reset gradients
            optimizer.zero_grad()
            loss = 0
            
            N_repeat = 1
            for i in range(N_repeat):
                seiki = 1/N_repeat
                # generate mini batch from training mini batch
                pos_z, pos_z_loc, pos_z_scale = encoder(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths)
                pri_z, pri_z_loc, pri_z_scale = prior(length = mini_batch.size(1), N_generate= mini_batch.size(0))
        
                # Regularizer (KL-diveergence(pos||pri))
                regularizer = seiki * D_Wass(pos_z_loc, pri_z_loc, pos_z_scale, pri_z_scale)
                # regularizer = seiki * D_KL(pos_z_loc, pri_z_loc, pos_z_scale, pri_z_scale)
                # regularizer = seiki * D_JS_Monte(pos_z_loc, pri_z_loc, pos_z_scale, pri_z_scale, pos_z, pri_z)
                reconstruction_error = seiki * torch.norm(mini_batch - decoder(pos_z), dim=2).sum()/mini_batch.size(0)/mini_batch.size(1)/mini_batch.size(2)

                loss += args.lam * regularizer  + reconstruction_error
            reconed_x = decoder(pos_z)
            # Reconstruction Error
            # reconstruction_error = torch.norm(mini_batch - reconed_x, dim=2).sum()/mini_batch.size(0)/mini_batch.size(1)/mini_batch.size(2)
            # loss += reconstruction_error

            # # Covariance Penalty
            # sumOfCovariance = torch.sum(pos_z_scale + pri_z_scale)
            # loss -= 0.00007 * sumOfCovariance
            
            # # generate mini batch from training mini batch
            # pos_z, pos_z_loc, pos_z_scale = encoder(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths)
            # pri_z, pri_z_loc, pri_z_scale = prior(length = mini_batch.size(1), N_generate= mini_batch.size(0))
    
            # reconed_x = decoder(pos_z)
            # # Reconstruction Error
            # reconstruction_error = torch.norm(mini_batch - reconed_x, dim=2).sum()/mini_batch.size(0)/mini_batch.size(1)/mini_batch.size(2)
            # # Regularizer (KL-diveergence(pos||pri))
            # regularizer = D_Wass(pos_z_loc, pri_z_loc, pos_z_scale, pri_z_scale)

            # loss += reconstruction_error + args.lam * regularizer

            # # NaN detector
            # if torch.isnan(loss):
            #     saveDic = {
            #     "optimizer":optimizer,
            #     "mini_batch":mini_batch,
            #     "Encoder_dic": encoder.state_dict,
            #     "Prior_dic": prior.state_dict,
            #     "Emitter_dic": decoder.state_dict,
            #     }
            #     torch.save(saveDic,os.path.join("saveData", now, "fail_DMM_dic"))
            #     assert False, ("LOSS ia NaN!")


            # saveDic = {
            # "optimizer":optimizer,
            # "mini_batch":mini_batch,
            # "Encoder_dic": encoder.state_dict,
            # "Prior_dic": prior.state_dict,
            # "Emitter_dic": decoder.state_dict,
            # }
            # torch.save(saveDic,os.path.join("saveData", now, "DMM_dic"))

            # do an actual gradient step
            loss.backward()
            optimizer.step()
            scheduler.step()
            # if args.rnn_clip != None:
            #     for p in dmm.rnn.parameters():
            #         p.data.clamp_(-args.rnn_clip, args.rnn_clip)

            epoch_nll += loss
            epoch_recon_error += reconstruction_error
        
        # report training diagnostics
        times.append(time.time())
        losses.append(epoch_nll)
        recon_errors.append(epoch_recon_error)
        # epoch_time = times[-1] - times[-2]
        pbar.set_description("LOSS = %f " % epoch_nll)
        
        if epoch % args.checkpoint_freq == 0:
            path = os.path.join("saveData", now, "Epoch%d"%epoch)
            os.makedirs(path, exist_ok=True)
            if data_dim != 1:
                reco_list = generate_Xs(reconed_x)
                gene_list = generate_Xs(decoder(pri_z))
                for i in range(args.N_generate):
                    save_as_midi(song=reco_list[i], path=path, name="No%d_Reco.midi"%i)
                    save_as_midi(song=mini_batch[i], path=path, name="No%d_Real.midi"%i)
                    save_as_midi(song=gene_list[i], path=path, name="No%d_Gene.midi"%i)
            else:
                for i in range(len(mini_batch)):
                    saveReconSinGraph(mini_batch[i], reconed_x[i], args.length, path, i)
                    saveGeneSinGraph(decoder(pri_z)[i], args.length, path, i)

        if epoch == args.num_epochs-1:
            path = os.path.join("saveData", now, "Epoch%d"%args.num_epochs)
            os.makedirs(path, exist_ok=True)
            if data_dim != 1:
                reco_list = generate_Xs(reconed_x)
                gene_list = generate_Xs(decoder(pri_z))
                for i in range(args.N_generate):
                    save_as_midi(song=reco_list[i], path=path, name="No%d_Reco.midi"%i)
                    save_as_midi(song=mini_batch[i], path=path, name="No%d_Real.midi"%i)
                    save_as_midi(song=gene_list[i], path=path, name="No%d_Gene.midi"%i)
            else:
                for i in range(len(mini_batch)):
                    saveReconSinGraph(mini_batch[i], reconed_x[i], args.length, path, i)
                    saveGeneSinGraph(decoder(pri_z)[i], args.length, path, i)

            saveGraph(losses, recon_errors, now)
            saveDic = {
                "optimizer":optimizer,
                "mini_batch":mini_batch,
                "Encoder_dic": encoder.state_dict,
                "Prior_dic": prior.state_dict,
                "Emitter_dic": decoder.state_dict,
                "epoch_times": times,
                "losses":losses
            }
            # torch.save(saveDic,os.path.join("saveData", now, "dic_Epoch%d"%(epoch+1)))
            torch.save(saveDic,os.path.join("saveData", now, "DMM_dic"))


# parse command-line arguments and execute the main method
if __name__ == '__main__':
    # assert pyro.__version__.startswith('1.5.1')

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', type=int, default=5000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.00001)
    parser.add_argument('-lam', '--lam', type=float, default=0.1)
    parser.add_argument('-nsongs', '--N-songs', type=int, default=10)
    parser.add_argument('-length', '--length', type=int, default=10)
    parser.add_argument('-ngen', '--N-generate', type=int, default=5)
    parser.add_argument('-t', '--T', type=int, default=10)
    parser.add_argument('-z', '--z-dim', type=int, default=100)
    parser.add_argument('-rnn', '--rnn-dim', type=int, default=200)
    parser.add_argument('-tra', '--transition-dim', type=int, default=200)
    parser.add_argument('-emi', '--emission-dim', type=int, default=100)
    # parser.add_argument('-rcl', '--rnn-clip', type=float, default=None)
    parser.add_argument('--sin', action='store_true')
    parser.add_argument('-clip', "--clip",type=float, default=None)
    parser.add_argument('--doremifasorasido', action='store_true')
    parser.add_argument('--dododorerere', action='store_true')
    parser.add_argument('--dodododododo', action='store_true')
    parser.add_argument('-b1', '--beta1', type=float, default=0.96)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=10.0)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=0.99996)
    parser.add_argument('-wd', '--weight-decay', type=float, default=2.0)
    parser.add_argument('-mbs', '--mini-batch-size', type=int, default=20)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=1000)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.2)
    parser.add_argument('-rdr', '--rnn-dropout-rate', type=float, default=0.1)
    parser.add_argument('-iafs', '--num-iafs', type=int, default=0)
    parser.add_argument('-id', '--iaf-dim', type=int, default=100)
    parser.add_argument('-cf', '--checkpoint-freq', type=int, default=0)
    parser.add_argument('-lopt', '--load-opt', type=str, default='')
    parser.add_argument('-lmod', '--load-model', type=str, default='')
    parser.add_argument('-sopt', '--save-opt', type=str, default='')
    parser.add_argument('-smod', '--save-model', type=str, default='')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--tmc', action='store_true')
    parser.add_argument('--tmcelbo', action='store_true')
    parser.add_argument('--tmc-num-samples', default=10, type=int)
    parser.add_argument('-l', '--log', type=str, default='dmm.log')
    args = parser.parse_args()

    main(args)



