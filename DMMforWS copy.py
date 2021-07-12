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

import wave
from scipy.io.wavfile import write
# import fluidsynth


class Emitter(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`
    """

    def __init__(self, input_dim, z_dim, emission_dim):
        super().__init__()
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
        ps = torch.sigmoid(self.lin_hidden_to_input(h2))
        return ps

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

class DMM(nn.Module):
    
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """

    def __init__(self, input_dim=88, z_dim=100, emission_dim=100,
                 transition_dim=200, rnn_dim=60, num_layers=1, rnn_dropout_rate=0.1,
                 num_iafs=0, iaf_dim=50, use_cuda=False, rnn_check=False, rpt=True):
        super().__init__()
        # instantiate PyTorch modules used in the model and guide below
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)
        # dropout just takes effect on inner layers of rnn
        rnn_dropout_rate = 0. if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                          batch_first=True, bidirectional=False, num_layers=num_layers,
                          dropout=rnn_dropout_rate)

        # if we're using normalizing flows, instantiate those too
        self.iafs = [affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()

        self.rnn_check = rnn_check
        self.rpt = rpt

    def forward(self, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, annealing_factor=1.0):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)

        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
        # if any(torch.isnan(h_0_contig.reshape(-1))):
        #     print("h_0_contig")
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        if self.rnn_check:
            if any(torch.isnan(rnn_output.data.reshape(-1))):
                # print("rnn_output First")
                # print(self.rnn.state_dict().items())
                # print(rnn_output)
                # torch.save(rnn_output, "out")
                # torch.save(self.rnn.state_dict().items, "dic")
                # torch.save(mini_batch_reversed, "mini_batch_reversed")
                # torch.save(h_0_contig, "h_0_contig")
                assert False

        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        # print(rnn_output.size())
        # assert False
        # if any(torch.isnan(rnn_output.reshape(-1))):
        #     print("rnn_output")
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))
        # if any(torch.isnan(z_prev.reshape(-1))):
        #     print("z_prev")

        x_container = []
        for t in range(1,T_max+1):
            # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
            z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

            # Reparameterization Trick
            if self.use_cuda:
                eps = torch.randn(z_loc.size()).cuda()
            else: eps = torch.randn(z_loc.size())
            z_t = z_loc + z_scale * eps

            # compute the probabilities that parameterize the bernoulli likelihood
            emission_probs_t = self.emitter(z_t)

            # the next statement instructs pyro to observe x_t according to the
            # bernoulli distribution p(x_t|z_t)

            #Reparameterization Trick
            # eps = torch.rand(88)
            # # assert len(emission_probs_t) == 88
            # appxm = torch.log(eps) - torch.log(1-eps) + torch.log(probs) - torch.log(1-probs)
            # x = torch.sigmoid(args)

            # No Reparameterization Trick
            x = emission_probs_t

            #Reparameterization Trick
            if self.rpt :
                if self.use_cuda: 
                    eps = torch.rand(88).cuda()
                else : eps = torch.rand(88)
                # assert len(emission_probs_t) == 88
                appxm = torch.log(eps + 1e-20) - torch.log(1-eps + 1e-20) + torch.log(x + 1e-20) - torch.log(1-x + 1e-20)
                # appxm = torch.log(eps) - torch.log(1-eps) + torch.log(x) - torch.log(1-x)
                x = torch.sigmoid(appxm)

            # the latent sampled at this time step will be conditioned upon
            # in the next time step so keep track of it
            z_prev = z_t
            x_container.append(x)

        x_container = torch.stack(x_container)
        return x_container.transpose(0,1)

class WGAN_network(nn.Module):
    def __init__(self, hiddden_dim=256, use_cuda=False):
        super().__init__()

        ## the number of tones
        self.song_size = 88
        ## the length of each song
        self.song_length = 8
        
        self.input_size = self.song_size * self.song_length
        self.hidden_size = hiddden_dim

        self.D =  nn.Sequential(
                    # nn.Linear(self.input_size, self.hidden_size),
                    nn.Linear(self.song_size, self.hidden_size),
                    nn.LeakyReLU(0.2),
                    nn.Linear(self.hidden_size, 1))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()
    
    def forward(self, train_mini_batch, generated_mini_batch):
        # train_mini_batch =  train_mini_batch.reshape(len(train_mini_batch),-1)
        # generated_mini_batch = generated_mini_batch.reshape(len(generated_mini_batch),-1)
        outputs_real = self.D(train_mini_batch)
        outputs_fake = self.D(generated_mini_batch)
        # print(outputs_real.size())
        # print(outputs_fake.size())
        # if any(torch.isnan(outputs_real.reshape(-1))):
        #     print("REAL")
        # if any(torch.isnan(generated_mini_batch.reshape(-1))):
        #     print("GENERATED")
        # if any(torch.isnan(outputs_fake.reshape(-1))):
        #     print("FAKE")
        # if torch.isnan(-(torch.mean(outputs_real) - torch.mean(outputs_fake))):
        #     print("OUTPUT")
        return (torch.mean(outputs_real) - torch.mean(outputs_fake))

class WassersteinLoss(nn.Module):
    def __init__(self, WGAN_network, N_loops=5, lr=0.00001, use_cuda=False):
        super().__init__()
        self.WGAN_network = WGAN_network
        self.D = self.WGAN_network.D
        self.optimizer = torch.optim.RMSprop(self.D.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adam(self.D.parameters(), lr=lr)
        # the number of loops of calculation of Wass
        self.N_loops = N_loops

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()


    def calc(self, train_mini_batch, generated_mini_batch):
        # vanish grad_fn of DMM's parameters
        no_grad_generated_mini_batch = generated_mini_batch.clone().detach()
        # no_grad_generated_mini_batch = torch.tensor(generated_mini_batch)

        # activate grad_fn of Wass calculator's parameters 
        for p in self.D.parameters():
                p.requires_grad = True

        ### CALCULATION LOOP ###       
        for i in range(self.N_loops):
            # print("{0}%".format(i*20))
            # NaN_detect(self.D,i,"Before calc")
            minus_loss = - self.WGAN_network(train_mini_batch, no_grad_generated_mini_batch)
            # print(loss)
            self.optimizer.zero_grad()
            minus_loss.backward() # 勾配を計算
            self.optimizer.step() # 重みパラメータを更新 (Algorithm 1のStep 6)

            # NaN_detect(self.D,i,"Before Clip")
            # 重みパラメータの値を-0.01から0.01の間にクリッピングする (Algorithm 1のStep 7)
            for p in self.D.parameters():
                p.data.clamp_(-0.01, 0.01)
            # NaN_detect(self.D,i,"After Clip")            

        # vanish grad_fn of Wass calculator's parameters
        # because we don't need to update Wass calculator's parameters anymore
        for p in self.D.parameters():
                p.requires_grad = False

        # calculate Wass without Wass calculator's parameters's grad_fn
        loss = self.WGAN_network(train_mini_batch, generated_mini_batch)

        # save model
        # torch.save(self.D.state_dict(), "D")

        # Wass is positive
        return loss

def EMD(xT, mud_arr):
    # print("DONE0")
    x_p = xT.detach().clone().numpy()
    mud_p = mud_arr.detach().clone().numpy()
    
    n_points = xT.size()[0]
    n_tones = xT.size()[1]
    dim = xT.size()[2]
    # print("DONE1")
    # x_p = xT.reshape((n_points, dim)).cpu().detach().numpy()
    x_p = x_p.reshape((n_points, n_tones * dim))
    # mud_p = mud_arr.reshape((n_points, dim)).cpu().detach().numpy()
    mud_p = mud_p.reshape((n_points, n_tones * dim))
    # print(x_p.shape)
    # print(mud_p.shape)
    # print("DONE2")
    M = ot.dist(x_p, mud_p)
    # M /= M.max()
    a, b = np.ones((n_points,)) / n_points, np.ones((n_points,)) / n_points  # uniform distribution on samples
    gamma = ot.emd(a, b, M)
    # wdis = ot.emd2_1d(xT, mud_arr)
    # print('Wasserstein distance:', wdis)
    # print("DONE3")

    Tx = []
    for i, xTi in enumerate(x_p):
        ind = gamma[i,:].argmax()
        Tx.append(mud_arr[ind])
        # print(f'index {i}: ind {ind} mud_p[{ind}]={mud_p[ind]}') 
    # print("DONE4")
    # print(Tx[0])
    # Tx = torch.from_numpy(np.array(Tx).reshape((n_points, 1, n_tones*dim)))
    # Tx = torch.tensor(Tx,dtype=torch.float32)
    # x = xT.reshape(n_points,1,n_tones*dim)
    # Wasser = F.mse_loss(x, Tx)
    Tx = torch.stack(Tx)
    return torch.norm(Tx-xT, dim=1).sum()


def main(args):

    ## ドレミ
    def easyTones(N_songs):
        max = 70
        interval = 10
        N = N_songs
        training_seq_lengths = torch.tensor([8]*N)
        training_data_sequences = torch.zeros(N,8,88)
        for i in range(N):
            training_data_sequences[i][0][int(max-i*interval)  ] = 1
            training_data_sequences[i][1][int(max-i*interval)+2] = 1
            training_data_sequences[i][2][int(max-i*interval)+4] = 1
            training_data_sequences[i][3][int(max-i*interval)+5] = 1
            training_data_sequences[i][4][int(max-i*interval)+7] = 1
            training_data_sequences[i][5][int(max-i*interval)+9] = 1
            training_data_sequences[i][6][int(max-i*interval)+11] = 1
            training_data_sequences[i][7][int(max-i*interval)+12] = 1
        return training_seq_lengths, training_data_sequences
    ## ドドド、レレレ
    def superEasyTones(N_songs):
        training_seq_lengths = torch.tensor([8]*N_songs)
        training_data_sequences = torch.zeros(N_songs,8,88)
        for i in range(N_songs):
            for j in range(8):
                training_data_sequences[i][j][int(30+i*5)] = 1
        return training_seq_lengths, training_data_sequences
    ## ドドド、ドドド、ドドド
    def easiestTones(N_songs):
        training_seq_lengths = torch.tensor([8]*N_songs)
        training_data_sequences = torch.zeros(N_songs,8,88)
        for i in range(N_songs):
            for j in range(8):
                training_data_sequences[i][j][int(70)] = 1
        return training_seq_lengths, training_data_sequences
    # rep is short for "repeat"
    # which means how many times we use certain sample to do validation/test evaluation during training
    def rep(x):
            rep_shape = torch.Size([x.size(0) * n_eval_samples]) + x.size()[1:]
            repeat_dims = [1] * len(x.size())
            repeat_dims[0] = n_eval_samples
            return x.repeat(repeat_dims).reshape(n_eval_samples, -1).transpose(1, 0).reshape(rep_shape)

    ## This func is for saving losses at final
    def saveGraph(loss_list, now):
        FS = 10
        fig = plt.figure()
        plt.rcParams["font.size"] = FS
        plt.plot(loss_list)
        plt.ylim(bottom=0)
        plt.title("Wasserstein Loss")
        plt.xlabel("epoch", fontsize=FS)
        plt.ylabel("loss", fontsize=FS)
        fig.savefig(os.path.join("saveData", now, "LOSS.png"))

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
    def generate_Xs(dmm, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths):
        songs_list = []
        generated_mini_batch = dmm(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths)
        for i, song in enumerate(generated_mini_batch):
            tones_container = []
            for time in range(mini_batch_seq_lengths[i]):
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


    ## 長さ最長129、例えば長さが60のやつは61~129はすべて0データ
    data = poly.load_data(poly.JSB_CHORALES)
    training_seq_lengths = data['train']['sequence_lengths'][:args.N_songs]
    training_data_sequences = data['train']['sequences'][:args.N_songs,:8]
    training_seq_lengths = torch.tensor([8]*args.N_songs)

    if args.eas:
        # ## ドドド、レレレ、ミミミ、ドレミ
        training_seq_lengths, training_data_sequences = easyTones(args.N_songs)
    
    if args.sup:
        # ドドド、レレレ
        training_seq_lengths, training_data_sequences = superEasyTones(args.N_songs)

    if args.est:
        ## ドドドのみ
        training_seq_lengths, training_data_sequences = easiestTones(args.N_songs)


    # test_seq_lengths = data['test']['sequence_lengths'][:3]
    # test_seq_lengths = torch.tensor([8,8,8])
    # test_data_sequences = data['test']['sequences'][:3,:8]
    # val_seq_lengths = data['valid']['sequence_lengths']
    # val_data_sequences = data['valid']['sequences']
    N_train_data = len(training_seq_lengths)
    # N_train_data = len(test_seq_lengths)
    N_train_time_slices = float(torch.sum(training_seq_lengths))
    N_mini_batches = int(N_train_data / args.mini_batch_size +
                        int(N_train_data % args.mini_batch_size > 0))


    # how often we do validation/test evaluation during training
    val_test_frequency = 50
    # the number of samples we use to do the evaluation
    n_eval_samples = 1

    # get the validation/test data ready for the dmm: pack into sequences, etc.
    # val_seq_lengths = rep(val_seq_lengths)
    # # test_seq_lengths = rep(test_seq_lengths)
    # val_batch, val_batch_reversed, val_batch_mask, val_seq_lengths = poly.get_mini_batch(
    #     torch.arange(n_eval_samples * val_data_sequences.shape[0]), rep(val_data_sequences),
    #     val_seq_lengths, cuda=args.cuda)
    # test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths = poly.get_mini_batch(
    #     torch.arange(n_eval_samples * test_data_sequences.shape[0]), rep(test_data_sequences),
    #     test_seq_lengths, cuda=args.cuda)



    dmm = DMM(use_cuda=args.cuda, rnn_check=args.rck, rpt=args.rpt)
    WN = WGAN_network(use_cuda = args.cuda)
    W = WassersteinLoss(WN, N_loops=args.N_loops, lr=args.w_learning_rate, use_cuda = args.cuda)

    # Create optimizer algorithm
    # optimizer = optim.SGD(dmm.parameters(), lr=args.learning_rate)
    # optimizer = optim.Adam(dmm.parameters(), lr=args.learning_rate, betas=(0.96, 0.999), weight_decay=2.0)
    optimizer = optim.Adam(dmm.parameters(), lr=args.learning_rate)
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
    pbar = tqdm.tqdm(range(args.num_epochs))
    for epoch in pbar:
        epoch_nll = 0
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
            # mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
            #     = poly.get_mini_batch(mini_batch_indices, test_data_sequences,
            #                             test_seq_lengths, cuda=args.cuda)

            # reset gradients
            optimizer.zero_grad()

            # generate mini batch from training mini batch
            # NaN_detect(dmm, epoch, message="Before Generate")                          
            generated_mini_batch = dmm(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths)
            # NaN_detect(dmm, epoch, message="After Generate")
            # if any(torch.isnan(generated_mini_batch.reshape(-1))):
            #     print("GENERATED")
            #     assert False
            # NaN_detect(WN, epoch, message="calc_Before")
            # calculate loss
            if args.wass:
                loss = EMD(mini_batch, generated_mini_batch)
            if args.wgan:
                loss = W.calc(mini_batch, generated_mini_batch)
            # NaN_detect(WN, epoch, message="calc_After")

            # NaN_detect(dmm, epoch, message="step_Before")        
            # do an actual gradient step
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.rnn_clip != None:
                for p in dmm.rnn.parameters():
                    p.data.clamp_(-args.rnn_clip, args.rnn_clip)
                    # p.data.clamp_(-0.01, 0.01)
            # NaN_detect(dmm, epoch, message="step_After")        

            epoch_nll += loss
        
        # report training diagnostics
        times.append(time.time())
        losses.append(epoch_nll)
        epoch_time = times[-1] - times[-2]
        # logging.info("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" %
        #                 (epoch, epoch_nll / N_train_time_slices, epoch_time))
        # if epoch % 10 == 0:
            # print("epoch %d time : %d sec" % (epoch, int(epoch_time)))
        pbar.set_description("LOSS = %f " % epoch_nll)
            # tqdm.write("\r"+"        loss : %f " % epoch_nll, end="")
        
        if epoch % args.checkpoint_freq == 0:
            path = os.path.join("saveData", now, "Epoch%d"%epoch)
            os.makedirs(path, exist_ok=True)
            songs_list = generate_Xs(dmm, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths)
            saveSongs(songs_list, mini_batch, N_songs=2, path=path)

        if epoch == args.num_epochs-1:
            path = os.path.join("saveData", now, "Epoch%d"%args.num_epochs)
            os.makedirs(path, exist_ok=True)
            songs_list = generate_Xs(dmm, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths)
            saveSongs(songs_list, mini_batch, N_songs=2, path=path)

            saveGraph(losses, now)
            saveDic = {
                "DMM_dic": dmm.state_dict,
                "WGAN_Network_dic": WN.state_dict,
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
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0003)
    parser.add_argument('-wlr', '--w-learning-rate', type=float, default=0.00001)
    parser.add_argument('-nls', '--N-loops', type=int, default=10)
    parser.add_argument('-nsongs', '--N-songs', type=int, default=10)
    parser.add_argument('--rpt', action='store_true')
    parser.add_argument('-rcl', '--rnn-clip', type=float, default=None)
    parser.add_argument('--wgan', action='store_true')
    parser.add_argument('--wass', action='store_true')
    parser.add_argument('--rck', action='store_true')
    parser.add_argument('--est', action='store_true')
    parser.add_argument('--sup', action='store_true')
    parser.add_argument('--eas', action='store_true')
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



