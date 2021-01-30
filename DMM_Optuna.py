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

import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage

from midi2audio import FluidSynth

import pretty_midi
from scipy.io import wavfile

import optuna



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
                 transition_dim=200, rnn_dim=600, num_layers=1, rnn_dropout_rate=0.1,
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
            eps = torch.randn(z_loc.size())
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
                eps = torch.rand(88)
                rpt_eps = 1e-20
                # assert len(emission_probs_t) == 88
                appxm = torch.log(eps + rpt_eps) - torch.log(1-eps + rpt_eps) + torch.log(x + rpt_eps) - torch.log(1-x + rpt_eps)
                # appxm = torch.log(eps) - torch.log(1-eps) + torch.log(x) - torch.log(1-x)
                x = torch.sigmoid(appxm)

            # the latent sampled at this time step will be conditioned upon
            # in the next time step so keep track of it
            z_prev = z_t
            x_container.append(x)

        x_container = torch.stack(x_container)
        return x_container.transpose(0,1)

class WGAN_network(nn.Module):
    def __init__(self, hiddden_dim=256):
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

class WassersteinLoss():
    def __init__(self, WGAN_network, N_loops=5, lr=0.00001):
        self.WGAN_network = WGAN_network
        self.D = self.WGAN_network.D
        self.optimizer = torch.optim.RMSprop(self.D.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adam(self.D.parameters(), lr=lr)
        # the number of loops of calculation of Wass
        self.N_loops = N_loops


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


def objective(trial):

    num_epochs=5000
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-8, 1e-2)
    if_rnn_clip = False
    rnn_clip = 0.01
    w_learning_rate=0.00001
    N_loops = 30
    rck = False
    eas = False
    sup = True
    est = False
    # beta1=0.96
    # beta2=0.999
    # clip_norm=10.0
    # lr_decay=0.99996
    mini_batch_size=20
    # annealing_epochs=1000
    # minimum_annealing_factor=0.2
    # rnn_dropout_rate=0.1
    # num_iafs=0
    # iaf_dim=100
    # checkpoint_freq=100
    # load_opt=''
    # load_model=''
    # save_opt=''
    # save_model=''
    cuda=False
    # jit=False
    # tmc=False
    # tmcelbo=False
    rpt = True
    # tmc_num_samples=10

    ## ドドド、レレレ、ミミミ、ドレミ
    def easyTones():
        training_seq_lengths = torch.tensor([8]*10)
        training_data_sequences = torch.zeros(10,8,88)
        for i in range(5):
            for j in range(8):
                training_data_sequences[i][j][int(20+i*10)] = 1
        for i in range(5,8):
            training_data_sequences[i][0][int(110-i*10)  ] = 1
            training_data_sequences[i][1][int(110-i*10)+2] = 1
            training_data_sequences[i][2][int(110-i*10)+4] = 1
            training_data_sequences[i][3][int(110-i*10)+5] = 1
            training_data_sequences[i][4][int(110-i*10)+7] = 1
            training_data_sequences[i][5][int(110-i*10)+9] = 1
            training_data_sequences[i][6][int(110-i*10)+11] = 1
            training_data_sequences[i][7][int(110-i*10)+12] = 1
        return training_seq_lengths, training_data_sequences
    ## ドドド、レレレ
    def superEasyTones():
        training_seq_lengths = torch.tensor([8]*10)
        training_data_sequences = torch.zeros(10,8,88)
        for i in range(10):
            for j in range(8):
                training_data_sequences[i][j][int(30+i*5)] = 1
        return training_seq_lengths, training_data_sequences
    ## ドドド、ドドド、ドドド
    def easiestTones():
        training_seq_lengths = torch.tensor([8]*10)
        training_data_sequences = torch.zeros(10,8,88)
        for i in range(10):
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


    ## 長さ最長129、例えば長さが60のやつは61~129はすべて0データ
    data = poly.load_data(poly.JSB_CHORALES)
    training_seq_lengths = data['train']['sequence_lengths']
    training_data_sequences = data['train']['sequences']

    if eas:
        # ## ドドド、レレレ、ミミミ、ドレミ
        training_seq_lengths, training_data_sequences = easyTones()
    
    if sup:
        # ドドド、レレレ
        training_seq_lengths, training_data_sequences = superEasyTones()

    if est:
        ## ドドドのみ
        training_seq_lengths, training_data_sequences = easiestTones()


    test_seq_lengths = data['test']['sequence_lengths']
    test_data_sequences = data['test']['sequences']
    val_seq_lengths = data['valid']['sequence_lengths']
    val_data_sequences = data['valid']['sequences']
    N_train_data = len(training_seq_lengths)
    N_train_time_slices = float(torch.sum(training_seq_lengths))
    N_mini_batches = int(N_train_data / mini_batch_size +
                        int(N_train_data % mini_batch_size > 0))


    # how often we do validation/test evaluation during training
    val_test_frequency = 50
    # the number of samples we use to do the evaluation
    n_eval_samples = 1

    # get the validation/test data ready for the dmm: pack into sequences, etc.
    val_seq_lengths = rep(val_seq_lengths)
    test_seq_lengths = rep(test_seq_lengths)
    val_batch, val_batch_reversed, val_batch_mask, val_seq_lengths = poly.get_mini_batch(
        torch.arange(n_eval_samples * val_data_sequences.shape[0]), rep(val_data_sequences),
        val_seq_lengths, cuda=cuda)
    test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths = poly.get_mini_batch(
        torch.arange(n_eval_samples * test_data_sequences.shape[0]), rep(test_data_sequences),
        test_seq_lengths, cuda=cuda)



    dmm = DMM(use_cuda=cuda, rnn_check=rck, rpt=rpt)
    WN = WGAN_network()
    W = WassersteinLoss(WN, N_loops=N_loops, lr=w_learning_rate)

    # Create optimizer algorithm
    # optimizer = optim.SGD(dmm.parameters(), lr=learning_rate)
    optimizer = optim.Adam(dmm.parameters(), lr=learning_rate)
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9999) #1でもやってみる

    # make directory for Data
    # now = datetime.datetime.now().strftime('%Y%m%d_%H_%M')
    # os.makedirs(os.path.join("saveData", now), exist_ok=True)

    # save args as TEXT
    # f = open(os.path.join("saveData", now, 'args.txt'), 'w') # 書き込みモードで開く
    # for name, site in vars(args).items():
    #     f.write(name +" = ") 
    #     f.write(str(site) + "\n")
    # f.close() # ファイルを閉じる

    #######################
    #### TRAINING LOOP ####
    #######################
    for epoch in range(num_epochs):
        epoch_nll = 0
        shuffled_indices = torch.randperm(N_train_data)
        # print("Proceeding: %.2f " % (epoch*100/5000) + "%")

        # process each mini-batch; this is where we take gradient steps
        for which_mini_batch in range(N_mini_batches):

            # compute which sequences in the training set we should grab
            mini_batch_start = (which_mini_batch * mini_batch_size)
            mini_batch_end = np.min([(which_mini_batch + 1) * mini_batch_size, N_train_data])
            mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]

            # grab a fully prepped mini-batch using the helper function in the data loader
            mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
                = poly.get_mini_batch(mini_batch_indices, training_data_sequences,
                                        training_seq_lengths, cuda=cuda)

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
            loss = W.calc(mini_batch, generated_mini_batch)
            # NaN_detect(WN, epoch, message="calc_After")

            # NaN_detect(dmm, epoch, message="step_Before")        
            # do an actual gradient step
            loss.backward()
            optimizer.step()
            scheduler.step()
            if if_rnn_clip != None:
                for p in dmm.rnn.parameters():
                    p.data.clamp_(-rnn_clip, rnn_clip)
                    # p.data.clamp_(-0.01, 0.01)
            # NaN_detect(dmm, epoch, message="step_After")        

            epoch_nll += loss
        
    return epoch_nll


def main(args):
    study = optuna.create_study()
    study.optimize(objective, n_trials=args.epochs)

    # make directory for Data
    now = datetime.datetime.now().strftime('%Y%m%d_%H_%M')
    os.makedirs(os.path.join("saveOptuna", now), exist_ok=True)

    # save args as TEXT
    f = open(os.path.join("saveOptuna", now, 'Optuna.txt'), 'w') # 書き込みモードで開く
    for name, site in study.best_params.items():
        f.write(name +" = ") 
        f.write(str(site) + "\n")
    f.write("BEAT VALUE = ")
    f.write(str(study.best_value))
    f.close() # ファイルを閉じる
    



# parse command-line arguments and execute the main method
if __name__ == '__main__':
    # assert pyro.__version__.startswith('1.5.1')

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--epochs', type=int, default=5000)
    args = parser.parse_args()

    main(args)



