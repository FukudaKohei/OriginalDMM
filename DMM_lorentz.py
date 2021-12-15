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
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, z_t):
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        if self.input_dim == 1:
            x = self.lin_hidden_to_input(h2)
        else:
            # ps = torch.sigmoid(self.lin_hidden_to_input(h2))
            # if self.use_cuda: 
            #     eps = torch.rand(self.input_dim).cuda()
            # else : eps = torch.rand(self.input_dim)
            # appxm = torch.log(eps + 1e-20) - torch.log(1-eps + 1e-20) + torch.log(ps + 1e-20) - torch.log(1-ps + 1e-20)
            # x = torch.sigmoid(appxm)
            x = self.lin_hidden_to_input(h2)
        return x

class GatedTransition(nn.Module):
    def __init__(self, z_dim, transition_dim):
        super().__init__()
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        return loc, scale

class Combiner(nn.Module):
    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        loc = self.lin_hidden_to_loc(h_combined)
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        return loc, scale

class Encoder(nn.Module):
    def __init__(self, input_dim=88, z_dim=100,rnn_dim=60, num_layers=1, rnn_dropout_rate=0.1,num_iafs=0, iaf_dim=50, N_z0 = 10, use_cuda=False, rnn_check=False):
        super().__init__()
        self.combiner = Combiner(z_dim, rnn_dim)
        rnn_dropout_rate = 0. if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                          batch_first=True, bidirectional=False, num_layers=num_layers,
                          dropout=rnn_dropout_rate)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.randn(1, 1, rnn_dim))
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()
        self.rnn_check = rnn_check

    def forward(self, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, annealing_factor=1.0):
        T_max = mini_batch.size(1)
        h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))
        z_container = []
        z_loc_container = []
        z_scale_container = []
        for t in range(1,T_max+1):
            z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
            if args.clip != None:
                z_scale = torch.clamp(z_scale, min = args.clip)
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

class Prior(nn.Module):
    def __init__(self, z_dim=100, transition_dim=200,  N_z0 = 10, use_cuda=False):
        super().__init__()
        self.trans = GatedTransition(z_dim, transition_dim)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        self.use_cuda = use_cuda

    def forward(self, length, N_generate):
        T_max = length
        z_prev = self.z_q_0.expand(N_generate, self.z_q_0.size(0))
        z_container = []
        z_loc_container = []
        z_scale_container = []
        for t in range(1,T_max+1):
            z_loc, z_scale = self.trans(z_prev)
            if args.clip != None:
                z_scale = torch.clamp(z_scale, min = args.clip)
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
    pi_2_d = torch.tensor(np.sqrt((np.pi * 2) ** loc.size(-1)))
    det = torch.sum(scale, dim=2)
    mom = torch.sqrt(det) * pi_2_d
    exp_arg = -0.5 * torch.sum((x-loc) * (x-loc) / scale, dim=2)
    return torch.exp(exp_arg) / mom

def D_KL(p_loc, q_loc, p_scale, q_scale):
    det_p_scale = torch.prod(p_scale, dim=2)
    det_q_scale = torch.prod(q_scale, dim=2)
    dim = p_loc.size(-1)
    beats = p_loc.size(1)
    songs = p_loc.size(0)
    trace = torch.sum(p_scale/q_scale, dim = 2)
    niji = torch.sum((q_loc-p_loc)*(q_loc-p_loc)/q_scale, dim=2)
    KL = 0.5 *(torch.log(det_q_scale) - torch.log(det_p_scale) - dim + trace + niji)
    return KL.sum() / (beats * songs)

def D_Wass(p_loc, q_loc, p_scale, q_scale):
    beats = p_loc.size(1)
    songs = p_loc.size(0)
    norm_2 = torch.sum((p_loc - q_loc)*(p_loc - q_loc), dim=2)
    trace_p_scale = torch.sum(p_scale, dim=2)
    trace_q_scale = torch.sum(q_scale, dim=2)
    trace = torch.sum(torch.sqrt(p_scale*q_scale), dim=2)
    Wass = torch.sqrt(norm_2 + trace_p_scale + trace_q_scale - 2 * trace)
    return Wass.sum() / (beats * songs)

def multi_normal_prob(loc,scale,x):
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
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(-25,25)
        ax.set_ylim(-25,25)
        ax.set_zlim(0,50)
        plt.plot(train_data[:,0], train_data[:,1], train_data[:,2] , label="Training data")
        plt.plot(recon_data.detach().numpy()[:,0], recon_data.detach().numpy()[:,1], recon_data.detach().numpy()[:,2] , label="Reconstructed data")
        plt.title("Lorentz Curves")
        # plt.ylim(top=10, bottom=-10)
        # plt.xlabel("time", fontsize=FS)
        # plt.ylabel("y", fontsize=FS)
        plt.legend()
        fig.savefig(os.path.join(path, "Reconstruction"+str(number)+".png"))
        plt.close()

    def saveGeneSinGraph(gene_data, length, path, number):
        FS = 10
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(-25,25)
        ax.set_ylim(-25,25)
        ax.set_zlim(0,50)
        # plt.rcParams["font.size"] = FS
        plt.plot(gene_data.detach().numpy()[:,0], gene_data.detach().numpy()[:,1], gene_data.detach().numpy()[:,2] , label="Generated data")
        plt.title("Lorentz Curves")
        # plt.ylim(top=10, bottom=-10)
        # plt.xlabel("time", fontsize=FS)
        # plt.ylabel("y", fontsize=FS)
        plt.legend()
        fig.savefig(os.path.join(path, "Generation"+str(number)+".png"))
        plt.close()

    FS = 10
    plt.rcParams["font.size"] = FS

    ## 長さ最長129、例えば長さが60のやつは61~129はすべて0データ
    data = poly.load_data(poly.JSB_CHORALES)
    training_seq_lengths = data['train']['sequence_lengths'][:args.N_songs]
    training_data_sequences = data['train']['sequences'][:args.N_songs,:8]
    training_seq_lengths = torch.tensor([8]*args.N_songs)

    traing_data = []
    for i in range(args.N_songs):
        traing_data.append(musics.lorentz(torch.randn(3), args.length, args.T))
    training_data_sequences = torch.stack(traing_data)
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
    optimizer = optim.SGD(params, lr=args.learning_rate)
    # optimizer = optim.Adam(params, lr=args.learning_rate)
    # Add learning rate scheduler
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.999) 
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9999) 
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 1.) 
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
            for i in range(len(mini_batch)):
                saveReconSinGraph(mini_batch[i], reconed_x[i], args.length, path, i)
                saveGeneSinGraph(decoder(pri_z)[i], args.length, path, i)

        if epoch == args.num_epochs-1:
            path = os.path.join("saveData", now, "Epoch%d"%args.num_epochs)
            os.makedirs(path, exist_ok=True)
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
    parser.add_argument('-length', '--length', type=int, default=1000)
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



