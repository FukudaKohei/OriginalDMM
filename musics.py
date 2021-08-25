import numpy as np
from numpy.core.numerictypes import ScalarType
from numpy.lib.index_tricks import diag_indices_from
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation



## ドレミ
def doremifasorasido(N_songs):
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
def dododorerere(N_songs):
    training_seq_lengths = torch.tensor([8]*N_songs)
    training_data_sequences = torch.zeros(N_songs,8,88)
    for i in range(N_songs):
        for j in range(8):
            training_data_sequences[i][j][int(30+i*5)] = 1
    return training_seq_lengths, training_data_sequences

## ドドド、ドドド、ドドド
def dodododododo(N_songs):
    training_seq_lengths = torch.tensor([8]*N_songs)
    training_data_sequences = torch.zeros(N_songs,8,88)
    for i in range(N_songs):
        for j in range(8):
            training_data_sequences[i][j][int(70)] = 1
    return training_seq_lengths, training_data_sequences

def multi_normal_prob(loc,scale,x):
    # locは平均, scaleは共分散行列, xはサンプルとする。
    pi_2_d = torch.tensor(np.sqrt((np.pi * 2) ** loc.size(-1)))
    det = torch.sum(scale, dim=2)
    mom = torch.sqrt(det) * pi_2_d
    exp_arg = -0.5 * torch.sum((x-loc) * (x-loc) / scale, dim=2)
    print(mom[0])
    print(exp_arg[0])
    print(torch.exp(exp_arg)[0])
    return torch.exp(exp_arg) / mom

def KL_divergence(x_prob, y_prob):
    return (torch.t(x_prob) * ((torch.t(x_prob) / torch.t(y_prob))).log()).sum()

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
    KL = 0.5 *(torch.log(det_q_scale) - torch.log(det_p_scale) - dim + trace + niji)
    return KL.sum() / (beats * songs)

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

def createSin_phaseChanged(N_data, length):
    constant = torch.zores(N_data)
    phase = torch.rand(N_data) * 2*np.pi
    amplitude = torch.ones(N_data)
    frequency = torch.ones(N_data) * 2*np.pi
    interval = frequency / length
    return torch.tensor([[[amplitude[i] * torch.sin(interval[i] * j + phase[i]) + constant[i] ] for j in range(length)] for i in range(N_data)])

def createSin_constantChanged(N_data, length):
    constant = torch.randn(N_data)
    phase = torch.zeros(N_data)
    amplitude = torch.ones(N_data)
    frequency = torch.ones(N_data) * 2*np.pi
    interval = frequency / length
    return torch.tensor([[[amplitude[i] * torch.sin(interval[i] * j + phase[i]) + constant[i] ] for j in range(length)] for i in range(N_data)])

def createSin_amplitudeChanged(N_data, length):
    constant = torch.zeros(N_data)
    phase = torch.zeros(N_data)
    amplitude = torch.rand(N_data) * 3
    frequency = torch.ones(N_data) * 2*np.pi
    interval = frequency / length
    return torch.tensor([[[amplitude[i] * torch.sin(interval[i] * j + phase[i]) + constant[i] ] for j in range(length)] for i in range(N_data)])

def createSin_frequencyChanged(N_data, length):
    constant = torch.zeros(N_data)
    phase = torch.zeros(N_data)
    amplitude = torch.ones(N_data)
    frequency = torch.rand(N_data) * 6*np.pi
    interval = frequency / length
    return torch.tensor([[[amplitude[i] * torch.sin(interval[i] * j + phase[i]) + constant[i] ] for j in range(length)] for i in range(N_data)])

def createSin_allChanged(N_data, length):
    # constant = torch.randn(N_data)
    constant = torch.zeros(N_data)
    # phase = torch.rand(N_data) * 2*np.pi
    phase = torch.zeros(N_data)
    # amplitude = torch.rand(N_data) * 3
    amplitude = torch.ones(N_data) * 2
    frequency = torch.rand(N_data) * 6*np.pi
    interval = frequency / length
    return torch.tensor([[[amplitude[i] * torch.sin(interval[i] * j + phase[i]) + constant[i] ] for j in range(length)] for i in range(N_data)])



# a = torch.tensor([ i for i in range(10)])
# # print(createSin_phaseChanged(20,8).size())
# N = 2
# length =100
# data = createSin_allChanged(N,length)
# x = np.linspace(0, 2*np.pi, length)
# plt.plot(x, data[0])
# plt.savefig("sin_curve")

# xaxis_Max = 4*np.pi
# interval = 100

# fig = plt.figure(1)
# ax = fig.add_subplot(111)
# plt.xlim(0,xaxis_Max)
# plt.ylim(-1.2,1.2)
# plt.grid()
# x = np.linspace(0,xaxis_Max,interval)

# line, = ax.plot(x[0], np.sin(x[0]), "r")
# dot, = ax.plot(x[0], np.sin(x[0]), ".", color="red")

# def update(i):
#     line.set_data(x[:i+1], np.sin(x[:i+1]+1))
#     dot.set_data(x[i], np.sin(x[i]+1))


# ims = []
# for i in range(interval):
#     y = np.sin(x[:i+1])
#     line, = plt.plot(x[:i],y[:i],"r")
#     ims.append([line])
#     line, = plt.plot(x[i],y[i],".")
#     ims.append([line])

# ani = animation.ArtistAnimation(fig, ims)

# ani = animation.FuncAnimation(fig, update, interval, blit=False, interval=10, repeat=False)
# ani.save('anim.gif', writer="pillow")

# N = 20
# dim = 88
# x_loc = torch.zeros(N,8,dim)
# y_loc = torch.zeros(N,8,dim)
# x_scale = torch.ones(N,8,dim)
# y_scale = torch.ones(N,8,dim)

# print(decoder(x_loc).size())

# x_normal = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
# y_normal = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
# x = x_normal.rsample((N,8))
# y = y_normal.rsample((N,8))
# x_prob = multi_normal_prob(x_loc,x_scale,x)
# # y_prob = multi_normal_prob(x_loc,y_scale,y)
# print(x_prob)
# print(torch.tensor(1e-60))
# print(y_prob)
# print(KL_divergence(x_prob, y_prob))
# print(x)