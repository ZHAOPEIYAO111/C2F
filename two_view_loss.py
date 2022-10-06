import torch
from torch import nn
import math
import torch
from torch import nn
import torch.nn.functional as F

eps = 1e-12



class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False, dataset=None): # 128,13752
        super(NCEAverage, self).__init__()
        self.nLem = outputSize  # the number of nodes  13752
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        # self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax
        self.dataset = dataset
        self.batch_norm = nn.BatchNorm1d(num_features=inputSize)   #  latent space dimension 128

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum, -1, -1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))# 矩阵  #缓冲区不更新 # 分别都跟同一个负样队列相乘
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, ab, l,  ss, y, idx=None): # l, ab是GCN出来之后的节点的表示，Z，的两个view表示 # l=0.7, ab=0.7  ss=0.4
        K = int(self.params[0].item())         # (0.8, 1, 0.7)
        T = self.params[1].item()
        Z_l = self.params[2].item() # Z_l 是batch所有节点的表示
        Z_orig = self.params[3].item()
        Z_ss = self.params[5].item()


        momentum = self.params[4].item()
        batchSize = l.size(0) #batchsize 就是有节点总数 amazon_computer n_nodes:13752

        outputSize = self.memory_l.size(0) # the latent features of all nodes are stored in memory
        inputSize = self.memory_l.size(1) # inputsize:hidden=128, outputsize:n_nodes=13752

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        #print("l", l)  # [13752, 128]
        #print(y.shape) # 13752 节点个数
        #print("idx.shape",idx.shape) # [13752, 1025]
        # sample
        ############################################################################################
        ##############weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()  ###不懂了

        ##############weight_l = weight_l.view(batchSize, K + 1, inputSize) # [13752, 1025, 128]


        # inner product distance
        #print(ab.shape) # [13752, 128]
        ###########out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1)) # [13752, 1025, 1] # 负样是memory_l
        #print(out_ab.shape)

        # sample
        #############################################################################################
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize) # batchsize 有多少个节点，inputsize有多少维度

        # Inner product distance
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))

        out_ss = torch.bmm(weight_ab, ss.view(batchSize, inputSize, 1))

        out_orig = torch.bmm(weight_ab, ab.view(batchSize, inputSize, 1))




        if self.use_softmax:
            #out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            out_ss = torch.div(out_ss, T)

            out_orig = torch.div(out_orig, T)
            out_l = out_l.contiguous()
            #out_ab = out_ab.contiguous()
            out_ss = out_ss.contiguous()
            out_orig = out_orig.contiguous()

        else:
            #out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            out_ss = torch.exp(torch.div(out_ss, T))
            out_orig = torch.exp(torch.div(out_orig, T))

            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))

           # if Z_ab < 0:
            #    self.params[3] = out_ab.mean() * outputSize
            #    Z_ab = self.params[3].clone().detach().item()
            #    print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            if Z_orig < 0:
                self.params[3] = out_orig.mean() * outputSize
                Z_orig = self.params[3].clone().detach().item()
                print("normalization constant Z_orig is set to {:.1f}".format(Z_orig))

            if Z_ss < 0:
                self.params[5] = out_ss.mean() * outputSize
                Z_ss = self.params[5].clone().detach().item()
                print("normalization constant Z_ss is set to {:.1f}".format(Z_ss))


            # compute out_l, out_ab
            out_l = torch.div(out_l, Z_l).contiguous()
            out_orig = torch.div(out_orig, Z_orig).contiguous()  # 除以z就是标准化
            out_ss = torch.div(out_ss, Z_ss).contiguous()


        # # update memory
        with torch.no_grad():
            '''
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)
            '''

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))

            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)
        #print("out_l",out_l.shape)  # [13752, 1025, 1]
        #print("out_ab", out_ab.shape)  # [13752, 1025, 1]
        #print("out_ss", out_ss.shape)

        return out_orig, out_l, out_ss #(1, 0.8, 0.7)

class NCESoftmaxLossranking(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLossranking, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y): # 输入的X是[n,m]的矩阵----[13752, 1025]

        x = x.squeeze()
        y = y.squeeze()

        X = torch.cat([x,y],dim=1).cuda()
        Y = torch.cat([y,x],dim=1).cuda()


        bsz = X.shape[0]
        label_x = torch.zeros([bsz]).cuda().long()
        label_y = torch.zeros([bsz]).cuda().long() # 第一列是 positive pair similarity


        loss_x = self.criterion(X, label_x)
        loss_y = self.criterion(Y, label_y)

        return loss_x, loss_y

class self_rankingloss(nn.Module):

    def __init__(self):
        super(self_rankingloss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z, x, y): # z is original

        x = x.squeeze()
        y = y.squeeze()
        z = z.squeeze() # [1917,1025]


        X = torch.cat([x,y],dim=1).cuda()
        Z = torch.cat([z,z],dim=1).cuda()
        p_orig = nn.functional.softmax(Z, dim=-1)
        log_p_view = nn.functional.log_softmax(X, dim=-1)
        nll = -1.0 * torch.einsum('nk,nk->n', [p_orig, log_p_view])
        loss_alig = torch.mean(nll)

        return loss_alig


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x): # 输入的X是[n,m]的矩阵----[13752, 1025]
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class divergenceloss(nn.Module):

    def __init__(self):
        super(divergenceloss, self).__init__()
    def forward(self, x, y):# out_l, out_ss
        x = x.squeeze(-1)
        y = y.squeeze(-1)

        p_weak = nn.functional.softmax(x, dim=-1)
        log_p_s = nn.functional.log_softmax(y, dim=-1)
        nll = -1.0 * torch.einsum('nk,nk->n', [p_weak, log_p_s])
        loss_ddm = torch.mean(nll)

        return loss_ddm


class NCECriterion(nn.Module):
    """
    Eq. (12): L_{NCE}
    """

    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = (torch.div(P_pos, P_pos.add(m * Pn + eps)) + eps).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = (torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)) + eps).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss




class softmaxxx(nn.Module):
    def __init__(self):
        super(softmaxxx, self).__init__()
    def forward(self, theta):
        tt = torch.exp(theta)
        y = tt/torch.sum(tt)
        return y











class AliasMethod(object):
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """
        Draw N samples from multinomial
        :param N: number of samples
        :return: samples
        """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj


'''
import torch.nn.functional as F
criterion = nn.CrossEntropyLoss()
x = torch.randn(4,5)
print("x", x)
bsz = x.shape[0] #4
target = torch.zeros([bsz]).long()
label = torch.tensor([0,0,0,0])
print(label)
one_hot = F.one_hot(label).float()
print("one_hot",one_hot)
softmax = torch.exp(x)/torch.sum(torch.exp(x), dim=0)
logsoftmax = torch.log(softmax)
print("logsoftmax",logsoftmax)
nllloss =  one_hot * logsoftmax
print("nllloss", nllloss)
result = torch.sum(nllloss)/4
print(result)
labels = torch.zeros(4, dtype=torch.long)
print(labels)
resultt = criterion(x, labels)
print(resultt)


x = torch.randn(5,5)
softmax = torch.exp(x)/torch.sum(torch.exp(x), dim = 1).reshape(-1, 1)
print(softmax)



criterion = nn.CrossEntropyLoss()


x=torch.randn(5,5)
bsz = x.shape[0]
print(bsz) #5
x = x.squeeze()
print(x)
label = torch.zeros([bsz]).long()
print(label)
loss = criterion(x, label)
print(loss)

import torch.nn.functional as F
one = F.one_hot(label).float()
print(one)

y = torch.LongTensor(range(10))
print(y.data)
batchSize = 10
K = 5
nLem = 10
unigrams = torch.ones(nLem)
print(unigrams)
multinomial = AliasMethod(unigrams)
print(multinomial)
idx = multinomial.draw(batchSize * (K + 1)).view(batchSize, -1)
print(multinomial.draw(batchSize * (K + 1)))
print(idx)
print(idx.select(1,0))
a = idx.select(1, 0).copy_(y.data)
print(a)
'''



