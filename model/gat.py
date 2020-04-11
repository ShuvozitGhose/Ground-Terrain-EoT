import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class GraphAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W) # shape: B*N*out_features
        N = h.size()[1]

        f_1 = torch.matmul(h, self.a1).expand(h.shape[0], N, N) # shape: B*N*N
        f_2 = torch.matmul(h, self.a2).expand(h.shape[0], N, N) # shape: B*N*N
        e = self.leakyrelu(f_1 + f_2.transpose(1,2)) # shape: B*N*N

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class IntraDomain(nn.Module):
    def __init__(self, n_nodes, feature_dim, out_dim, use_gpu=True, hidden_dim=512, n_heads=3, alpha=0.2, dropout=0.6):
        """
            cuda, hidden_dim, n_heads, alpha, dropout can be made default
        """
        super(IntraDomain, self).__init__()
        self.adj = self.get_adj(n_nodes)
        if use_gpu:
            self.adj = self.adj.cuda()
        self.model = GAT(
            nfeat=feature_dim,
            nhid=hidden_dim,
            nclass=out_dim,
            dropout=dropout,
            nheads=n_heads,
            alpha=alpha)

    def get_adj(self, nodes): # Create the adjacency matrix
        return torch.ones((nodes, nodes))

    def forward(self, x):
        output = self.model(x, self.adj)
        return output


class InterDomain(nn.Module):
    def __init__(self, Nx, Ny, Dx, Dy):
        """
            Nx: number of nodes in domain_1
            Dx: number of feature maps in domain_1
            Ny: number of nodes in domain_2
            Dy: number of feature maps in domain_2
        """
        super(InterDomain, self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.Dx = Dx
        self.Dy = Dy
        
        # Wx shape: (Dx+Dy)*Dx
        self.Wx = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self.Dx+self.Dy, self.Dx).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        # Wx shape: (Dx+Dy)*Dy
        self.Wy = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self.Dx+self.Dy, self.Dy).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)


    def __get_cosine():
        pass

    
    def forward(self, x, y, x_cos, y_cos):
        """
            x: B*16*512
            y: B*16*512
            x_cos: cosine values for each 'x' with dimension B*16*1
            y_cos: cosine_values for each 'y' with dimensino B*16*1
        """
        if len(x_cos.shape)<3:
            x_cos = x_cos.reshape(x_cos.shape[0], x_cos.shape[1], 1)
            warnings.warn("please enter x_cos shape as (B*16*1) and NOT as (B*16)")
        if len(y_cos.shape)<3:
            y_cos = y_cos.reshape(y_cos.shape[0], y_cos.shape[1], 1)
            warnings.warn("please enter y_cos shape as (B*16*1) and NOT as (B*16)")

        x_mul_cos = torch.mul(x, x_cos.expand(x_cos.shape[0], self.Nx, self.Dx)) # shape: B*Nx*Dx
        y_mul_cos = torch.mul(y, y_cos.expand(y_cos.shape[0], self.Ny, self.Dy)) # shape: B*Ny*Dy

        # summation along dim=1 to convert: (B*Nx*Dx) => (B*Dx)
        x_mul_cos = x_mul_cos.sum(dim=1)
        y_mul_cos = y_mul_cos.sum(dim=1)

        updated_x = torch.cat([x, (y_mul_cos.reshape(-1, 1, self.Dy)).expand(-1, self.Nx, self.Dy)], dim=-1) # shape: B*Nx*(Dx+Dy)
        updated_y = torch.cat([y, (x_mul_cos.reshape(-1, 1, self.Dx)).expand(-1, self.Ny, self.Dx)], dim=-1) # shape: B*Ny*(Dx+Dy)

        final_x = torch.matmul(updated_x, self.Wx) # shape: Nx*Dx
        final_y = torch.matmul(updated_y, self.Wy) # shape: Ny*Dy

        return final_x, final_y


if __name__ == "__main__":
    print ("testing modules using dummy data only...")
    use_gpu = torch.cuda.is_available()

    def __inter_testing():
        print ("testing inter domain...")
        B, Nx, Ny, Dx, Dy = 1, 2, 2, 3, 3
        x = torch.randn(B, Nx, Dx)
        x_cos = torch.randn(B, Nx, 1)
        print ("domain_1: feature shape: {}; cosine_shape: {}".format(x.shape, x_cos.shape))
        
        y = torch.randn(B, Ny, Dy)
        y_cos = torch.randn(B, Ny, 1)
        print ("domain_2: feature shape: {}; cosine_shape: {}".format(y.shape, y_cos.shape))
        
        if use_gpu:
            print ("found gpu, hence using it...")
            x = x.cuda()
            y = y.cuda()
            x_cos = x_cos.cuda()
            y_cos = y_cos.cuda()

        interdomain = InterDomain(Nx, Ny, Dx, Dy)

        output_x, output_y = interdomain(x, y, x_cos, y_cos)

        print ("output shape of X: {}; output shape of y: {}".format(output_x.shape, output_y.shape))

    def __intra_testing():
        print ("testing intra domain...")
        B, N, D = 1, 2, 3
        x = torch.randn(B, N, D)
        if use_gpu:
            print ("gpu found and hence using")
            x = x.cuda()

        intradomain = IntraDomain(N, D, D)

        output = intradomain(x)
        print ("output shape of x:{}".format(output.shape))

    __intra_testing()
    __inter_testing()