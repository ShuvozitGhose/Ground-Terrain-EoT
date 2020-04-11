import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import encoding
import torchvision.models as resnet
from torch_geometric.nn import ChebConv

import torch.nn.functional as F

def loader(idx_features_labels):
######## Yet to be tested ########
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    edges = np.array()  ### to add
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return features , adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)





class Net(nn.Module):
    def __init__(self, nclass, backbone='resnet18'):
        super(Net, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=True)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=True)
        elif backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=True)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        n_codes = 8


        self.head = nn.Sequential(
            #nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            #nn.ReLU(inplace=True),
            encoding.nn.Encoding(D=512,K=n_codes),
            encoding.nn.View(-1, 512*n_codes),
            encoding.nn.Normalize(),
            nn.Linear(512*n_codes, 64),
            #nn.BatchNorm1d(64),
        )
        '''
        self.head_1 = nn.BatchNorm2d(512)
        self.head_2 = encoding.nn.Encoding(D=512,K=n_codes)
        self.head_3 = encoding.nn.View(-1, 512*n_codes)
        self.head_4 = encoding.nn.Normalize()
        self.head_5 = nn.Linear(512*n_codes, 64)
        '''
        self.pool = nn.Sequential(
            nn.AvgPool2d(7),
            encoding.nn.View(-1, 512),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
        )

        self.lstm = nn.LSTM(128, 1024, num_layers=2, bidirectional=True)

        self.fc_layer = nn.Sequential(
            encoding.nn.Normalize(),
            nn.Linear(2048, 128),
            encoding.nn.Normalize(),
            nn.Linear(128, nclass))

        self.conv1 = ChebConv(data.num_features, 256, K=2)
        self.conv2 = ChebConv(256, 32, K=2)
        self.conv3 = ChebConv(32, data.num_features, K=2)


    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            var_input = x
            while not isinstance(var_input, Variable):
                var_input = var_input[0]
            _, _, h, w = var_input.size()
        else:
            raise RuntimeError('unknown input type: ', type(x))

        if self.backbone == 'resnet18' or self.backbone == 'resnet50' or self.backbone == 'resnet101' \
            or self.backbone == 'resnet152':
            # pre-trained ResNet feature
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            x = self.pretrained.layer1(x)
            x = self.pretrained.layer2(x)
            x = self.pretrained.layer3(x)
            x = self.pretrained.layer4(x)

            extract_patches = x.unfold(2, 3, 1).unfold(3, 3, 1)  # 64, 512, 6, 6, 3, 3
            patches = extract_patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(x.shape[0], -1, 512, 3, 3)  # 64, 36, 512, 3, 3
            #patch_pool = patches.contiguous().view(64, patches.shape[1], 512, -1).mean(3)  # 64, 36, 512, 9 =>  64, 36, 512

            patches_texture = []
            patches_shape = []
            for i in range(patches.shape[1]):
                patches_texture.append(self.head(patches[:,i,:,:,:]))
                patches_shape.append(self.pool(patches[:,i,:,:,:]))

            patches_texture = torch.stack(patches_texture, 1)
            patches_shape = torch.stack(patches_shape, 1)

            LSTM_input = torch.cat((patches_texture, patches_shape), dim=2).permute(1,0,2)
            features , adj = data() ###### to do 

            print(LSTM_input.size())
            output, _ = self.lstm(LSTM_input)

            x = self.fc_layer(output[-1,:,:])

        else:
            x = self.pretrained(x)
        return x


def test():
    net = Net(nclass=23).cuda()
    print(net)
    x = Variable(torch.randn(1,3,224,224)).cuda()
    y = net(x)
    print(y)
    params = net.parameters()
    sum = 0
    for param in params:
        sum += param.nelement()
    print('Total params:', sum)


if __name__ == "__main__":
    test()
