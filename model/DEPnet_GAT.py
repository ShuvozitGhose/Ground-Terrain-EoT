import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import encoding
import torchvision.models as resnet

import torch.nn.functional as F

from model.gat import IntraDomain, InterDomain

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

        self.fc_layer = nn.Sequential(
            encoding.nn.Normalize(),
            # nn.Linear(2048, 128),
            nn.Linear(128, 128),
            encoding.nn.Normalize(),
            nn.Linear(128, nclass))

        self.interdomain_1 = InterDomain(Nx=36, Ny=36, Dx=64, Dy=64)
        self.intradomain_texture_1 = IntraDomain(n_nodes=36, feature_dim=64, out_dim=64)
        self.intradomain_shape_1 = IntraDomain(n_nodes=36, feature_dim=64, out_dim=64)

        self.interdomain_2 = InterDomain(Nx=36, Ny=36, Dx=64, Dy=64)
        self.intradomain_texture_2 = IntraDomain(n_nodes=36, feature_dim=64, out_dim=64)
        self.intradomain_shape_2 = IntraDomain(n_nodes=36, feature_dim=64, out_dim=64)
        
        self.W_feature = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(36, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.W_shape = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(36, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)



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
            patch_pool = patches.contiguous().view(patches.shape[0], patches.shape[1], 512, -1).mean(3)  # 64, 36, 512, 9 =>  64, 36, 512
            sub_patch_pool = patches[:,:,:, 1:2, 1:2].squeeze(-1).squeeze(-1) # 64x36x512
            cos_sim_x = F.cosine_similarity(patch_pool, sub_patch_pool, 2)/0.5 - 1 # =.  Extent of Texture Information [0 -1] Normalizeds
            cos_sim_y = 1. - cos_sim_x # - Extent of Shape Information 

            patches_texture = []
            patches_shape = []
            for i in range(patches.shape[1]):
                patches_texture.append(self.head(patches[:,i,:,:,:]))
                patches_shape.append(self.pool(patches[:,i,:,:,:]))

            patches_texture = torch.stack(patches_texture, 1)
            patches_shape = torch.stack(patches_shape, 1)

            # GAT networks
            patches_texture = self.intradomain_texture_1(patches_texture)
            patches_shape = self.intradomain_shape_1(patches_shape)
            patches_texture, patches_shape = self.interdomain_1(x=patches_texture, y=patches_shape, x_cos=cos_sim_x, y_cos=cos_sim_y)
            
            patches_texture = self.intradomain_texture_2(patches_texture)
            patches_shape = self.intradomain_shape_2(patches_shape)
            patches_texture, patches_shape = self.interdomain_2(x=patches_texture, y=patches_shape, x_cos=cos_sim_x, y_cos=cos_sim_y)

            texture_vector = torch.matmul(patches_texture.permute(0,2,1), self.W_feature).squeeze()
            shape_vector = torch.matmul(patches_shape.permute(0,2,1), self.W_shape).squeeze()

            output = torch.cat([texture_vector, shape_vector], dim=-1)

            x = self.fc_layer(output)

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
