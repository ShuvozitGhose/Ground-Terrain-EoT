import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import encoding
import torchvision.models as resnet

import torch.nn.functional as F

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

        ''' 
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

        self.pool = nn.Sequential(
            nn.AvgPool2d(7),
            encoding.nn.View(-1, 512),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
        )

        self.fc = nn.Sequential(
            encoding.nn.Normalize(),
            nn.Linear(64*64, 128),
            encoding.nn.Normalize(),
            nn.Linear(128, nclass))

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
            x = self.pretrained.layer4(x) # 64, 512, 8, 8

            cos_sim , cos_sim_2, cos_sim_3 = self.texture_extent(x)

            #x1 = self.head(x)
            x1 = self.head_1(x)
            x1 = self.head_2(x1)
            x1 = self.head_3(x1)
            x1 = self.head_4(x1)
            x1 = self.head_5(x1)

            x2 = self.pool(x)
            x1 = x1.unsqueeze(1).expand(x1.size(0),x2.size(1),x1.size(-1))
            x = x1*x2.unsqueeze(-1)
            x = x.view(-1,x1.size(-1)*x2.size(1))
            x = self.fc(x), cos_sim, cos_sim_2 , cos_sim_3
        else:
            x = self.pretrained(x)
        return x

    def texture_extent(self, featuremap):
        #featuremap 64, 512, 8, 8
        extract_patches = featuremap.unfold(2, 4, 2).unfold(3, 4, 2)  # 64, 512, 3, 3, 4, 4
        patches = extract_patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(64, -1, 512, 4, 4)  # 64, 9, 512, 4, 4
        patch_pool = patches.contiguous().view(64, 9, 512, -1).mean(3) # 64, 9, 512, 16 =>  64, 9, 512

        subpatches = patches[:, :, :, 1:3, 1:3]
        subpatches_pool = subpatches.contiguous().view(64, 9, 512, -1).mean(3) # 64, 9, 512

        #patches[:, :, :, 1:3, 1:3] = 0.
        #patch_pool = patches.contiguous().view(64, 9, 512, -1).max(3)[0]

        cos_sim = F.cosine_similarity(patch_pool, subpatches_pool, 2)
        cos_sim = cos_sim.view(64, 3, 3)


        extract_patches = featuremap.unfold(2, 3, 1).unfold(3, 3, 1)  # 64, 512, 6, 6, 3, 3
        patches = extract_patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(64, -1, 512, 3, 3)  # 64, 36, 512, 3, 3
        patch_pool = patches.contiguous().view(64, patches.shape[1], 512, -1).mean(3) # 64, 36, 512, 9 =>  64, 36, 512

        subpatches = patches[:, :, :, 1:2, 1:2]
        subpatches_pool = subpatches.contiguous().view(64, subpatches.shape[1], 512, -1).mean(3) # 64, 36, 512

        #patches[:, :, :, 1:3, 1:3] = 0.
        #patch_pool = patches.contiguous().view(64, 9, 512, -1).max(3)[0]

        cos_sim_2 = F.cosine_similarity(patch_pool, subpatches_pool, 2)
        cos_sim_2 = cos_sim_2.view(64, 6, 6)

        cos_sim_3 = F.cosine_similarity(patch_pool, F.avg_pool2d(featuremap, 8).squeeze(-1).squeeze(-1).view(64, 1, 512), 2)
        cos_sim_3 = cos_sim_3.view(64, 6, 6)

        return cos_sim.detach().cpu().numpy(), cos_sim_2.detach().cpu().numpy(), cos_sim_3.detach().cpu().numpy()


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
