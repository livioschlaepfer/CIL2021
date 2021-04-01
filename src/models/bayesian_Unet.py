from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision import transforms
from src.models.bm_parts import composite_bay_conv, composite_bay_trans_conv

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

class Bayesian_Unet(nn.Module):
    def __init__(self, config):
        super(Bayesian_Unet, self).__init__()
        self.config = config
        self.init_model()
        self.init_criterion()
        self.model = self

    def init_model(self):
        self.batch_size = self.config.batch_size
        self.conv1 = composite_bay_conv(3, 64, 3, batch_size=self.batch_size, padding=1)
        self.conv2 = composite_bay_conv(64, 64, 3, batch_size=self.batch_size, padding=1)
        self.max1 = nn.MaxPool2d(2)
        self.conv3 = composite_bay_conv(64, 128, 3, batch_size=self.batch_size, padding=1)
        self.conv4 = composite_bay_conv(128, 128, 3, batch_size=self.batch_size, padding=1)
        self.max2 = nn.MaxPool2d(2)
        self.conv5 = composite_bay_conv(128, 256, 3, batch_size=self.batch_size, padding=1)
        self.conv6 = composite_bay_conv(256, 256, 3, batch_size=self.batch_size, padding=1)
        self.max3 = nn.MaxPool2d(2)
        self.conv7 = composite_bay_conv(256, 512, 3, batch_size=self.batch_size, padding=1)
        self.conv8 = composite_bay_conv(512, 512, 3, batch_size=self.batch_size, padding=1)
        self.trans_conv1 = composite_bay_trans_conv(512, 256, 2, batch_size=self.batch_size, stride=2)
        self.conv9 = composite_bay_conv(512, 256, 3, batch_size=self.batch_size, padding=1)
        self.conv10 = composite_bay_conv(256, 256, 3, batch_size=self.batch_size, padding=1)
        self.trans_conv2 = composite_bay_trans_conv(256, 128, 2, batch_size=self.batch_size, stride=2)
        self.conv11 = composite_bay_conv(256, 128, 3, batch_size=self.batch_size, padding=1)
        self.conv12 = composite_bay_conv(128, 128, 3, batch_size=self.batch_size, padding=1)
        self.trans_conv3 = composite_bay_trans_conv(128, 64, 2, batch_size=self.batch_size, stride=2)
        self.conv13 = composite_bay_conv(128, 64, 3, batch_size=self.batch_size, padding=1)
        self.conv14 = composite_bay_conv(64, 64, 3, batch_size=self.batch_size, padding=1)
        self.conv14 = composite_bay_conv(64, 1, 1, batch_size=self.batch_size, padding=1)
        # last sigmoid layer to get predictions in [0,1]
        self.out_sig = nn.Sigmoid()

    def init_criterion(self):
        # Setup the loss
        # self.criterion = torch.nn.MSELoss(reduction='mean')
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        # self.criterion = nn.NLLLoss()

                
    def forward(self, x):
        self.shape = x.shape
        self.crop_out = transforms.CenterCrop((self.shape[2], self.shape[3]))
        #print(self.shape)        
        x = self.conv1(x)
        skip1 = self.conv2(x)
        #print("shape after 1st conv segment: ",x.shape)
        x = self.max1(skip1)
        x = self.conv3(x)
        skip2 = self.conv4(x)
        #print("shape after 2nd conv segment: " ,x.shape)
        x = self.max2(skip2)
        x = self.conv5(x)
        skip3 = self.conv6(x)
        #print("shape after 3rd conv segment: " ,x.shape)
        x = self.max3(skip3)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.trans_conv1(x)
        #print("shape after 1st trans_conv segment: " ,x.shape)
        x = self.conv9(torch.cat((x, skip3), dim=1))
        x = self.conv10(x)
        x = self.trans_conv2(x)
        #print("shape after 2nd trans_conv segment: " ,x.shape)
        self.crop_skip2 = transforms.CenterCrop((skip2.shape[2],skip2.shape[3]))
        x = self.conv11(torch.cat((self.crop_skip2(x), skip2), dim=1))
        x = self.conv12(x)
        x = self.trans_conv3(x)
        #print("shape after 3rd trans_conv segment: " ,x.shape)
        self.crop_skip1 = transforms.CenterCrop((skip1.shape[2],skip1.shape[3]))
        x = self.conv13(torch.cat((self.crop_skip1(x), skip1), dim=1))
        x = self.conv14(x)
        x = self.crop_out(x)
        out = self.out_sig(x)
                
        return out
    
    def predict_class_probs(self, x, num_forward_passes=10):
        batch_size = x.shape[0]

        # make n random forward passes
        # compute the categorical softmax probabilities
        # marginalize the probabilities over the n forward passes
        probs = torch.zeros([num_forward_passes, batch_size, x.shape[2], x.shape[3]])
        
        for i in range(num_forward_passes):
            prob_sigmoid = self.forward(x)
            probs[i,:,:,:] = torch.squeeze(prob_sigmoid)            

        return torch.mean(probs, dim=0), torch.var(probs, dim=0)


    def kl_loss(self):
        '''
        Computes the KL divergence loss for all layers.
        '''
        # TODO: enter your code here
        
        kl_sum = torch.Tensor([0])
        
        for m in self.children():
            try:
                kl = m.kl_loss_()
                kl_sum += kl
            except:
                print("fail")
                continue
    
        return kl_sum