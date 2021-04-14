import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision import transforms

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

# Bayesian Unet implementation
class Bayesian_Unet(nn.Module):
    def __init__(self, in_channels, batch_size):
        super(Bayesian_Unet, self).__init__()
        self.in_channels=in_channels
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
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
        #print(skip3.shape)
        #print("shape after 3rd conv segment: " ,x.shape)
        x = self.max3(skip3)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.trans_conv1(x)
        #print("shape after 1st trans_conv segment: " ,x.shape)
        #print(x.shape)
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
        
        kl_sum = torch.Tensor([0]).to(self.device)
        
        for m in self.children():
            try:
                kl = m.kl_loss_()
                kl_sum += kl
                #print("succeeded to calculate kl-loss")
            except:
                #print("failed to calculate kl-loss")
                #print(m)
                continue
    
        return kl_sum

# class for a bayesian conv layer
class Bayesian_CNNLayer(torch.nn.Module):
    """
    formula for output dimensions: (((input_size)+2*padding-(kernel_size-1)-1)/stride) +1
    example: square input 100 pixel, stride = 5, padding = 3, kernel_size = 9 --> output 20*20
    
    """
    
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, priors=None, verbose=False):
        super(Bayesian_CNNLayer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size) # possible to enter a tuple or a single digit
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        # encode prior assumptions
        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']
        
        self.W_mu = Parameter(torch.empty((output_channels, input_channels, *self.kernel_size), device=self.device))
        self.W_rho = Parameter(torch.empty((output_channels, input_channels, *self.kernel_size), device=self.device))
        
        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((output_channels), device=self.device))
            self.bias_rho = Parameter(torch.empty((output_channels), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        
        # mode defined below
        self.reset_parameters()
        
    # when layer is initialized, initialize the weigth matrices
    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)
            
    def forward(self, input, sample=True):
        # sampling the standard normal
        W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
        self.W_sigma = torch.log1p(torch.exp(self.W_rho)) # exp() to make entries positive, log1p returns tensor ln(1 + input), hence stays positive
        # affine transform of standard normal to get the weights
        weight = self.W_mu + W_eps * self.W_sigma
        
        # same thing with the bias
        if self.use_bias:
            bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_eps * self.bias_sigma
        else:
            bias = None

        if self.verbose:
            print(weight[0,0,0,:])
            
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
    
    def kl_loss(self):
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
        
# class for a bayesian transposed conv layer (basically the same as above, rewrite last line)            
class Bayesian_transposed_CNNLayer(torch.nn.Module):
    """
    formula for output dimensions: (input_size-1)*stride - 2*padding + (kernel_size -1) +1
    example: square input 100 pixel, stride = 5, padding = 3, kernel_size = 9 --> output 498*498
    
    """
    
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, priors=None, output_padding=0, verbose=False):
        super(Bayesian_transposed_CNNLayer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size) # possible to enter a tuple or a single digit
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.output_padding = output_padding
        self.verbose = verbose
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # encode prior assumptions
        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']
        
        self.W_mu = Parameter(torch.empty((input_channels, output_channels, *self.kernel_size), device=self.device))
        self.W_rho = Parameter(torch.empty((input_channels, output_channels, *self.kernel_size), device=self.device))
        
        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((output_channels), device=self.device))
            self.bias_rho = Parameter(torch.empty((output_channels), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        
        # mode defined below
        self.reset_parameters()
        
    # when layer is initialized, initialize the weigth matrices
    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)
            
    def forward(self, input, sample=True):
        # sampling the standard normal
        W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
        self.W_sigma = torch.log1p(torch.exp(self.W_rho)) # exp() to make entries positive, log1p returns tensor ln(1 + input), hence stays positive
        # affine transform of standard normal to get the weights
        weight = self.W_mu + W_eps * self.W_sigma
        
        # same thing with the bias
        if self.use_bias:
            bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_eps * self.bias_sigma
        else:
            bias = None

        if self.verbose:
            print(weight)
            
        return F.conv_transpose2d(input, weight, bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
    
    def kl_loss(self):
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl

# composite layer concatenating baysian layer with with activation, batch normalization, dropout
class composite_bay_conv(torch.nn.Module):
    """    
    concatenating baysian layer with with activation, batch normalization, dropout
    
    """
    def __init__(self, input_channels, output_channels, kernel_size, 
                 batch_size, stride=1, padding=0, dilation=1, bias=True, priors=None, 
                 activation=nn.ReLU(), dropout=False, rate=0.3, batch_norm=True, verbose=False):
        super(composite_bay_conv, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size=kernel_size
        self.stride = stride
        self.padding=padding
        self.dilation=dilation
        self.bias=bias
        self.priors=priors
        self.activation = activation
        self.dropout = dropout
        self.rate = rate
        self.batch_norm = batch_norm
        self.batch_size = batch_size
        self.verbose=verbose

        self.bl = Bayesian_CNNLayer(input_channels=self.input_channels, 
                              output_channels=self.output_channels, 
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              bias=self.bias,
                              priors=self.priors,
                              verbose = self.verbose
                              )
        
        if batch_norm:
            self.batch_norm_lay = nn.BatchNorm2d(self.output_channels)
            
        if dropout:
            self.dropout_lay = nn.Dropout2d(self.rate)
        
    def forward(self, x):
        #assert(x.shape[0] == self.batch_size)
        x = self.bl(x)
        x = self.activation(x)
        if self.batch_norm:
            x = self.batch_norm_lay(x)
        if self.dropout:
            x = self.dropout_lay(x)
        
        return x
    
    def kl_loss_(self):
        return self.bl.kl_loss()

# composite transposed layer concatenating baysian layer with with activation, batch normalization, dropout
class composite_bay_trans_conv(torch.nn.Module):
    """
    
    concatenating baysian layer with with activation, batch normalization, dropout
    
    """
    
    def __init__(self, input_channels, output_channels, kernel_size, 
                 batch_size, stride=1, padding=0, dilation=1, bias=True, priors=None, 
                 activation=nn.Identity(), dropout=False, rate=0.3, batch_norm=True, output_padding=0, verbose=False):
        super(composite_bay_trans_conv, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size=kernel_size
        self.stride = stride
        self.padding=padding
        self.output_padding = output_padding
        self.dilation=dilation
        self.bias=bias
        self.priors=priors
        self.activation = activation
        self.dropout = dropout
        self.rate = rate
        self.batch_norm = batch_norm
        self.batch_size = batch_size
        self.verbose=verbose

        self.trans_bl = Bayesian_transposed_CNNLayer(input_channels=self.input_channels, 
                              output_channels=self.output_channels, 
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              bias=self.bias,
                              priors=self.priors,
                              output_padding=self.output_padding,
                              verbose=self.verbose
                              )
        
        if batch_norm:
            self.batch_norm_lay = nn.BatchNorm2d(self.output_channels)
            
        if dropout:
            self.dropout_lay = nn.Dropout2d(self.rate)
        
    def forward(self, x):
        #assert(x.shape[0] == self.batch_size)
        x = self.trans_bl(x)
        x = self.activation(x)
        if self.batch_norm:
            x = self.batch_norm_lay(x)
        if self.dropout:
            x = self.dropout_lay(x)
        
        return x
    
    def kl_loss_(self):
        return self.trans_bl.kl_loss()