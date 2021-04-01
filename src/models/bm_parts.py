import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision import transforms

# class for a bayesian conv layer
class Bayesian_CNNLayer(torch.nn.Module):
    """
    formula for output dimensions: (((input_size)+2*padding-(kernel_size-1)-1)/stride) +1
    example: square input 100 pixel, stride = 5, padding = 3, kernel_size = 9 --> output 20*20
    
    """
    
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, priors=None):
        super(Bayesian_CNNLayer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size) # possible to enter a tuple or a single digit
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        
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
        
        self.W_mu = Parameter(torch.empty((output_channels, input_channels, *self.kernel_size)))
        self.W_rho = Parameter(torch.empty((output_channels, input_channels, *self.kernel_size)))
        
        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((output_channels)))
            self.bias_rho = Parameter(torch.empty((output_channels)))
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
        W_eps = torch.empty(self.W_mu.size()).normal_(0, 1)
        self.W_sigma = torch.log1p(torch.exp(self.W_rho)) # exp() to make entries positive, log1p returns tensor ln(1 + input), hence stays positive
        # affine transform of standard normal to get the weights
        weight = self.W_mu + W_eps * self.W_sigma
        
        # same thing with the bias
        if self.use_bias:
            bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1)
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_eps * self.bias_sigma
        else:
            bias = None
        #print(input.shape)
            
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
    
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, priors=None, output_padding=0):
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
        
        self.W_mu = Parameter(torch.empty((input_channels, output_channels, *self.kernel_size)))
        self.W_rho = Parameter(torch.empty((input_channels, output_channels, *self.kernel_size)))
        
        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((output_channels)))
            self.bias_rho = Parameter(torch.empty((output_channels)))
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
        W_eps = torch.empty(self.W_mu.size()).normal_(0, 1)
        self.W_sigma = torch.log1p(torch.exp(self.W_rho)) # exp() to make entries positive, log1p returns tensor ln(1 + input), hence stays positive
        # affine transform of standard normal to get the weights
        weight = self.W_mu + W_eps * self.W_sigma
        
        # same thing with the bias
        if self.use_bias:
            bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1)
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_eps * self.bias_sigma
        else:
            bias = None
        #print(input.shape)
            
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
                 activation=nn.ReLU(), dropout=False, rate=0.3, batch_norm=True):
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
        
        if batch_norm:
            self.batch_norm_lay = nn.BatchNorm2d(self.output_channels)
            
        if dropout:
            self.dropout_lay = nn.Dropout2d(self.rate)
        
    def forward(self, x):
        assert(x.shape[0] == self.batch_size)
        self.bl = Bayesian_CNNLayer(input_channels=self.input_channels, 
                              output_channels=self.output_channels, 
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              bias=self.bias,
                              priors=self.priors
                              )
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
                 activation=nn.Identity(), dropout=False, rate=0.3, batch_norm=True, output_padding=0):
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
        
        if batch_norm:
            self.batch_norm_lay = nn.BatchNorm2d(self.output_channels)
            
        if dropout:
            self.dropout_lay = nn.Dropout2d(self.rate)
        
    def forward(self, x):
        assert(x.shape[0] == self.batch_size)
        self.trans_bl = Bayesian_transposed_CNNLayer(input_channels=self.input_channels, 
                              output_channels=self.output_channels, 
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              bias=self.bias,
                              priors=self.priors,
                              output_padding=self.output_padding
                              )
        x = self.trans_bl(x)
        x = self.activation(x)
        if self.batch_norm:
            x = self.batch_norm_lay(x)
        if self.dropout:
            x = self.dropout_lay(x)
        
        return x
    
    def kl_loss_(self):
        return self.trans_bl.kl_loss()