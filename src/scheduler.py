from torch.optim import lr_scheduler

def get_scheduler(optimizer, config):
    """Return a learning rate scheduler
    Parameters:
        configimizer          -- the configimizer of the network
        config (configion class) -- stores all the experiment flags; needs to be a subclass of Baseconfigions．　
                              config.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <config.n_epochs> epochs
    and linearly decay the rate to zero over the next <config.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/configim.html for more details.
    """
    if config.lr.lr_policy == 'none':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    elif config.lr.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - config.num_epochs/2) / float(config.num_epochs/2 + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config.lr.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.num_epochs/5, gamma=0.5)
    elif config.lr.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config.lr_policy)
    return scheduler