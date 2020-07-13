import torch.optim as optim


class Optimizer(object):
    def __init__(self, parameters, config):
        self.config = config
        self.optimizer = build_optimizer(parameters, config)
        self.global_step = 1
        self.current_epoch = 0
        self.lr = config.lr
        self.decay_ratio = config.decay_ratio
        self.epoch_decay_flag = False

    def step(self):
        self.global_step += 1
        self.optimizer.step()

    def epoch(self):
        self.current_epoch += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def decay_lr(self):
        self.lr *= self.decay_ratio
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def step_decay_lr(self):
        first_adjust = 4e3
        second_adjust = 3e4
        final_step = 2.3e5  # 16 batch 30 epoch
        max_lr = 2e-4
        min_lr = 2e-6
        if self.global_step <= first_adjust:
            self.lr = liner(self.global_step, 1e-6, 4000, 2.5e-4)
        elif self.global_step <= second_adjust:
            self.lr = max_lr
        else:
            self.lr = exp(self.global_step, first_adjust, 2.5e-4, final_step, 2.5e-6)


def liner(step, y1, x2, y2):
    y = (y2-y1)/x2 * step + y1
    return y

def exp(step, x1, y1, x2, y2):

    return

def build_optimizer(parameters, config):
    if config.type == 'adam':
        return optim.Adam(
            parameters,
            lr=config.lr,
            betas=(0.9, 0.98),
            eps=1e-08,
            weight_decay=config.weight_decay
        )
    elif config.type == 'sgd':
        return optim.SGD(
            params=parameters,
            lr=config.lr,
            momentum=config.momentum,
            nesterov=config.nesterov,
            weight_decay=config.weight_decay
        )
    elif config.type == 'adadelta':
        return optim.Adadelta(
            params=parameters,
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    else:
        raise NotImplementedError
