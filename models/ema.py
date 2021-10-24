from copy import deepcopy

import paddle

class ModelEMA(object):
    def __init__(self, args, model):
        self.ema = deepcopy(model)
        self.ema.to(args.device)
        self.ema.eval()
        self.decay = args.ema_decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.stop_gradient = True

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with paddle.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].set_value(ema_v * self.decay + (1. - self.decay) * model_v)
                # esd[k]=ema_v * self.decay + (1. - self.decay) * model_v
            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])
