import torch

from inspect import signature


def soft_sync(targ_model, model, tau):
    with torch.no_grad():
        params = model.parameters()
        targ_params = targ_model.parameters()
        for p, p_targ in zip(params, targ_params):
            p_targ.data.mul_(1 - tau)
            p_targ.data.add_(tau * p.data)


def hard_sync(targ_model, model):
    with torch.no_grad():
        params = model.parameters()
        targ_params = targ_model.parameters()
        for p, p_targ in zip(params, targ_params):
            p_targ.data.copy_(p.data)


def set_eval_mode(impl):
    for key in dir(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            module.eval()


def set_train_mode(impl):
    for key in dir(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            module.train()


def to_cuda(impl, device):
    for key in dir(impl):
        module = getattr(impl, key)
        if isinstance(module, (torch.nn.Module, torch.nn.Parameter)):
            module.cuda(device)


def to_cpu(impl):
    for key in dir(impl):
        module = getattr(impl, key)
        if isinstance(module, (torch.nn.Module, torch.nn.Parameter)):
            module.cpu()


def freeze(impl):
    for key in dir(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            for p in module.parameters():
                p.requires_grad = False


def unfreeze(impl):
    for key in dir(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            for p in module.parameters():
                p.requires_grad = True


def compute_augmentation_mean(augmentation, n_augmentations, func, inputs,
                              targets):
    ret = 0.0
    for _ in range(n_augmentations):
        kwargs = dict(inputs)
        for target in targets:
            kwargs[target] = augmentation.transform(kwargs[target])
        ret += func(**kwargs)
    return ret / n_augmentations


def get_state_dict(impl):
    rets = {}
    for key in dir(impl):
        obj = getattr(impl, key)
        if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
            rets[key] = obj.state_dict()
    return rets


def set_state_dict(impl, chkpt):
    for key in dir(impl):
        obj = getattr(impl, key)
        if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
            obj.load_state_dict(chkpt[key])


def map_location(device):
    if 'cuda' in device:
        return lambda storage, loc: storage.cuda(device)
    if 'cpu' in device:
        return 'cpu'
    raise ValueError('invalid device={}'.format(device))


def torch_api(scaler_targets=[]):
    def _torch_api(f):
        # get argument names
        sig = signature(f)
        arg_keys = list(sig.parameters.keys())[1:]

        def wrapper(self, *args, **kwargs):
            # convert all args to torch.Tensor
            tensors = []
            for i, val in enumerate(args):
                if isinstance(val, torch.Tensor):
                    tensor = val
                else:
                    tensor = torch.tensor(data=val,
                                          dtype=torch.float32,
                                          device=self.device)

                # preprocess
                if self.scaler and arg_keys[i] in scaler_targets:
                    tensor = self.scaler.transform(tensor)

                tensors.append(tensor)
            return f(self, *tensors, **kwargs)

        return wrapper

    return _torch_api


def eval_api(f):
    def wrapper(self, *args, **kwargs):
        set_eval_mode(self)
        return f(self, *args, **kwargs)

    return wrapper


def train_api(f):
    def wrapper(self, *args, **kwargs):
        set_train_mode(self)
        return f(self, *args, **kwargs)

    return wrapper
