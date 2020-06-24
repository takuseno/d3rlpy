import torch


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


def torch_api(f):
    def wrapper(self, *args, **kwargs):
        # convert all args to torch.Tensor
        tensors = []
        for val in args:
            tensor = torch.tensor(val, dtype=torch.float32, device=self.device)
            tensors.append(tensor)
        return f(self, *tensors, **kwargs)

    return wrapper
