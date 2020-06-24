import numpy as np
import os
import json

from datetime import datetime


class SkbrlLogger:
    def __init__(self,
                 experiment_name,
                 root_dir='logs',
                 verbose=True,
                 tensorboard=True):
        self.verbose = verbose

        # add timestamp to prevent unintentional overwrites
        date = datetime.now().strftime('%Y%m%d%H%M%S')
        self.experiment_name = experiment_name + '_' + date
        self.logdir = os.path.join(root_dir, self.experiment_name)
        self.metrics_buffer = {}

        os.makedirs(self.logdir)

        if tensorboard:
            from tensorboardX import SummaryWriter
            tfboard_path = os.path.join('runs', self.experiment_name)
            self.writer = SummaryWriter(logdir=tfboard_path)
        else:
            self.writer = None

        self.params = None

    def add_params(self, params):
        assert self.params is None, 'add_params can be called only once.'

        # remove non-scalar objects
        params = {k: v for k, v in params.items() if np.isscalar(v)}

        # save dictionary as json file
        with open(os.path.join(self.logdir, 'params.json'), 'w') as f:
            f.write(json.dumps(params))

        if self.verbose:
            for key, val in params.items():
                print('{}={}'.format(key, val))

        self.params = params

    def add_metric(self, name, value):
        if name not in self.metrics_buffer:
            self.metrics_buffer[name] = []
        self.metrics_buffer[name].append(value)

    def commit(self, epoch, step):
        metrics = {}
        for name, buffer in self.metrics_buffer.items():
            metric = sum(buffer) / len(buffer)

            with open(os.path.join(self.logdir, name + '.csv'), 'a') as f:
                print('%d,%d,%f' % (epoch, step, metric), file=f)

            if self.verbose:
                print('epoch=%d step=%d %s=%f' % (epoch, step, name, metric))

            if self.writer:
                self.writer.add_scalar('metrics/' + name, metric, epoch)

            metrics[name] = metric
            self.metrics_buffer[name] = []

        if self.params and self.writer:
            self.writer.add_hparams(self.params,
                                    metrics,
                                    name=self.experiment_name,
                                    global_step=epoch)
