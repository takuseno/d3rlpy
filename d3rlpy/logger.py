import numpy as np
import os
import json
import time

from contextlib import contextmanager
from datetime import datetime


# default json encoder for numpy objects
def default_json_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError


class D3RLPyLogger:
    def __init__(self,
                 experiment_name,
                 save_metrics=True,
                 root_dir='logs',
                 verbose=True,
                 tensorboard=True,
                 with_timestamp=True):
        self.save_metrics = save_metrics
        self.verbose = verbose

        # add timestamp to prevent unintentional overwrites
        while True:
            if with_timestamp:
                date = datetime.now().strftime('%Y%m%d%H%M%S')
                self.experiment_name = experiment_name + '_' + date
            else:
                self.experiment_name = experiment_name

            if self.save_metrics:
                self.logdir = os.path.join(root_dir, self.experiment_name)
                if not os.path.exists(self.logdir):
                    os.makedirs(self.logdir)
                    break
                else:
                    if with_timestamp:
                        time.sleep(1.0)
                    else:
                        raise ValueError('%s already exists.' % self.logdir)
            else:
                break

        self.metrics_buffer = {}

        if tensorboard:
            from tensorboardX import SummaryWriter
            tfboard_path = os.path.join('runs', self.experiment_name)
            self.writer = SummaryWriter(logdir=tfboard_path)
        else:
            self.writer = None

        self.params = None

    def add_params(self, params):
        assert self.params is None, 'add_params can be called only once.'

        if self.save_metrics:
            # save dictionary as json file
            with open(os.path.join(self.logdir, 'params.json'), 'w') as f:
                json_str = json.dumps(params,
                                      default=default_json_encoder,
                                      indent=2)
                f.write(json_str)

        if self.verbose:
            for key, val in params.items():
                print('{}={}'.format(key, val))

        # remove non-scaler values for HParams
        self.params = {k: v for k, v in params.items() if np.isscalar(v)}

    def add_metric(self, name, value):
        if name not in self.metrics_buffer:
            self.metrics_buffer[name] = []
        self.metrics_buffer[name].append(value)

    def commit(self, epoch, step):
        metrics = {}
        for name, buffer in self.metrics_buffer.items():
            metric = sum(buffer) / len(buffer)

            if self.save_metrics:
                with open(os.path.join(self.logdir, name + '.csv'), 'a') as f:
                    print('%d,%d,%f' % (epoch, step, metric), file=f)

            if self.verbose:
                print('epoch=%d step=%d %s=%f' % (epoch, step, name, metric))

            if self.writer:
                self.writer.add_scalar('metrics/' + name, metric, epoch)

            metrics[name] = metric

        if self.params and self.writer:
            self.writer.add_hparams(self.params,
                                    metrics,
                                    name=self.experiment_name,
                                    global_step=epoch)

        # initialize metrics buffer
        self.metrics_buffer = {}

    def save_model(self, epoch, algo):
        if self.save_metrics:
            # save entire model
            model_path = os.path.join(self.logdir, 'model_%d.pt' % epoch)
            algo.save_model(model_path)

    @contextmanager
    def measure_time(self, name):
        name = 'time_' + name
        start = time.time()
        try:
            yield
        finally:
            self.add_metric(name, time.time() - start)
