import os
import json
import msgpack
import d3rlpy.experimental.server.async as async

from flask import Flask, request, jsonify, send_file
from d3rlpy.algos import create_algo
from .tasks import train
from .message import unpack_experience


class Server:
    def __init__(self,
                 algo_name,
                 algo_params,
                 dataset,
                 dir_path='d3rlpy_logs/worker'):
        self.algo_name = algo_name
        self.algo_params = algo_params
        self.dataset = dataset
        self.dir_path = os.path.abspath(dir_path)
        self.n_trials = 0

        # setup flask server
        self.app = Flask(__name__)
        self.app.add_url_rule('/train', 'train', self.train, methods=['POST'])
        self.app.add_url_rule('/data',
                              'data',
                              self.append_data,
                              methods=['POST'])
        self.app.add_url_rule('/model',
                              'model',
                              self.get_model,
                              methods=['GET'])
        self.app.add_url_rule('/status',
                              'status',
                              self.get_status,
                              methods=['GET'])

        # prepare directory
        os.makedirs(dir_path)

        # save initial dataset
        self.dataset_path = os.path.join(self.dir_path, 'dataset.h5')
        dataset.dump(self.dataset_path)

        self.model_save_path_tmpl = os.path.join(self.dir_path, 'model_%d.pt')
        self.experiment_name_tmpl = 'worker_training_%d'
        self.train_uid = None
        self.latest_metrics = {}

        # start initial training
        self._dispatch_training_job()

    def run(self, host='0.0.0.0', port=8000, debug=False):
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        self.app.run(host=host, port=port, debug=debug)

    def append_data(self):
        req_data = request.data
        observations, actions, rewards, terminals = unpack_experience(req_data)
        self.dataset.append(observations, actions, rewards, terminals)
        self.dataset.dump(os.path.join(self.dir_path, 'dataset.h5'))
        return jsonify({'dataset': {'total_episodes': len(self.dataset)}})

    def get_model(self):
        # initialize algorithm
        algo = create_algo(self.algo_name, self.dataset.is_action_discrete(),
                           **self.algo_params)
        algo.create_impl(self.dataset.get_observation_shape(),
                         self.dataset.get_action_size())

        # load latest model
        trial = self.n_trials - 1
        while trial >= 0:
            model_path = self.model_save_path_tmpl % trial
            if os.path.exists(model_path):
                algo.load_model(model_path)
                break
            trial -= 1

        if trial < 0:
            return jsonify({'status': 'empty'})

        # save policy
        policy_path = os.path.join(self.dir_path, 'policy.pt')
        algo.save_policy(policy_path)

        # send back policy data
        res = send_file(policy_path,
                        as_attachment=True,
                        attachment_filename='policy.pt')

        return res

    def get_status(self):
        is_training = self._check_training_status()
        rets = {
            'training': is_training,
            'latest_metrics': self.latest_metrics,
            'n_trials': self.n_trials
        }
        return jsonify(rets)

    def train(self):
        if self._check_training_status():
            return jsonify({'status': 'training'})

        self._dispatch_training_job()

        return jsonify({'status': 'success'})

    def _dispatch_training_job(self):
        model_save_path = self.model_save_path_tmpl % self.n_trials
        experiment_name = self.experiment_name_tmpl % self.n_trials

        # dispatch training job
        self.train_uid = async.dispatch(train,
                                        self.algo_name,
                                        self.algo_params,
                                        self.dataset_path,
                                        model_save_path,
                                        experiment_name=experiment_name,
                                        with_timestamp=False,
                                        logdir=self.dir_path)

        self.n_trials += 1

    def _check_training_status(self):
        if self.train_uid is None:
            return False
        elif async.is_running(self.train_uid):
            return True

        # training is finished
        self.latest_metrics = async.get(self.train_uid)
        self.train_uid = None
        return False
