import os

import torch


def save_and_load_tester(impl):
    impl.save_model(os.path.join("test_data", "model.pt"))
    impl.load_model(os.path.join("test_data", "model.pt"))


def predict_best_action_tester(impl, discrete):
    observations = torch.rand((100,) + impl.observation_shape)
    y = impl.predict_best_action(observations)
    if discrete:
        assert y.shape == (100,)
    else:
        assert y.shape == (100, impl.action_size)


def sample_action_tester(impl, discrete):
    observations = torch.rand((100,) + impl.observation_shape)
    y = impl.sample_action(observations)
    if discrete:
        assert y.shape == (100,)
    else:
        assert y.shape == (100, impl.action_size)


def predict_value_tester(impl, discrete):
    observations = torch.rand((100,) + impl.observation_shape)
    if discrete:
        actions = torch.randint(size=(100,), low=0, high=impl.action_size)
    else:
        actions = torch.rand((100, impl.action_size))
    value = impl.predict_value(observations, actions)
    assert value.shape == (100,)


def impl_tester(impl, discrete, test_predict_value=True):
    impl.build()
    save_and_load_tester(impl)
    predict_best_action_tester(impl, discrete)
    sample_action_tester(impl, discrete)
    if test_predict_value:
        predict_value_tester(impl, discrete)
