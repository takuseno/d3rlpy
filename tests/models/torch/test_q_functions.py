import numpy as np
import pytest
import torch

from d3rlpy.models.torch.q_functions import create_discrete_q_function
from d3rlpy.models.torch.q_functions import create_continuous_q_function
from d3rlpy.models.torch.q_functions import _pick_value_by_action
from d3rlpy.models.torch.q_functions import _make_taus_prime
from d3rlpy.models.torch.q_functions import _quantile_huber_loss
from d3rlpy.models.torch.q_functions import _reduce_ensemble
from d3rlpy.models.torch.q_functions import _reduce_quantile_ensemble
from d3rlpy.models.torch.q_functions import DiscreteQRQFunction
from d3rlpy.models.torch.q_functions import ContinuousQRQFunction
from d3rlpy.models.torch.q_functions import DiscreteIQNQFunction
from d3rlpy.models.torch.q_functions import ContinuousIQNQFunction
from d3rlpy.models.torch.q_functions import DiscreteFQFQFunction
from d3rlpy.models.torch.q_functions import ContinuousFQFQFunction
from d3rlpy.models.torch.q_functions import DiscreteQFunction
from d3rlpy.models.torch.q_functions import EnsembleDiscreteQFunction
from d3rlpy.models.torch.q_functions import ContinuousQFunction
from d3rlpy.models.torch.q_functions import EnsembleContinuousQFunction
from d3rlpy.models.torch.q_functions import compute_max_with_n_actions
from .model_test import check_parameter_updates, DummyHead


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('n_ensembles', [1, 2])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('embed_size', [64])
@pytest.mark.parametrize('use_batch_norm', [False, True])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
def test_create_discrete_q_function(observation_shape, action_size, batch_size,
                                    n_ensembles, n_quantiles, embed_size,
                                    use_batch_norm, q_func_type):
    q_func = create_discrete_q_function(observation_shape, action_size,
                                        n_ensembles, n_quantiles, embed_size,
                                        use_batch_norm, q_func_type)

    assert isinstance(q_func, EnsembleDiscreteQFunction)
    for f in q_func.q_funcs:
        assert f.head.use_batch_norm == use_batch_norm
        if q_func_type == 'mean':
            assert isinstance(f, DiscreteQFunction)
        elif q_func_type == 'qr':
            assert isinstance(f, DiscreteQRQFunction)
        elif q_func_type == 'iqn':
            assert isinstance(f, DiscreteIQNQFunction)
        elif q_func_type == 'fqf':
            assert isinstance(f, DiscreteFQFQFunction)

    x = torch.rand((batch_size, ) + observation_shape)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('n_ensembles', [1, 2])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('embed_size', [64])
@pytest.mark.parametrize('use_batch_norm', [False, True])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
def test_create_continuous_q_function(observation_shape, action_size,
                                      batch_size, n_ensembles, n_quantiles,
                                      embed_size, use_batch_norm, q_func_type):
    q_func = create_continuous_q_function(observation_shape, action_size,
                                          n_ensembles, n_quantiles, embed_size,
                                          use_batch_norm, q_func_type)

    assert isinstance(q_func, EnsembleContinuousQFunction)
    for f in q_func.q_funcs:
        if q_func_type == 'mean':
            assert isinstance(f, ContinuousQFunction)
        elif q_func_type == 'qr':
            assert isinstance(f, ContinuousQRQFunction)
        elif q_func_type == 'iqn':
            assert isinstance(f, ContinuousIQNQFunction)
        elif q_func_type == 'fqf':
            assert isinstance(f, ContinuousFQFQFunction)
        assert f.head.use_batch_norm == use_batch_norm

    x = torch.rand((batch_size, ) + observation_shape)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)


@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_quantiles', [0, 200])
@pytest.mark.parametrize('keepdims', [True, False])
def test_pick_value_by_action(batch_size, action_size, n_quantiles, keepdims):
    if n_quantiles == 0:
        values = torch.rand(batch_size, action_size)
    else:
        values = torch.rand(batch_size, action_size, n_quantiles)

    action = torch.randint(action_size, size=(batch_size, ))

    rets = _pick_value_by_action(values, action, keepdims)

    if n_quantiles == 0:
        if keepdims:
            assert rets.shape == (batch_size, 1)
        else:
            assert rets.shape == (batch_size, )
    else:
        if keepdims:
            assert rets.shape == (batch_size, 1, n_quantiles)
        else:
            assert rets.shape == (batch_size, n_quantiles)

    rets = rets.view(batch_size, -1)

    for i in range(batch_size):
        assert (rets[i] == values[i][action[i]]).all()


def ref_quantile_huber_loss(a, b, taus, n_quantiles):
    abs_diff = np.abs(a - b).reshape((-1, ))
    l2_diff = ((a - b)**2).reshape((-1, ))
    huber_diff = np.zeros_like(abs_diff)
    huber_diff[abs_diff < 1.0] = 0.5 * l2_diff[abs_diff < 1.0]
    huber_diff[abs_diff >= 1.0] = abs_diff[abs_diff >= 1.0] - 0.5
    huber_diff = huber_diff.reshape(-1, n_quantiles, n_quantiles)
    delta = np.array((b - a) < 0.0, dtype=np.float32)
    element_wise_loss = np.abs(taus - delta) * huber_diff
    return element_wise_loss.sum(axis=2).mean()


@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('n_quantiles', [200])
def test_quantile_huber_loss(batch_size, n_quantiles):
    y = np.random.random((batch_size, n_quantiles, 1))
    target = np.random.random((batch_size, 1, n_quantiles))
    taus = np.random.random((1, 1, n_quantiles))

    ref_loss = ref_quantile_huber_loss(y, target, taus, n_quantiles)
    loss = _quantile_huber_loss(torch.tensor(y), torch.tensor(target),
                                torch.tensor(taus))

    assert np.allclose(loss.cpu().detach().numpy(), ref_loss)


@pytest.mark.parametrize('n_quantiles', [200])
def test_make_taus_prime(n_quantiles):
    taus = _make_taus_prime(n_quantiles, 'cpu:0')

    assert taus.shape == (1, n_quantiles)

    step = 1 / n_quantiles
    for i in range(n_quantiles):
        assert np.allclose(taus[0][i].numpy(), i * step + step / 2.0)


@pytest.mark.parametrize('n_ensembles', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('reduction', ['min', 'max', 'mean', 'none'])
def test_reduce_ensemble(n_ensembles, batch_size, reduction):
    y = torch.rand(n_ensembles, batch_size, 1)
    ret = _reduce_ensemble(y, reduction)
    if reduction == 'min':
        assert ret.shape == (batch_size, 1)
        assert torch.allclose(ret, y.min(dim=0).values)
    elif reduction == 'max':
        assert ret.shape == (batch_size, 1)
        assert torch.allclose(ret, y.max(dim=0).values)
    elif reduction == 'mean':
        assert ret.shape == (batch_size, 1)
        assert torch.allclose(ret, y.mean(dim=0))
    elif reduction == 'none':
        assert ret.shape == (n_ensembles, batch_size, 1)
        assert (ret == y).all()


@pytest.mark.parametrize('n_ensembles', [2])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('reduction', ['min', 'max'])
def test_reduce_quantile_ensemble(n_ensembles, n_quantiles, batch_size,
                                  reduction):
    y = torch.rand(n_ensembles, batch_size, n_quantiles)
    ret = _reduce_quantile_ensemble(y, reduction)
    mean = y.mean(dim=2)
    if reduction == 'min':
        assert ret.shape == (batch_size, n_quantiles)
        indices = mean.min(dim=0).indices
        assert torch.allclose(ret, y[indices, torch.arange(batch_size)])
    elif reduction == 'max':
        assert ret.shape == (batch_size, n_quantiles)
        indices = mean.max(dim=0).indices
        assert torch.allclose(ret, y[indices, torch.arange(batch_size)])


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('gamma', [0.99])
def test_discrete_qr_q_function(feature_size, action_size, n_quantiles,
                                batch_size, gamma):
    head = DummyHead(feature_size)
    q_func = DiscreteQRQFunction(head, action_size, n_quantiles)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)

    action = torch.randint(high=action_size, size=(batch_size, ))
    target = q_func.compute_target(x, action)
    quantiles = q_func(x, as_quantiles=True)
    assert target.shape == (batch_size, n_quantiles)
    assert (quantiles[torch.arange(batch_size), action] == target).all()

    # check quantile huber loss
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.randint(action_size, size=(batch_size, ))
    rew_tp1 = torch.rand(batch_size, 1)
    q_tp1 = torch.rand(batch_size, n_quantiles)
    loss = q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1)

    target = (rew_tp1.numpy() + gamma * q_tp1.numpy())
    y = _pick_value_by_action(q_func(obs_t, as_quantiles=True), act_t)
    taus = _make_taus_prime(n_quantiles, 'cpu:0').numpy()

    reshaped_target = np.reshape(target, (batch_size, -1, 1))
    reshaped_y = np.reshape(y.detach().numpy(), (batch_size, 1, -1))
    reshaped_taus = np.reshape(taus, (1, 1, -1))

    ref_loss = ref_quantile_huber_loss(reshaped_y, reshaped_target,
                                       reshaped_taus, n_quantiles)
    assert np.allclose(loss.cpu().detach(), ref_loss)

    # check layer connection
    check_parameter_updates(q_func, (x, ))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('gamma', [0.99])
def test_continuous_qr_q_function(feature_size, action_size, n_quantiles,
                                  batch_size, gamma):
    head = DummyHead(feature_size, action_size, concat=True)
    q_func = ContinuousQRQFunction(head, n_quantiles)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)

    target = q_func.compute_target(x, action)
    quantiles = q_func(x, action, as_quantiles=True)
    assert target.shape == (batch_size, n_quantiles)
    assert (target == quantiles).all()

    # check quantile huber loss
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.rand(batch_size, action_size)
    rew_tp1 = torch.rand(batch_size, 1)
    q_tp1 = torch.rand(batch_size, n_quantiles)
    loss = q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1)

    target = rew_tp1.numpy() + gamma * q_tp1.numpy()
    y = q_func(obs_t, act_t, as_quantiles=True).detach().numpy()
    taus = _make_taus_prime(n_quantiles, 'cpu:0').numpy()

    reshaped_target = target.reshape((batch_size, -1, 1))
    reshaped_y = y.reshape((batch_size, 1, -1))
    reshaped_taus = taus.reshape((1, 1, -1))

    ref_loss = ref_quantile_huber_loss(reshaped_y, reshaped_target,
                                       reshaped_taus, n_quantiles)
    assert np.allclose(loss.cpu().detach(), ref_loss)

    # check layer connection
    check_parameter_updates(q_func, (x, action))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('embed_size', [64])
@pytest.mark.parametrize('gamma', [0.99])
def test_discrete_iqn_q_function(feature_size, action_size, n_quantiles,
                                 batch_size, embed_size, gamma):
    head = DummyHead(feature_size)
    q_func = DiscreteIQNQFunction(head, action_size, n_quantiles, embed_size)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)

    action = torch.randint(high=action_size, size=(batch_size, ))
    target = q_func.compute_target(x, action)
    assert target.shape == (batch_size, n_quantiles)

    # TODO: check quantile huber loss
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.randint(action_size, size=(batch_size, ))
    rew_tp1 = torch.rand(batch_size, 1)
    q_tp1 = torch.rand(batch_size, n_quantiles)
    loss = q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1)

    # check layer connection
    check_parameter_updates(q_func, (x, ))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('embed_size', [64])
@pytest.mark.parametrize('gamma', [0.99])
def test_continuous_iqn_q_function(feature_size, action_size, n_quantiles,
                                   batch_size, embed_size, gamma):
    head = DummyHead(feature_size, action_size)
    q_func = ContinuousIQNQFunction(head, n_quantiles, embed_size)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)

    target = q_func.compute_target(x, action)
    assert target.shape == (batch_size, n_quantiles)

    # TODO: check quantile huber loss
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.randint(action_size, size=(batch_size, ))
    rew_tp1 = torch.rand(batch_size, 1)
    q_tp1 = torch.rand(batch_size, n_quantiles)
    loss = q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1)

    # check layer connection
    check_parameter_updates(q_func, (x, action))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('embed_size', [64])
@pytest.mark.parametrize('gamma', [0.99])
def test_discrete_fqf_q_function(feature_size, action_size, n_quantiles,
                                 batch_size, embed_size, gamma):
    head = DummyHead(feature_size)
    q_func = DiscreteFQFQFunction(head, action_size, n_quantiles, embed_size)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)

    action = torch.randint(high=action_size, size=(batch_size, ))
    target = q_func.compute_target(x, action)
    assert target.shape == (batch_size, n_quantiles)

    # TODO: check quantile huber loss
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.randint(action_size, size=(batch_size, ))
    rew_tp1 = torch.rand(batch_size, 1)
    q_tp1 = torch.rand(batch_size, n_quantiles)
    loss = q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1)

    # check layer connection
    check_parameter_updates(q_func, (x, ))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('embed_size', [64])
@pytest.mark.parametrize('gamma', [0.99])
def test_continuous_fqf_q_function(feature_size, action_size, n_quantiles,
                                   batch_size, embed_size, gamma):
    head = DummyHead(feature_size, action_size)
    q_func = ContinuousFQFQFunction(head, n_quantiles, embed_size)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)

    target = q_func.compute_target(x, action)
    assert target.shape == (batch_size, n_quantiles)

    # TODO: check quantile huber loss
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.randint(action_size, size=(batch_size, ))
    rew_tp1 = torch.rand(batch_size, 1)
    q_tp1 = torch.rand(batch_size, n_quantiles)
    loss = q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1)

    # check layer connection
    check_parameter_updates(q_func, (x, action))


def ref_huber_loss(a, b):
    abs_diff = np.abs(a - b).reshape((-1, ))
    l2_diff = ((a - b)**2).reshape((-1, ))
    huber_diff = np.zeros_like(abs_diff)
    huber_diff[abs_diff < 1.0] = 0.5 * l2_diff[abs_diff < 1.0]
    huber_diff[abs_diff >= 1.0] = abs_diff[abs_diff >= 1.0] - 0.5
    return np.mean(huber_diff)


def filter_by_action(value, action, action_size):
    act_one_hot = np.identity(action_size)[np.reshape(action, (-1, ))]
    return (value * act_one_hot).sum(axis=1)


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('gamma', [0.99])
def test_discrete_q_function(feature_size, action_size, batch_size, gamma):
    head = DummyHead(feature_size)
    q_func = DiscreteQFunction(head, action_size)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)

    # check compute_target
    action = torch.randint(high=action_size, size=(batch_size, ))
    target = q_func.compute_target(x, action)
    assert target.shape == (batch_size, 1)
    assert torch.allclose(y[torch.arange(batch_size), action], target.view(-1))

    # check td calculation
    q_tp1 = np.random.random((batch_size, 1))
    rew_tp1 = np.random.random((batch_size, 1))
    target = rew_tp1 + gamma * q_tp1

    obs_t = torch.rand(batch_size, feature_size)
    act_t = np.random.randint(action_size, size=(batch_size, 1))
    q_t = filter_by_action(q_func(obs_t).detach().numpy(), act_t, action_size)
    ref_loss = ref_huber_loss(q_t.reshape((-1, 1)), target)

    loss = q_func.compute_error(obs_t,
                                torch.tensor(act_t, dtype=torch.int64),
                                torch.tensor(rew_tp1, dtype=torch.float32),
                                torch.tensor(q_tp1, dtype=torch.float32),
                                gamma=gamma)

    assert np.allclose(loss.detach().numpy(), ref_loss)

    # check layer connection
    check_parameter_updates(q_func, (x, ))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('ensemble_size', [5])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('embed_size', [64])
@pytest.mark.parametrize('bootstrap', [False, True])
def test_ensemble_discrete_q_function(feature_size, action_size, batch_size,
                                      gamma, ensemble_size, q_func_type,
                                      n_quantiles, embed_size, bootstrap):
    q_funcs = []
    for _ in range(ensemble_size):
        head = DummyHead(feature_size)
        if q_func_type == 'mean':
            q_func = DiscreteQFunction(head, action_size)
        elif q_func_type == 'qr':
            q_func = DiscreteQRQFunction(head, action_size, n_quantiles)
        elif q_func_type == 'iqn':
            q_func = DiscreteIQNQFunction(head, action_size, n_quantiles,
                                          embed_size)
        elif q_func_type == 'fqf':
            q_func = DiscreteFQFQFunction(head, action_size, n_quantiles,
                                          embed_size)
        q_funcs.append(q_func)
    q_func = EnsembleDiscreteQFunction(q_funcs, bootstrap)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    values = q_func(x, 'none')
    assert values.shape == (ensemble_size, batch_size, action_size)

    # check compute_target
    action = torch.randint(high=action_size, size=(batch_size, ))
    target = q_func.compute_target(x, action)
    if q_func_type == 'mean':
        assert target.shape == (batch_size, 1)
        min_values = values.min(dim=0).values
        assert torch.allclose(min_values[torch.arange(batch_size), action],
                              target.view(-1))
    else:
        assert target.shape == (batch_size, n_quantiles)

    # check reductions
    if q_func_type != 'iqn':
        assert torch.allclose(values.min(dim=0).values, q_func(x, 'min'))
        assert torch.allclose(values.max(dim=0).values, q_func(x, 'max'))
        assert torch.allclose(values.mean(dim=0), q_func(x, 'mean'))

    # check td computation
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.randint(0,
                          action_size,
                          size=(batch_size, 1),
                          dtype=torch.int64)
    rew_tp1 = torch.rand(batch_size, 1)
    if q_func_type == 'mean':
        q_tp1 = torch.rand(batch_size, 1)
    else:
        q_tp1 = torch.rand(batch_size, n_quantiles)
    ref_td_sum = 0.0
    for i in range(ensemble_size):
        f = q_func.q_funcs[i]
        ref_td_sum += f.compute_error(obs_t, act_t, rew_tp1, q_tp1, gamma)
    loss = q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1, gamma)
    if bootstrap:
        assert not torch.allclose(ref_td_sum, loss)
    elif q_func_type != 'iqn':
        assert torch.allclose(ref_td_sum, loss)

    # check layer connection
    check_parameter_updates(q_func, (x, 'mean'))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('gamma', [0.99])
def test_continuous_q_function(feature_size, action_size, batch_size, gamma):
    head = DummyHead(feature_size, action_size, concat=True)
    q_func = ContinuousQFunction(head)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)

    # check compute_target
    target = q_func.compute_target(x, action)
    assert target.shape == (batch_size, 1)
    assert (target == y).all()

    # check td calculation
    q_tp1 = np.random.random((batch_size, 1))
    rew_tp1 = np.random.random((batch_size, 1))
    target = rew_tp1 + gamma * q_tp1

    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.rand(batch_size, action_size)
    q_t = q_func(obs_t, act_t).detach().numpy()
    ref_loss = ((q_t - target)**2).mean()

    loss = q_func.compute_error(obs_t, act_t,
                                torch.tensor(rew_tp1, dtype=torch.float32),
                                torch.tensor(q_tp1, dtype=torch.float32),
                                gamma)

    assert np.allclose(loss.detach().numpy(), ref_loss)

    # check layer connection
    check_parameter_updates(q_func, (x, action))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('ensemble_size', [5])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('embed_size', [64])
@pytest.mark.parametrize('bootstrap', [False, True])
def test_ensemble_continuous_q_function(feature_size, action_size, batch_size,
                                        gamma, ensemble_size, q_func_type,
                                        n_quantiles, embed_size, bootstrap):
    q_funcs = []
    for _ in range(ensemble_size):
        head = DummyHead(feature_size, action_size, concat=True)
        if q_func_type == 'mean':
            q_func = ContinuousQFunction(head)
        elif q_func_type == 'qr':
            q_func = ContinuousQRQFunction(head, n_quantiles)
        elif q_func_type == 'iqn':
            q_func = ContinuousIQNQFunction(head, n_quantiles, embed_size)
        elif q_func_type == 'fqf':
            q_func = ContinuousFQFQFunction(head, n_quantiles, embed_size)
        q_funcs.append(q_func)

    q_func = EnsembleContinuousQFunction(q_funcs, bootstrap)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    values = q_func(x, action, 'none')
    assert values.shape == (ensemble_size, batch_size, 1)

    # check compute_target
    target = q_func.compute_target(x, action)
    if q_func_type == 'mean':
        assert target.shape == (batch_size, 1)
        min_values = values.min(dim=0).values
        assert (target == min_values).all()
    else:
        assert target.shape == (batch_size, n_quantiles)

    # check reductions
    if q_func_type != 'iqn':
        assert torch.allclose(values.min(dim=0)[0], q_func(x, action, 'min'))
        assert torch.allclose(values.max(dim=0)[0], q_func(x, action, 'max'))
        assert torch.allclose(values.mean(dim=0), q_func(x, action, 'mean'))

    # check td computation
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.rand(batch_size, action_size)
    rew_tp1 = torch.rand(batch_size, 1)
    if q_func_type == 'mean':
        q_tp1 = torch.rand(batch_size, 1)
    else:
        q_tp1 = torch.rand(batch_size, n_quantiles)
    ref_td_sum = 0.0
    for i in range(ensemble_size):
        f = q_func.q_funcs[i]
        ref_td_sum += f.compute_error(obs_t, act_t, rew_tp1, q_tp1, gamma)
    loss = q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1, gamma)
    if bootstrap:
        assert not torch.allclose(ref_td_sum, loss)
    elif q_func_type != 'iqn':
        assert torch.allclose(ref_td_sum, loss)

    # check layer connection
    check_parameter_updates(q_func, (x, action, 'mean'))


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [3])
@pytest.mark.parametrize('n_ensembles', [2])
@pytest.mark.parametrize('batch_size', [100])
@pytest.mark.parametrize('n_quantiles', [32])
@pytest.mark.parametrize('n_actions', [10])
@pytest.mark.parametrize('lam', [0.75])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr'])
def test_compute_max_with_n_actions(observation_shape, action_size,
                                    n_ensembles, batch_size, n_quantiles,
                                    n_actions, lam, q_func_type):
    q_func = create_continuous_q_function(observation_shape,
                                          action_size,
                                          n_ensembles,
                                          n_quantiles,
                                          q_func_type=q_func_type)
    x = torch.rand(batch_size, *observation_shape)
    actions = torch.rand(batch_size, n_actions, action_size)

    y = compute_max_with_n_actions(x, actions, q_func, lam)

    if q_func_type == 'mean':
        assert y.shape == (batch_size, 1)
    else:
        assert y.shape == (batch_size, n_quantiles)
