"""Reset is a state boundary, not an extra actuator/physics substep."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import BuiltinMotorActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg
from mjlab.viewer.viser.viewer import ViserPlayViewer


@pytest.fixture(params=["builtin", "custom"])
def env(request):
  cfg = cartpole_balance_env_cfg()
  cfg.seed = 812
  cfg.scene.num_envs = 3
  kwargs = dict(target_names_expr=("slider",), delay_min_lag=2, delay_max_lag=2)
  actuator = (
    BuiltinMotorActuatorCfg(effort_limit=10.0, **kwargs)
    if request.param == "builtin"
    else IdealPdActuatorCfg(stiffness=10.0, damping=1.0, **kwargs)
  )
  cfg.scene.entities["cartpole"].articulation = EntityArticulationInfoCfg(
    actuators=(actuator,)
  )
  result = ManagerBasedRlEnv(cfg, device=get_test_device())
  yield result
  result.close()


def _delay(env):
  delay = env.scene["cartpole"].actuators[0]._delay_buffer
  assert delay is not None
  return delay


@pytest.mark.parametrize("indices", [[], [0, 2], None])
def test_explicit_reset_does_not_evaluate_actuators(env, indices):
  env.reset()
  env.step(torch.ones(3, 1, device=env.device))
  delay = _delay(env)
  pushes = delay._buffer._num_pushes.clone()
  ctrl = env.sim.data.ctrl.clone()
  model_damping = env.sim.model.dof_damping.clone()
  ids = (
    None
    if indices is None
    else torch.tensor(indices, device=env.device, dtype=torch.long)
  )
  selected = slice(None) if ids is None else ids
  env.reset(env_ids=ids)
  pushes[selected] = 0
  ctrl[selected] = 0
  torch.testing.assert_close(delay._buffer._num_pushes, pushes, rtol=0, atol=0)
  torch.testing.assert_close(env.sim.data.ctrl[:], ctrl, rtol=0, atol=0)
  torch.testing.assert_close(
    env.sim.model.dof_damping[:], model_damping, rtol=0, atol=0
  )


def test_empty_reset_is_a_noop_including_rng_and_observation_history(env):
  env.reset()
  env.step(torch.zeros(3, 1, device=env.device))
  obs = env.obs_buf
  log = env.extras["log"]
  rng = torch.random.get_rng_state().clone()
  returned, _ = env.reset(env_ids=torch.empty(0, device=env.device, dtype=torch.long))
  assert returned is obs
  assert env.extras["log"] is log
  assert torch.equal(torch.random.get_rng_state(), rng)


def test_normal_steps_advance_delay_once_per_physics_substep(env):
  env.reset()
  delay = _delay(env)
  assert torch.count_nonzero(delay._buffer._num_pushes) == 0
  for iteration in range(1, 4):
    env.step(torch.ones(3, 1, device=env.device))
    torch.testing.assert_close(
      delay._buffer._num_pushes,
      torch.full(
        (3,), iteration * env.cfg.decimation, device=env.device, dtype=torch.long
      ),
    )


def test_auto_reset_does_not_add_a_control_tick(env):
  env.reset()
  env.step(torch.ones(3, 1, device=env.device))
  delay = _delay(env)
  pushes = delay._buffer._num_pushes.clone()
  env.episode_length_buf[0] = env.max_episode_length - 1
  _, _, _, truncated, _ = env.step(torch.ones(3, 1, device=env.device))
  assert truncated.tolist() == [True, False, False]
  pushes += env.cfg.decimation
  pushes[0] = 0
  torch.testing.assert_close(delay._buffer._num_pushes, pushes, rtol=0, atol=0)
  assert torch.count_nonzero(env.sim.data.ctrl[0]) == 0


@pytest.mark.parametrize("owner", ["entity", "scene"])
@pytest.mark.parametrize("indices", [[], [1], None])
def test_direct_reset_clears_only_owned_controls_without_compute(env, owner, indices):
  entity = env.scene["cartpole"]
  env.reset()
  env.scene.write_data_to_sim()
  delay = _delay(env)
  pushes = delay._buffer._num_pushes.clone()
  entity.write_ctrl_to_sim(torch.full((3, 1), 0.7, device=env.device))
  ctrl = env.sim.data.ctrl.clone()
  ids = (
    None
    if indices is None
    else torch.tensor(indices, device=env.device, dtype=torch.long)
  )
  target = entity if owner == "entity" else env.scene
  target.reset(ids)
  selected = slice(None) if ids is None else ids
  pushes[selected] = 0
  ctrl[selected] = 0
  torch.testing.assert_close(delay._buffer._num_pushes, pushes, rtol=0, atol=0)
  torch.testing.assert_close(env.sim.data.ctrl[:], ctrl, rtol=0, atol=0)


def test_low_level_control_write_does_not_tick_but_scene_write_does(env):
  env.reset()
  entity = env.scene["cartpole"]
  delay = _delay(env)
  before = delay._buffer._num_pushes.clone()
  entity.write_ctrl_to_sim(
    torch.tensor([[0.7]], device=env.device),
    env_ids=torch.tensor([1], device=env.device),
  )
  torch.testing.assert_close(delay._buffer._num_pushes, before)
  torch.testing.assert_close(
    env.sim.data.ctrl[:, 0], torch.tensor([0.0, 0.7, 0.0], device=env.device)
  )
  env.scene.write_data_to_sim()
  torch.testing.assert_close(delay._buffer._num_pushes, before + 1)


def test_gui_state_reset_does_not_evaluate_controls():
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    reset=Mock(),
    command_manager=SimpleNamespace(apply_gui_reset=Mock(return_value=True)),
    scene=SimpleNamespace(write_data_to_sim=Mock()),
    sim=SimpleNamespace(forward=Mock(), sense=Mock()),
  )
  viewer = object.__new__(ViserPlayViewer)
  viewer.env = SimpleNamespace(unwrapped=env)
  viewer._scene = SimpleNamespace(env_idx=1)
  viewer._sim_lock = nullcontext()
  viewer._pending_update_reasons = set()
  viewer._sync_ui_state = Mock()
  viewer._handle_gui_reset(all_envs=False)
  env.reset.assert_called_once()
  env.scene.write_data_to_sim.assert_not_called()
  env.sim.forward.assert_called_once()
  env.sim.sense.assert_called_once()


def test_entity_reset_leaves_other_entity_controls_untouched():
  cfg = cartpole_balance_env_cfg()
  cfg.scene.num_envs = 3
  cfg.scene.entities["other"] = deepcopy(cfg.scene.entities["cartpole"])
  env = ManagerBasedRlEnv(cfg, device=get_test_device())
  try:
    env.reset()
    env.sim.data.ctrl[:] = 0.7
    expected = env.sim.data.ctrl.clone()
    entity = env.scene["cartpole"]
    expected[1, entity.indexing.ctrl_ids] = 0.0
    entity.reset(torch.tensor([1], device=env.device))
    torch.testing.assert_close(env.sim.data.ctrl[:], expected, rtol=0, atol=0)
  finally:
    env.close()
