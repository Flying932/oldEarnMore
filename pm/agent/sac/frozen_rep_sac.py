import torch
import numpy as np

from copy import deepcopy
from torch import Tensor
from tqdm.auto import tqdm
from types import MethodType
from typing import Tuple
from torch.nn.utils import clip_grad_norm_
from einops import rearrange

from pm.registry import AGENT
from pm.registry import NET
from pm.registry import CRITERION
from pm.registry import OPTIMIZER
from pm.registry import SCHEDULER
from pm.utils import (
    ReplayBuffer,
    build_storage,
    get_optim_param,
    get_action_wrapper,
    forward_action_wrapper,
    get_action_logprob_wrapper,
)
from pm.metrics import ARR, VOL, DD, MDD, SR, CR, SOR


class _NoOpScheduler:
    def step_update(self, *args, **kwargs):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, _state_dict):
        return None


@AGENT.register_module()
class AgentSACFrozenRep:
    def __init__(
        self,
        act_lr: float = None,
        cri_lr: float = None,
        alpha_lr: float = None,
        adapter_lr: float = None,
        adapter_actor_lr: float = None,
        rep_net: dict = None,
        adapter_net: dict = None,
        act_net: dict = None,
        cri_net: dict = None,
        criterion: dict = None,
        optimizer: dict = None,
        scheduler: dict = None,
        if_use_per: bool = False,
        num_envs: int = 1,
        max_step: int = 1e4,
        transition_shape: dict = None,
        gamma: float = 0.99,
        reward_scale: int = 2 ** 0,
        repeat_times: float = 1.0,
        batch_size: int = 512,
        clip_grad_norm: float = 3.0,
        soft_update_tau: float = 0.0,
        state_value_tau: float = 5e-3,
        device: torch.device = torch.device("cpu"),
        action_wrapper_method: str = "reweight",
        T: float = 1.0,
        pretrain_path: str = None,
        pretrain_strict: bool = False,
        freeze_rep: bool = True,
        actor_trainable: bool = False,
        adapter_actor_grad: bool = False,
        actor_on_adapter_weight: float = 0.0,
        if_use_tqdm: bool = True,
    ):
        self.act_lr = act_lr
        self.cri_lr = cri_lr
        self.alpha_lr = alpha_lr
        self.adapter_lr = adapter_lr
        self.adapter_actor_lr = adapter_actor_lr

        self.num_envs = num_envs
        self.device = torch.device("cpu") if not device else device
        self.max_step = max_step
        self.if_use_per = if_use_per
        self.transition_shape = transition_shape
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.repeat_times = repeat_times
        self.batch_size = batch_size
        self.clip_grad_norm = clip_grad_norm
        self.soft_update_tau = soft_update_tau
        self.state_value_tau = state_value_tau
        self.last_state = None
        self.if_use_tqdm = if_use_tqdm

        self.actor_trainable = actor_trainable
        self.adapter_actor_grad = adapter_actor_grad
        self.actor_on_adapter_weight = actor_on_adapter_weight
        self.freeze_rep = freeze_rep

        self.rep = NET.build(rep_net).to(self.device)
        self.adapter = NET.build(adapter_net).to(self.device)
        self.act = NET.build(act_net).to(self.device)
        self.cri = NET.build(cri_net).to(self.device) if cri_net else self.act
        self.act_target = deepcopy(self.act).to(self.device)
        self.cri_target = deepcopy(self.cri).to(self.device)
        for param in self.act_target.parameters():
            param.requires_grad_(False)
        for param in self.cri_target.parameters():
            param.requires_grad_(False)

        self._load_pretrain_rep(pretrain_path, pretrain_strict)
        self._freeze_rep_if_needed()
        self._configure_trainable_flags()

        if optimizer is None:
            optimizer = dict(type="AdamW", params=None, lr=1e-4)
        if scheduler is None:
            scheduler = dict(type="MultiStepLRScheduler", optimizer=None, multi_steps=[1], t_initial=1, decay_t=1)
        if criterion is None:
            criterion = dict(type="MSELoss", reduction="mean")

        self._build_optimizers(optimizer)
        self._build_schedulers(scheduler)

        self.get_action = get_action_wrapper(self.act.get_action, method=action_wrapper_method, T=T)
        self.forward_action = forward_action_wrapper(self.act.forward, method=action_wrapper_method, T=T)
        self.get_action_logprob = get_action_logprob_wrapper(
            self.act.get_action_logprob,
            method=action_wrapper_method,
            T=T,
        )

        if self.if_use_per:
            criterion.update(dict(reduction="none"))
            self.get_obj_critic = self.get_obj_critic_per
        else:
            criterion.update(dict(reduction="mean"))
            self.get_obj_critic = self.get_obj_critic_raw
        self.criterion = CRITERION.build(criterion)

        self.target_entropy = transition_shape["action"]["shape"][-1]
        self.global_step = 0

    def _load_pretrain_rep(self, pretrain_path: str, strict: bool):
        if not pretrain_path:
            return
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "rep" in checkpoint:
            checkpoint = checkpoint["rep"]
        self.rep.load_state_dict(checkpoint, strict=strict)
        print(f"loaded pretrain rep from {pretrain_path}")

    def _freeze_rep_if_needed(self):
        if not self.freeze_rep:
            return
        self.rep.eval()
        for param in self.rep.parameters():
            param.requires_grad = False

    def _configure_trainable_flags(self):
        # Adapter/Critic are always trainable in SPF Stage-3.
        for param in self.adapter.parameters():
            param.requires_grad_(True)
        for param in self.cri.parameters():
            param.requires_grad_(True)

        # Actor is frozen by default and can be enabled by ablation.
        for param in self.act.parameters():
            param.requires_grad_(bool(self.actor_trainable))

    def _build_optimizers(self, optimizer_cfg: dict):
        self.alpha_log = torch.tensor(
            (-1,),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )

        cri_optimizer = deepcopy(optimizer_cfg)
        effective_adapter_lr = self.cri_lr if self.adapter_lr is None else self.adapter_lr
        cri_optimizer.update(
            dict(
                params=[
                    dict(params=list(self.cri.parameters()), lr=self.cri_lr),
                    dict(params=list(self.adapter.parameters()), lr=effective_adapter_lr),
                ],
                lr=self.cri_lr,
            )
        )
        self.cri_optimizer = OPTIMIZER.build(cri_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        self.act_optimizer = None
        if self.actor_trainable:
            act_optimizer = deepcopy(optimizer_cfg)
            act_optimizer.update(dict(params=self.act.parameters(), lr=self.act_lr))
            self.act_optimizer = OPTIMIZER.build(act_optimizer)
            self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)

        alpha_optimizer = deepcopy(optimizer_cfg)
        effective_alpha_lr = optimizer_cfg["lr"] if self.alpha_lr is None else self.alpha_lr
        alpha_optimizer.update(dict(params=(self.alpha_log,), lr=effective_alpha_lr))
        self.alpha_optimizer = OPTIMIZER.build(alpha_optimizer)
        self.alpha_optimizer.parameters = MethodType(get_optim_param, self.alpha_optimizer)

        self.adapter_actor_optimizer = None
        if self.adapter_actor_grad and self.actor_on_adapter_weight > 0.0:
            adapter_actor_optimizer = deepcopy(optimizer_cfg)
            if self.adapter_actor_lr is None:
                effective_adapter_actor_lr = self.act_lr if self.act_lr is not None else effective_adapter_lr
            else:
                effective_adapter_actor_lr = self.adapter_actor_lr
            adapter_actor_optimizer.update(
                dict(params=self.adapter.parameters(), lr=effective_adapter_actor_lr)
            )
            self.adapter_actor_optimizer = OPTIMIZER.build(adapter_actor_optimizer)
            self.adapter_actor_optimizer.parameters = MethodType(get_optim_param, self.adapter_actor_optimizer)

    def _build_schedulers(self, scheduler_cfg: dict):
        cri_scheduler = deepcopy(scheduler_cfg)
        cri_scheduler.update(dict(optimizer=self.cri_optimizer))
        self.cri_scheduler = SCHEDULER.build(cri_scheduler)

        self.act_scheduler = _NoOpScheduler()
        if self.act_optimizer is not None:
            act_scheduler = deepcopy(scheduler_cfg)
            act_scheduler.update(dict(optimizer=self.act_optimizer))
            self.act_scheduler = SCHEDULER.build(act_scheduler)

        alpha_scheduler = deepcopy(scheduler_cfg)
        alpha_scheduler.update(dict(optimizer=self.alpha_optimizer))
        self.alpha_scheduler = SCHEDULER.build(alpha_scheduler)

        self.adapter_actor_scheduler = _NoOpScheduler()
        if self.adapter_actor_optimizer is not None:
            adapter_actor_scheduler = deepcopy(scheduler_cfg)
            adapter_actor_scheduler.update(dict(optimizer=self.adapter_actor_optimizer))
            self.adapter_actor_scheduler = SCHEDULER.build(adapter_actor_scheduler)

    def _forward_rep(self, states: Tensor) -> Tensor:
        if self.freeze_rep:
            with torch.no_grad():
                rep_state, _, _ = self.rep.forward_state(states, if_mask=False)
        else:
            rep_state, _, _ = self.rep.forward_state(states, if_mask=False)
        return rep_state

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def optimizer_update(self, optimizer: torch.optim.Optimizer, objective: Tensor, lr_scheduler=None, step=None):
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        params = []
        for group in optimizer.param_groups:
            params.extend(group["params"])
        clip_grad_norm_(parameters=params, max_norm=self.clip_grad_norm)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step_update(step)

    def get_state_dict(self):
        state_dict = {
            "alpha_log": self.alpha_log,
            "rep": self.rep.state_dict(),
            "adapter": self.adapter.state_dict(),
            "act": self.act.state_dict(),
            "cri": self.cri.state_dict(),
            "act_target": self.act_target.state_dict(),
            "cri_target": self.cri_target.state_dict(),
            "cri_optimizer": self.cri_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "cri_scheduler": self.cri_scheduler.state_dict(),
            "alpha_scheduler": self.alpha_scheduler.state_dict(),
            "actor_trainable": self.actor_trainable,
            "adapter_actor_grad": self.adapter_actor_grad,
            "actor_on_adapter_weight": self.actor_on_adapter_weight,
        }
        if self.act_optimizer is not None:
            state_dict["act_optimizer"] = self.act_optimizer.state_dict()
            state_dict["act_scheduler"] = self.act_scheduler.state_dict()
        if self.adapter_actor_optimizer is not None:
            state_dict["adapter_actor_optimizer"] = self.adapter_actor_optimizer.state_dict()
            state_dict["adapter_actor_scheduler"] = self.adapter_actor_scheduler.state_dict()
        return state_dict

    def set_state_dict(self, state_dict):
        if "alpha_log" in state_dict:
            with torch.no_grad():
                self.alpha_log.copy_(state_dict["alpha_log"].to(self.device))
        if "rep" in state_dict:
            self.rep.load_state_dict(state_dict["rep"])
        if "adapter" in state_dict:
            self.adapter.load_state_dict(state_dict["adapter"])
        if "act" in state_dict:
            self.act.load_state_dict(state_dict["act"])
        if "cri" in state_dict:
            self.cri.load_state_dict(state_dict["cri"])
        if "act_target" in state_dict:
            self.act_target.load_state_dict(state_dict["act_target"])
        if "cri_target" in state_dict:
            self.cri_target.load_state_dict(state_dict["cri_target"])

        if "cri_optimizer" in state_dict:
            self.cri_optimizer.load_state_dict(state_dict["cri_optimizer"])
        if "alpha_optimizer" in state_dict:
            self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        if self.act_optimizer is not None and "act_optimizer" in state_dict:
            self.act_optimizer.load_state_dict(state_dict["act_optimizer"])
        if self.adapter_actor_optimizer is not None and "adapter_actor_optimizer" in state_dict:
            self.adapter_actor_optimizer.load_state_dict(state_dict["adapter_actor_optimizer"])

        if "cri_scheduler" in state_dict:
            self.cri_scheduler.load_state_dict(state_dict["cri_scheduler"])
        if "alpha_scheduler" in state_dict:
            self.alpha_scheduler.load_state_dict(state_dict["alpha_scheduler"])
        if self.act_optimizer is not None and "act_scheduler" in state_dict:
            self.act_scheduler.load_state_dict(state_dict["act_scheduler"])
        if self.adapter_actor_optimizer is not None and "adapter_actor_scheduler" in state_dict:
            self.adapter_actor_scheduler.load_state_dict(state_dict["adapter_actor_scheduler"])

    def explore_env(self, env, horizon_len: int) -> Tuple[Tensor, ...]:
        states = build_storage(
            (horizon_len, *self.transition_shape["state"]["shape"]),
            self.transition_shape["state"]["type"],
            self.device,
        )
        actions = build_storage(
            (horizon_len, *self.transition_shape["action"]["shape"]),
            self.transition_shape["action"]["type"],
            self.device,
        )
        rewards = build_storage(
            (horizon_len, *self.transition_shape["reward"]["shape"]),
            self.transition_shape["reward"]["type"],
            self.device,
        )
        dones = build_storage(
            (horizon_len, *self.transition_shape["done"]["shape"]),
            self.transition_shape["done"]["type"],
            self.device,
        )
        next_states = build_storage(
            (horizon_len, *self.transition_shape["next_state"]["shape"]),
            self.transition_shape["next_state"]["type"],
            self.device,
        )

        state = self.last_state

        for t in range(horizon_len):
            b, e, n, d, f = state.shape
            flat_state = rearrange(state, "b e n d f -> (b e) n d f", b=b, e=e)
            rep_state = self._forward_rep(flat_state)
            adapted_state = self.adapter(rep_state)
            action = self.get_action(adapted_state)

            states[t] = state

            ary_action = action.detach().cpu().numpy()
            next_state, reward, done, _ = env.step(ary_action)
            ary_state = env.reset() if np.sum(done) > 0 else next_state

            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0)
            done = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(0)

            actions[t] = action.unsqueeze(0)
            rewards[t] = reward
            dones[t] = done
            next_states[t] = state

        self.last_state = state
        rewards *= self.reward_scale
        dones = dones.type(torch.float32)
        return states, actions, rewards, dones, next_states

    def get_obj_critic_raw(self, buffer, batch_size: int):
        with torch.no_grad():
            states, actions, rewards, dones, next_states = buffer.sample(batch_size)

            if states.device != self.device:
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                next_states = next_states.to(self.device)

            rep_next_states = self._forward_rep(next_states)
            adapted_next_states = self.adapter(rep_next_states)
            next_as, next_logprobs = self.get_action_logprob(adapted_next_states)
            next_qs = self.cri_target.get_q_min(adapted_next_states, next_as)
            alpha = self.alpha_log.exp().detach()
            q_labels = rewards + (1.0 - dones) * self.gamma * (next_qs - next_logprobs * alpha)

            rep_states = self._forward_rep(states)

        adapted_states = self.adapter(rep_states)
        q1, q2 = self.cri.get_q1_q2(adapted_states, actions)
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        return obj_critic, rep_states, adapted_states

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int):
        with torch.no_grad():
            states, actions, rewards, dones, next_states, is_weights, is_indices = buffer.sample_for_per(batch_size)

            if states.device != self.device:
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                next_states = next_states.to(self.device)
                is_weights = is_weights.to(self.device)
                is_indices = is_indices.to(self.device)

            rep_next_states = self._forward_rep(next_states)
            adapted_next_states = self.adapter(rep_next_states)
            next_as, next_logprobs = self.get_action_logprob(adapted_next_states)
            next_qs = self.cri_target.get_q_min(adapted_next_states, next_as)
            alpha = self.alpha_log.exp().detach()
            q_labels = rewards + (1.0 - dones) * self.gamma * (next_qs - next_logprobs * alpha)
            rep_states = self._forward_rep(states)

        adapted_states = self.adapter(rep_states)
        q1, q2 = self.cri.get_q1_q2(adapted_states, actions)
        td_errors = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        obj_critic = (td_errors * is_weights).mean()
        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, rep_states, adapted_states

    def _adapter_update_from_actor(self, rep_states: Tensor, alpha: Tensor):
        if self.adapter_actor_optimizer is None:
            return torch.zeros((), device=self.device)

        actor_requires_grad = [param.requires_grad for param in self.act.parameters()]
        for param in self.act.parameters():
            param.requires_grad_(False)

        adapted_states = self.adapter(rep_states)
        action_pg, log_prob = self.get_action_logprob(adapted_states)
        q_value_pg = self.cri_target(adapted_states, action_pg).mean()
        obj_actor = (q_value_pg - log_prob * alpha).mean()
        weighted_obj = -self.actor_on_adapter_weight * obj_actor
        self.optimizer_update(
            self.adapter_actor_optimizer,
            weighted_obj,
            self.adapter_actor_scheduler,
            self.global_step,
        )

        for param, original_flag in zip(self.act.parameters(), actor_requires_grad):
            param.requires_grad_(original_flag)
        return obj_actor.detach()

    def update_net(self, buffer: ReplayBuffer) -> dict:
        obj_critics = torch.zeros((), device=self.device)
        obj_actors = torch.zeros((), device=self.device)
        alphas = torch.zeros((), device=self.device)
        adapter_actor_objs = torch.zeros((), device=self.device)

        update_times = int(self.repeat_times)
        assert update_times >= 1
        updates = range(update_times)
        if self.if_use_tqdm:
            updates = tqdm(
                updates,
                bar_format="update net batch " + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}",
            )

        for _ in updates:
            obj_critic, rep_states, adapted_states = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.detach()
            self.optimizer_update(self.cri_optimizer, obj_critic, self.cri_scheduler, self.global_step)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            actor_states_detached = adapted_states.detach()
            action_pg, log_prob = self.get_action_logprob(actor_states_detached)
            obj_alpha = (self.alpha_log * (self.target_entropy - log_prob).detach()).mean()
            self.optimizer_update(self.alpha_optimizer, obj_alpha, self.alpha_scheduler, self.global_step)

            alpha = self.alpha_log.exp().detach()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            alphas += alpha.mean()

            q_value_pg = self.cri_target(actor_states_detached, action_pg).mean()
            obj_actor = (q_value_pg - log_prob * alpha).mean()
            obj_actors += obj_actor.detach()

            if self.actor_trainable and self.act_optimizer is not None:
                self.optimizer_update(self.act_optimizer, -obj_actor, self.act_scheduler, self.global_step)

            if self.adapter_actor_grad and self.actor_on_adapter_weight > 0.0:
                adapter_actor_obj = self._adapter_update_from_actor(rep_states, alpha)
                adapter_actor_objs += adapter_actor_obj

            self.global_step += 1

        stats = {
            "obj_critics": (obj_critics / update_times).item(),
            "obj_actors": (obj_actors / update_times).item(),
            "alphas": (alphas / update_times).item(),
            "cri_lr": self.cri_optimizer.param_groups[0]["lr"],
            "alpha_lr": self.alpha_optimizer.param_groups[0]["lr"],
            "adapter_actor_objs": (adapter_actor_objs / update_times).item(),
        }
        if self.act_optimizer is not None:
            stats["act_lr"] = self.act_optimizer.param_groups[0]["lr"]
        return stats

    def validate_net(self, environment):
        state = environment.reset()
        num_envs = environment.num_envs
        infos = {
            "portfolio_rets": [[] for _ in range(num_envs)],
            "portfolio_values": [[] for _ in range(num_envs)],
            "date": [[] for _ in range(num_envs)],
            "portfolios": [],
        }

        while True:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            b, e, n, d, f = state.shape
            flat_state = rearrange(state, "b e n d f -> (b e) n d f", b=b, e=e)
            rep_state = self._forward_rep(flat_state)
            adapted_state = self.adapter(rep_state)
            action = self.forward_action(adapted_state)
            ary_action = action.detach().cpu().numpy()
            infos["portfolios"].append(ary_action[0])
            state, reward, done, info = environment.step(ary_action)

            for i in range(num_envs):
                infos["portfolio_rets"][i].append(info[i]["portfolio_ret"])
                infos["portfolio_values"][i].append(info[i]["portfolio_value"])
                infos["date"][i].append(info[i]["date"])

            if np.sum(done) > 0:
                break

        metrics = {}
        for i in range(num_envs):
            rets = np.array(infos["portfolio_rets"][i])
            arr = ARR(rets)
            vol = VOL(rets)
            dd = DD(rets)
            mdd = MDD(rets)
            sr = SR(rets)
            cr = CR(rets, mdd)
            sor = SOR(rets, dd)
            metrics.update(
                {
                    f"ARR%_env{i}": arr * 100,
                    f"SR_env{i}": sr,
                    f"CR_env{i}": cr,
                    f"MDD%_env{i}": mdd * 100,
                    f"VOL_env{i}": vol,
                    f"DD_env{i}": dd,
                    f"SOR_env{i}": sor,
                }
            )
        return metrics, infos
