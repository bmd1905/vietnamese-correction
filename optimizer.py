# # Copyright 2023 Google Research. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
# """TF2 implementation of the Lion optimizer."""

# import tensorflow.compat.v2 as tf


# class Lion(tf.keras.optimizers.legacy.Optimizer):
#   r"""Optimizer that implements the Lion algorithm."""

#   def __init__(self,
#                learning_rate=0.0001,
#                beta_1=0.9,
#                beta_2=0.99,
#                wd=0,
#                name='lion',
#                **kwargs):
#     """Construct a new Lion optimizer."""

#     super(Lion, self).__init__(name, **kwargs)
#     self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
#     self._set_hyper('beta_1', beta_1)
#     self._set_hyper('beta_2', beta_2)
#     self._set_hyper('wd', wd)

#   def _create_slots(self, var_list):
#     # Create slots for the first and second moments.
#     # Separate for-loops to respect the ordering of slot variables from v1.
#     for var in var_list:
#       self.add_slot(var, 'm')

#   def _prepare_local(self, var_device, var_dtype, apply_state):
#     super(Lion, self)._prepare_local(var_device, var_dtype, apply_state)

#     beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
#     beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
#     wd_t = tf.identity(self._get_hyper('wd', var_dtype))
#     lr = apply_state[(var_device, var_dtype)]['lr_t']
#     apply_state[(var_device, var_dtype)].update(
#         dict(
#             lr=lr,
#             beta_1_t=beta_1_t,
#             one_minus_beta_1_t=1 - beta_1_t,
#             beta_2_t=beta_2_t,
#             one_minus_beta_2_t=1 - beta_2_t,
#             wd_t=wd_t))

#   @tf.function(jit_compile=True)
#   def _resource_apply_dense(self, grad, var, apply_state=None):
#     var_device, var_dtype = var.device, var.dtype.base_dtype
#     coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
#                     self._fallback_apply_state(var_device, var_dtype))

#     m = self.get_slot(var, 'm')
#     var_t = var.assign_sub(
#         coefficients['lr_t'] *
#         (tf.math.sign(m * coefficients['beta_1_t'] +
#                       grad * coefficients['one_minus_beta_1_t']) +
#          var * coefficients['wd_t']))
#     with tf.control_dependencies([var_t]):
#       m.assign(m * coefficients['beta_2_t'] +
#                grad * coefficients['one_minus_beta_2_t'])

#   @tf.function(jit_compile=True)
#   def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
#     var_device, var_dtype = var.device, var.dtype.base_dtype
#     coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
#                     self._fallback_apply_state(var_device, var_dtype))

#     m = self.get_slot(var, 'm')
#     m_t = m.assign(m * coefficients['beta_1_t'])
#     m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
#     m_t = m_t.scatter_add(tf.IndexedSlices(m_scaled_g_values, indices))
#     var_t = var.assign_sub(coefficients['lr'] *
#                            (tf.math.sign(m_t) + var * coefficients['wd_t']))

#     with tf.control_dependencies([var_t]):
#       m_t = m_t.scatter_add(tf.IndexedSlices(-m_scaled_g_values, indices))
#       m_t = m_t.assign(m_t * coefficients['beta_2_t'] /
#                        coefficients['beta_1_t'])
#       m_scaled_g_values = grad * coefficients['one_minus_beta_2_t']
#       m_t.scatter_add(tf.IndexedSlices(m_scaled_g_values, indices))

#   def get_config(self):
#     config = super(Lion, self).get_config()
#     config.update({
#         'learning_rate': self._serialize_hyperparameter('learning_rate'),
#         'beta_1': self._serialize_hyperparameter('beta_1'),
#         'beta_2': self._serialize_hyperparameter('beta_2'),
#         'wd': self._serialize_hyperparameter('wd'),
#     })
#     return config

from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

# update functions

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.data.mul_(1 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1).sign_()
    p.add_(update, alpha = -lr)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)

# class

class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

        self.update_fn = update_fn

        if use_triton:
            from lion_pytorch.triton import update_fn as triton_update_fn
            self.update_fn = triton_update_fn

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2
                )

        return loss