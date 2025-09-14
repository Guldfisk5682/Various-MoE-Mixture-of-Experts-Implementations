import jax
from jax import random
from flax import linen as nn
import jax.numpy as jnp


class Expert(nn.Module):
    dim: int
    hidden_dim: int
    dropout_rate: float = 0.1

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_dim, use_bias=False)
        self.dense2 = nn.Dense(self.hidden_dim, use_bias=False)
        self.dense3 = nn.Dense(self.dim, use_bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, deterministic: bool = True):
        x = nn.silu(self.dense1(x)) * self.dense2(x)
        x = self.dense3(x)
        return self.dropout(x, deterministic)


class SMoE(nn.Module):
    dim: int
    hidden_dim: int
    num_experts: int
    dropout_rate: float
    alpha: float
    top_k: int = 2
    capacity_factor: float = 1.25
    epsilon: float = 1e-8

    def setup(self):
        self.gate = nn.Dense(self.num_experts)
        self.experts = [
            Expert(self.dim, self.hidden_dim, self.dropout_rate)
            for i in range(self.num_experts)
        ]

    def __call__(self, x, deterministic: bool):
        b, s, d = x.shape
        x = x.reshape(-1, d)  # (b * s, d)
        num_tokens = x.shape[0]  # (b * s)
        capacity = int(num_tokens / self.num_experts * self.capacity_factor)

        gate_logits = self.gate(x)  # (num_tokens, num_experts)

        if not deterministic:
            rng = self.make_rng("dropout")
            noise = random.normal(rng, gate_logits.shape)
            gate_logits = gate_logits + noise

        top_k_gate_logits, top_k_indices = jax.lax.top_k(
            gate_logits, k=self.top_k, axis=1
        )  # (num_tokens, k)

        top_k_label_list = jax.nn.one_hot(
            top_k_indices, num_classes=self.num_experts
        )  # (num_tokens, k, num_experts)
        expert_label = jnp.sum(
            top_k_label_list, axis=1, keepdims=False
        )  # (num_tokens, num_experts) 每一行token喜欢的前k个专家为1
        cumsum_expert = jnp.cumsum(
            expert_label, axis=0
        )  # (num_tokens, num_experts) 最后一行代表每个expert有多少token
        token_for_expert_seq = (
            cumsum_expert * expert_label
        )  # (num_tokens, num_experts) 现在每个位置代表的当前token属于对应专家的第几个token

        valid_routing_mask = jnp.where(
            (token_for_expert_seq > 0) & (token_for_expert_seq <= capacity), True, False
        )  # (num_tokens, num_experts) topk需要知道一个 token 的哪一个专家选择是有效的

        routing_weights = jax.nn.softmax(
            top_k_gate_logits, axis=-1
        )  # (num_tokens, k) 组合专家输出时只关心被选中的k个专家权重
        all_expert_output = jnp.zeros((num_tokens, self.num_experts, d))

        for i in range(self.num_experts):
            is_expert_i_chosen = jnp.any(
                i == top_k_indices, axis=1, keepdims=False
            )  # 如果专家i是该token的top-k选择之一，则为True
            combined_mask = (
                valid_routing_mask[:, i] & is_expert_i_chosen
            )  # (num_valid_tokens_for_expert_i) 判断token所属专家并且判断他们是否有效
            if combined_mask.any():
                expert_input = x[combined_mask]
                expert_output_i = self.experts[i](
                    expert_input, deterministic
                )  # (num_valid_tokens_for_expert_i, dim)
                all_expert_output = all_expert_output.at[combined_mask, i, :].set(
                    expert_output_i
                )

        indices_for_gather = jnp.expand_dims(
            top_k_indices, axis=-1
        )  # (num_tokens, k, 1)
        selected_expert_outputs = jnp.take_along_axis(
            all_expert_output, indices_for_gather, axis=1
        )  # (num_tokens, k, dim) 对于每个 token，从 all_expert_output中，根据 top_k_indices提供的专家索引，
        #  挑选出k个对应的专家输出
        expanded_routing_weights = jnp.expand_dims(routing_weights, axis=-1)
        weighted_k_output = (
            expanded_routing_weights * selected_expert_outputs
        )  # (num_tokens, k, dim)
        weighted_output = jnp.sum(
            weighted_k_output, axis=1, keepdims=False
        )  # (num_tokens, dim)

        # aux_loss
        token_per_expert = jnp.sum(expert_label, axis=0, keepdims=False)
        all_experts_weights = jax.nn.softmax(
            gate_logits, axis=-1
        )  # (num_tokens, num_experts) 计算负载均衡损失时要训练整个门控网络，gate要学会如何均匀分配token给num_experts个专家

        f_i = token_per_expert / num_tokens
        p_i = jnp.mean(all_experts_weights, axis=0, keepdims=False)
        aux_loss = self.alpha * self.num_experts * jnp.sum(f_i * p_i)

        self.sow("losses", "aux_loss", aux_loss)

        return weighted_output.reshape(b, s, d)
