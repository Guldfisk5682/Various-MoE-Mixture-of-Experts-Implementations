import jax
from jax import random
from flax import linen as nn
import jax.numpy as jnp


class Expert(nn.Module):
    dim: int
    hidden_dim: int
    dropout_rate: float = 0.1

    def setup(self):
        self.dense1 = nn.Dense(self.dim, use_bias=False)
        self.dense2 = nn.Dense(self.hidden_dim, use_bias=False)
        self.dense3 = nn.Dense(self.hidden_dim, use_bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, deterministic: bool = True):
        x = nn.silu(self.dense2(x)) * self.dense3(x)
        x = self.dense1(x)
        return self.dropout(x, deterministic)


class SMoE(nn.Module):
    dim: int
    hidden_dim: int
    num_experts: int
    dropout_rate: float
    alpha: float
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

        token_for_expert = jnp.argmax(
            gate_logits, axis=1
        )  # (num_tokens) 每个值代表当前idx的token最喜爱的专家
        expert_label = jax.nn.one_hot(
            token_for_expert, num_classes=self.num_experts
        )  # (num_tokens, num_experts) 只有最喜爱的专家值才为1
        cumsum_expert = jnp.cumsum(
            expert_label, axis=0
        )  # (num_tokens, num_experts) 最后一行代表每个expert有多少token
        token_for_expert_seq = (
            cumsum_expert * expert_label
        )  # (num_tokens, num_experts) 现在每个位置代表的当前token属于对应专家的第几个token

        val_token_for_expert_list = jnp.where(
            (token_for_expert_seq > 0) & (token_for_expert_seq <= capacity), True, False
        )  # (num_tokens, num_experts) 只含有有效token
        val_token_for_expert = jnp.any(
            val_token_for_expert_list, axis=1, keepdims=False
        )  # (num_tokens)

        weights = jax.nn.softmax(gate_logits, axis=-1)
        expert_output = jnp.zeros_like(x)

        for i in range(self.num_experts):
            combined_mask = val_token_for_expert & (
                i == token_for_expert
            )  # (num_valid_tokens_for_expert_i) 判断token所属专家并且判断他们是否有效
            if combined_mask.any():
                expert_input = x[combined_mask]
                expert_output_i = self.experts[i](
                    expert_input, deterministic
                )  # (num_valid_tokens_for_expert_i, dim)
                expert_output = expert_output.at[combined_mask].set(expert_output_i)

        gate_weights = jnp.sum(
            weights * expert_label, axis=1, keepdims=True
        )  # (num_tokens, 1) 只用被选中的那个专家的门控权重来缩放其输出
        weighted_output = gate_weights * expert_output

        # auxiliary loss
        token_per_expert = jnp.sum(
            expert_label, axis=0, keepdims=False
        )  # 这里使用gate分配的token计算损失，因为希望它能做出更均衡的路由提议

        f_i = token_per_expert / num_tokens  # (num_experts) 每个专家平均接受的token比例
        p_i = jnp.mean(
            weights, axis=0
        )  # (num_experts) 对所有被发送到专家的token，平均门控概率有多大
        aux_loss = self.alpha * self.num_experts * jnp.sum(f_i * p_i)
        self.sow("losses", "aux_loss", aux_loss)

        return weighted_output.reshape(b, s, d)


class SwitchTransformer(nn.Module):
    dim: int
    num_head: int
    hidden_dim: int
    num_experts: int
    dropout_rate: float
    alpha: float
    capacity_factor: float = 1.25
    epsilon: float = 1e-8

    def setup(self):
        self.attn = nn.MultiHeadAttention(
            qkv_features=self.dim,
            out_features=self.dim,
            num_heads=self.num_head,
            dropout_rate=self.dropout_rate,
            use_bias=False,
        )

        self.smoe = SMoE(
            dim=self.dim,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            capacity_factor=self.capacity_factor,
            epsilon=self.epsilon,
            dropout_rate=self.dropout_rate,
            alpha=self.alpha,
        )

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, deterministic):
        # x = (b,s,d)
        residual = x
        x = self.norm1(x)
        attn = self.attn(x, deterministic=deterministic)
        x = residual + self.dropout(attn, deterministic)

        residual = x
        x = self.norm2(x)
        x = self.smoe(x, deterministic)
        return residual + self.dropout(x, deterministic)


if __name__ == "__main__":
    master_key = random.key(42)
    params_key, dropout_key = jax.random.split(master_key)

    model = SwitchTransformer(
        dim=24,
        num_head=8,
        hidden_dim=96,
        num_experts=8,
        capacity_factor=1.25,
        epsilon=1e-8,
        dropout_rate=0.5,
        alpha=0.5,
    )
    dummy_input = random.normal(params_key, (1, 64, 24))
    variables = model.init(
        {"params": params_key, "dropout": dropout_key}, dummy_input, deterministic=False
    )
    model_params = variables["params"]

    param_shape = jax.tree_util.tree_map(lambda x: x.shape, model_params)
    # print(param_shape)

    model_output, mutated_vars = model.apply(
        variables,
        dummy_input,
        deterministic=True,
        mutable=["losses"],
        # rngs={"dropout": dropout_key},
    )

    print(f"模型输出形状:{model_output.shape}")
