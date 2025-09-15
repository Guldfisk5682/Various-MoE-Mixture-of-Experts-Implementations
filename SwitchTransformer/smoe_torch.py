import torch
from torch import nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.dense1 = nn.Linear(hidden_dim, dim, bias=False)
        self.dense2 = nn.Linear(dim, hidden_dim, bias=False)
        self.dense3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.silu(self.dense2(x)) * self.dense3(x)
        x = self.dense1(x)
        return self.dropout(x)


class SMoE(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        num_experts,
        dropout_rate,
        alpha,
        capacity_factor=1.25,
        epsilon=1e-8,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            [Expert(dim, hidden_dim, dropout_rate) for _ in range(num_experts)]
        )

    def forward(self, x, is_training: bool):
        b, s, d = x.shape
        x = x.reshape(-1, d)
        num_tokens = x.shape[0]
        capacity = int(num_tokens / self.num_experts * self.capacity_factor)

        gate_logits = self.gate(x)

        if is_training:
            noise = torch.normal(size=(num_tokens, self.num_experts), device=x.device)
            gate_logits += noise

        token_for_expert = torch.argmax(gate_logits, dim=1)  # (num_tokens)
        expert_label = F.one_hot(token_for_expert, num_classes=self.num_experts)
        cumsum_expert = torch.cumsum(expert_label, dim=0)
        token_for_expert_seq = cumsum_expert * expert_label

        val_token_for_expert_list = torch.where(
            (token_for_expert_seq > 0) & (token_for_expert_seq <= capacity), True, False
        )
        val_token_for_expert = torch.any(
            val_token_for_expert_list, dim=1, keepdim=False
        )

        weights = F.softmax(gate_logits, dim=-1)  # (num_tokens, num_experts)
        expert_output = torch.zeros_like(x)

        for i in range(self.num_experts):
            combined_mask = (val_token_for_expert) & (token_for_expert == i)
            if combined_mask.any():
                expert_output[combined_mask] = self.experts[i](x[combined_mask])

        # aux_loss
        token_per_expert = torch.sum(
            expert_label, dim=0, keepdim=False
        )  # (num_experts)
        f_i = token_per_expert / num_tokens
        p_i = torch.mean(weights, dim=0, keepdim=False)
        aux_loss = self.alpha * self.num_experts * torch.sum(f_i * p_i)

        weighted_output = expert_output * torch.sum(
            weights * expert_label, dim=1, keepdim=True
        )
        return weighted_output, aux_loss
