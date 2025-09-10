# coding: utf-8
import math
import torch
import torch.nn.functional as F
from collections import deque

EPS = 1e-12
LAM_MAX_FACTOR = 50.0  # λ 的上限相对 ||Δw|| 的倍数（防爆）

@torch.no_grad()
def _flatten_model_params(model: torch.nn.Module) -> torch.Tensor:
    """把 nn.Module 参数展平为一维张量（不保留计算图）"""
    return torch.cat([p.detach().view(-1) for p in model.parameters()])

def _k_thresholds_by_dim(d: int):
    """
    k_95 / k_99 门限。优先使用原实现的表；若缺失则用二项分布正态近似：
      E = d/2, Var = d/4, k_95 = E + 1.645*sqrt(Var), k_99 = E + 2.326*sqrt(Var)
    """
    table = {
        1204682: (603244, 603618),
        139960:  (70288,  70415),
        717924:  (359659, 359948),
        145212:  (72919,  73049),
    }
    if d in table:
        return table[d]
    mu = 0.5 * d
    sigma = math.sqrt(d * 0.25)
    k95 = math.ceil(mu + 1.645 * sigma)
    k99 = math.ceil(mu + 2.326 * sigma)
    return (k95, k99)


class PoisonedFLStrict:
    """
    严格版 PoisonedFL（适配 SignGuard）
    - 第 1 轮：boundary_pert 风格初始化（良性集合内 score=||g||*(1-cos) 选 outlier 作为占位），并用其 sign 固定多轮方向
    - 第 2+ 轮：严格 PoisonedFL 缩放思想
        vt = ( |Δw − (||Δw||/||a_{t-1}||) * a_{t-1}| * sign_vector ) / ||·||_2
        λ  = c * ||Δw|| ，且每 50 轮做“符号一致性”检查（不足 k_99 时 c ← 0.7c）
        a_t = λ * vt
    - 数值防爆：λ 上限 + 自适应降档 + NaN/Inf 兜底
    """
    def __init__(self, num_malicious: int, initial_model, scaling_factor: float = 100):
        self.num_malicious = num_malicious
        self.scaling_factor = scaling_factor
        self.round = 0

        # 模型展平、环形缓存（用于 t-50）
        self.prev_global_flat = self._flatten(initial_model)  # 上一轮的全局参数向量 w_t
        self.model_ref = initial_model
        self.snapshots = deque(maxlen=50)
        self.snapshots.append(self.prev_global_flat.clone())

        # 攻击器内部状态
        self.sign_vector = None            # 固定的 ±1 符号向量（第一轮确定）
        self.last_attack_vector = None     # 上一轮恶意向量 a_{t-1}
        self.last_global_update = None     # 上一轮全局模型增量 Δw_t

    def _flatten(self, model_or_tensor):
        if isinstance(model_or_tensor, torch.nn.Module):
            return _flatten_model_params(model_or_tensor)
        elif isinstance(model_or_tensor, torch.Tensor):
            return model_or_tensor.detach().view(-1)
        elif isinstance(model_or_tensor, dict):  # state_dict
            return torch.cat([p.view(-1) for p in model_or_tensor.values()
                              if isinstance(p, torch.Tensor)])
        else:
            raise TypeError("不支持的模型类型")

    @torch.no_grad()
    def generate_attack(self, benign_grads: list, aggregator=None):
        """
        返回恶意客户端上传的“梯度列表”，长度 = num_malicious
        第 1 轮：boundary_pert 初始化（良性集合中选 outlier）
        之后：严格 poisonedfl + 数值防爆
        """
        self.round += 1
        device = self.prev_global_flat.device

        # ---------------- 第 1 轮：boundary_pert 风格评分（候选来自良性集合） ----------------
        if self.sign_vector is None or self.round == 1:
            if len(benign_grads) == 0:
                base_direction = torch.zeros_like(self.prev_global_flat)
            else:
                # B x D
                benign_stack = torch.stack(benign_grads, dim=0).to(device)
                # 平均方向
                mean_grad = benign_stack.mean(dim=0)
                mean_unit = mean_grad / (mean_grad.norm() + 1e-12)
                # 各向量范数
                norms = benign_stack.norm(p=2, dim=1)  # (B,)
                # 与平均方向的余弦相似度
                cos_sim = F.cosine_similarity(benign_stack, mean_unit.unsqueeze(0), dim=1)  # (B,)
                # 评分：既大又偏
                score = norms * (1.0 - cos_sim)
                best_idx = torch.argmax(score).item()
                base_direction = benign_stack[best_idx].detach().clone()
                # 归一化（避免第一轮占位过大）
                bd_norm = base_direction.norm() + EPS
                base_direction = base_direction / bd_norm

            # 用占位向量的符号固定多轮方向
            self.sign_vector = torch.sign(base_direction)
            self.sign_vector[self.sign_vector == 0] = 1.0
            self.last_attack_vector = base_direction.detach().clone()

            # 所有恶意客户端使用同一占位向量
            return [self.last_attack_vector.clone() for _ in range(self.num_malicious)]

        # ---------------- 第 2 轮及以后：严格 poisonedfl 逻辑 ----------------
        if self.last_global_update is None:
            # 极端兜底，不应发生
            self.last_global_update = torch.zeros_like(self.prev_global_flat)

        history = self.last_global_update.view(-1)                       # Δw_t
        history_norm = torch.norm(history, p=2)

        last_grad = (self.last_attack_vector if self.last_attack_vector is not None
                     else torch.zeros_like(history))
        last_grad_norm = torch.norm(last_grad, p=2)

        # (1) 计算单位“幅度形状”向量 v_t
        scale = torch.abs(history - last_grad * (history_norm / (last_grad_norm + EPS)))
        dev = scale * self.sign_vector.to(history.dtype).to(history.device)
        dev = dev / (torch.norm(dev, p=2) + EPS)                         # v_t

        # (2) λ = c * ||Δw||，并每 50 轮做“符号一致性”检查
        sf = self.scaling_factor
        lam = sf * history_norm

        if self.round % 50 == 0:
            current_model = self.prev_global_flat                        # 当前 w_t（在上一轮 update_state 已更新）
            last_50 = self.snapshots[0]                                  # t-50（不足 50 轮则取最早一个）
            total_update = current_model - last_50                       # w_t - w_{t-50}
            total_update = torch.where(total_update == 0, current_model, total_update)

            current_sign = torch.sign(total_update)
            d = self.sign_vector.numel()
            _, k_99 = _k_thresholds_by_dim(d)
            aligned = (current_sign == self.sign_vector).sum().item()

            if (aligned < k_99) and (self.scaling_factor * 0.7 >= 0.5):
                sf = self.scaling_factor * 0.7
            else:
                sf = self.scaling_factor
            lam = sf * history_norm

        # λ 上限（防爆）：不超过 LAM_MAX_FACTOR * ||Δw||
        lam = torch.clamp(lam, max=LAM_MAX_FACTOR * (history_norm + EPS))

        # (3) 最终恶意“更新/梯度”
        mal_update = lam * dev

        # 数值兜底与自适应降档
        if torch.isnan(mal_update).any() or torch.isinf(mal_update).any():
            # 连续降档直到稳定或达到下限
            for _ in range(5):
                sf = sf / 10.0
                lam = torch.clamp(sf * history_norm, max=LAM_MAX_FACTOR * (history_norm + EPS))
                mal_update = lam * dev
                if not (torch.isnan(mal_update).any() or torch.isinf(mal_update).any()):
                    break
            # 仍不稳定则小幅兜底
            if torch.isnan(mal_update).any() or torch.isinf(mal_update).any():
                mal_update = 0.01 * dev

        mal_update = torch.nan_to_num(mal_update, nan=0.0, posinf=0.0, neginf=0.0)

        self.last_attack_vector = mal_update.detach().clone()
        return [mal_update.clone() for _ in range(self.num_malicious)]

    @torch.no_grad()
    def update_state(self, new_global_model):
        """
        每轮聚合 + optimizer.step() 之后调用：
          - 计算 Δw_t = w_{t+1} - w_t
          - 刷新 w_t、快照队列
        """
        new_flat = self._flatten(new_global_model)
        self.last_global_update = new_flat - self.prev_global_flat
        self.prev_global_flat = new_flat.clone()
        self.snapshots.append(self.prev_global_flat.clone())
        self.model_ref = new_global_model
