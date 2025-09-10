# coding: utf-8
import math
import torch
import torch.nn.functional as F

class PoisonedFLAdaptive:
    """
    PoisonedFL 自适应攻击（稳健版）
    - 第 1 轮：恶意端像良性端一样本地训练 -> byz_grads；
              在 byz_grads 内用 boundary_pert 评分选出 outlier 作为方向；
              用其符号得到固定的 sign_vector（跨轮保持一致）；
              第一轮返回一个“温和幅度”的占位向量（避免极端值引起不稳定）。
    - 后续轮：固定方向 * 自适应幅度。幅度形状参考 |Δw| 与 |a_{t-1}| 的关系；
              步长 λ = c · ||Δw||，并做轻度自适应与上限/降档防爆。
    注意：框架上传的是“梯度”，而我们内部构造的是“期望的模型更新”；
         因此对外返回时需要取负号（梯度 = − 期望更新），
         内部状态 last_attack_vector 仍保存“期望更新”用于下一轮计算。
    """
    def __init__(self, num_malicious: int, initial_model, aggregator=None,
                 c_scale: float = 1.0, max_scale_mult: float = 50.0):
        self.prev_global_flat = self._flatten_model(initial_model)
        self.sign_vector = None
        self.last_attack_vector = None
        self.last_global_update = None
        self.num_malicious = num_malicious
        self.round = 0
        self.aggregator = aggregator
        self.c_scale = float(c_scale)
        self.max_scale_mult = float(max_scale_mult)

    # ------------------------- utils ------------------------- #
    def _flatten_model(self, model: torch.nn.Module | dict | torch.Tensor) -> torch.Tensor:
        """将模型参数展平为 1D Tensor（保持设备不变）"""
        if isinstance(model, torch.nn.Module):
            parts = [p.detach().view(-1) for p in model.parameters()]
            dev = parts[0].device if parts else torch.device("cpu")
            return torch.cat(parts).to(dev)
        elif isinstance(model, torch.Tensor):
            return model.detach().view(-1)
        elif isinstance(model, dict):  # state_dict
            parts = [p.view(-1) for p in model.values() if isinstance(p, torch.Tensor)]
            dev = parts[0].device if parts else torch.device("cpu")
            return torch.cat(parts).to(dev)
        else:
            raise TypeError(f"不支持的模型类型: {type(model)}")

    # ----------------------- core API ------------------------ #
    @torch.no_grad()
    def generate_attack(self, benign_grads: list,
                        aggregator=None,
                        byz_grads: list | None = None) -> list[torch.Tensor]:
        """
        生成恶意客户端“梯度向量”（与框架接口一致，外部返回的是梯度）
        Args:
            benign_grads : list[Tensor] 良性端的扁平梯度（未直接使用，保留接口）
            aggregator   : 预留
            byz_grads    : 第 1 轮传入恶意端本地训练得到的梯度，用于初始化
        Return:
            list[Tensor]  恶意端上报“梯度”列表（长度 = num_malicious）
        """
        self.round += 1
        dev = self.prev_global_flat.device

        # ======================== 第 1 轮：初始化 ======================== #
        if self.sign_vector is None or self.round == 1:
            if not byz_grads:   # 极端兜底：无可选候选
                base_direction = torch.zeros_like(self.prev_global_flat, device=dev)
            else:
                # 设备对齐后 stack
                byz_stack = torch.stack([g.detach().to(dev) for g in byz_grads], dim=0)  # [m_byz, D]
                mean_grad = byz_stack.mean(dim=0)
                mean_unit = mean_grad / (mean_grad.norm() + 1e-12)
                norms = byz_stack.norm(p=2, dim=1)  # (m_byz,)
                cos_sim = F.cosine_similarity(byz_stack, mean_unit.unsqueeze(0), dim=1)
                score = norms * (1.0 - cos_sim)     # 既大又偏离平均方向
                best_idx = torch.argmax(score).item()
                base_direction = byz_stack[best_idx].detach().clone()

            # 固定跨轮方向（只取符号，避免 0 符号）
            bd = base_direction / (base_direction.norm() + 1e-12)
            self.sign_vector = torch.sign(bd)
            self.sign_vector[self.sign_vector == 0] = 1.0

            # 第一轮占位“期望更新”（温和幅度）
            if self.last_global_update is not None:
                tau0 = 0.5 * self.last_global_update.norm()
            else:
                tau0 = 1.0  # 初始非常小的幅度
            first_update = self.sign_vector * tau0

            # 内部状态：保存“期望更新”
            self.last_attack_vector = first_update.detach().clone()

            # 对外上报：梯度 = − 期望更新
            first_grad = -first_update
            return [first_grad.clone() for _ in range(self.num_malicious)]

        # ===================== 2+ 轮：固定方向 + 幅度 ===================== #
        if self.last_global_update is None:
            # 极端兜底：理论上不会出现（正常应当已由上一轮 update_state 写入）
            self.last_global_update = torch.zeros_like(self.prev_global_flat, device=dev)

        abs_dw = self.last_global_update.to(dev).abs()        # |Δw|
        abs_a1 = self.last_attack_vector.to(dev).abs()        # |a_{t-1}|

        # 幅度形状：|Δw| - α|a_{t-1}|  （加入 clamp，避免负值反向）
        a1_norm = abs_a1.norm() + 1e-12
        dw_norm = abs_dw.norm() + 1e-12
        alpha = dw_norm / a1_norm
        vt_numer = (abs_dw - abs_a1 * alpha).clamp_min(0.0)

        # 归一化的幅度形状 v_t
        vt_norm = vt_numer.norm()
        if vt_norm <= 1e-12:
            v_t = torch.ones_like(vt_numer, device=dev) / math.sqrt(vt_numer.numel())
        else:
            v_t = vt_numer / vt_norm

        # 基准步长：λ = c · ||Δw||
        lam = self.c_scale * dw_norm
        # 轻度自适应：根据上轮透传比 r = ||Δw|| / ||a_{t-1}||
        r = dw_norm / (a1_norm)
        if r < 0.1:
            lam *= 0.5
        elif r > 0.9:
            lam *= 1.2

        # 上限 & 数值稳健
        lam = torch.clamp(lam, max=self.max_scale_mult * dw_norm)
        malicious_update = self.sign_vector * (lam * v_t)

        # NaN/Inf 兜底 + 降档
        if torch.isnan(malicious_update).any() or torch.isinf(malicious_update).any():
            lam = lam / 10.0
            malicious_update = self.sign_vector * (lam * v_t)
            malicious_update = torch.nan_to_num(malicious_update, 0.0, 0.0, 0.0)

        # 内部状态：保存“期望更新”
        self.last_attack_vector = malicious_update.detach().clone()

        # 对外上报：梯度 = − 期望更新
        malicious_grad = -malicious_update
        return [malicious_grad.clone() for _ in range(self.num_malicious)]

    @torch.no_grad()
    def update_state(self, new_global_model):
        """每轮聚合 + optimizer.step() 之后调用，刷新 Δw 与 w_{t-1}。"""
        new_global_flat = self._flatten_model(new_global_model).to(self.prev_global_flat.device)
        self.last_global_update = new_global_flat - self.prev_global_flat
        self.prev_global_flat = new_global_flat.detach().clone()
