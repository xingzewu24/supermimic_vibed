"""Mimic-feasible hand actuator mapping. Tracking targets come exclusively from motion."""

from __future__ import annotations

import torch

from .config import HandControlConfig, validate_hand_config
from .robot import HAND_JOINT_NAMES, HAND_MIMIC_JOINT_NAMES, hand_joint_limits, mimic_relations


class HandControl:
    def __init__(self, device, cfg: HandControlConfig | None = None):
        self.cfg = cfg or HandControlConfig()
        validate_hand_config(self.cfg)
        lo, hi = hand_joint_limits()
        self.lo = torch.tensor(lo, device=device, dtype=torch.float32)
        self.hi = torch.tensor(hi, device=device, dtype=torch.float32)
        self.mid, self.half = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2
        relations = mimic_relations()
        self.drivers = torch.tensor([HAND_JOINT_NAMES.index(relations[n][0]) for n in HAND_MIMIC_JOINT_NAMES], device=device)
        self.ratios = torch.tensor([relations[n][1] for n in HAND_MIMIC_JOINT_NAMES], device=device)
        self.offsets = torch.tensor([relations[n][2] for n in HAND_MIMIC_JOINT_NAMES], device=device)
        expanded_lo, expanded_hi = self.expand(self.lo), self.expand(self.hi)
        self.full_lo = torch.minimum(expanded_lo, expanded_hi)
        self.full_hi = torch.maximum(expanded_lo, expanded_hi)

    def action_target(self, action):
        """Unbounded policy outputs map to absolute, mimic-feasible PD targets."""
        return self.mid + self.half * torch.tanh(action)

    def expand(self, command):
        return torch.cat((command, command[..., self.drivers] * self.ratios + self.offsets), -1)
