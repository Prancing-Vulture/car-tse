"""Wrapper around vendored SpEx_Plus."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from car_tse.config import REPO_ROOT

VENDOR_ROOT = REPO_ROOT / "third_party" / "speaker_extraction_SpEx"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from nnet.spex_plus import SpEx_Plus  # noqa: E402


def spex_nnet_conf(spex_cfg: dict[str, Any], sample_rate: int = 8000) -> dict[str, Any]:
    """Convert window sizes in milliseconds to samples (vendor uses sample counts)."""
    return {
        "L1": int(spex_cfg["L1_ms"] * sample_rate / 1000),
        "L2": int(spex_cfg["L2_ms"] * sample_rate / 1000),
        "L3": int(spex_cfg["L3_ms"] * sample_rate / 1000),
        "N": spex_cfg["N"],
        "B": spex_cfg["B"],
        "O": spex_cfg["O"],
        "P": spex_cfg["P"],
        "Q": spex_cfg["Q"],
        "num_spks": spex_cfg["num_spks"],
        "spk_embed_dim": spex_cfg["spk_embed_dim"],
        "causal": spex_cfg.get("causal", False),
    }


def load_pretrained_spex(
    checkpoint_dir: Path | str,
    spex_cfg: dict[str, Any],
    sample_rate: int = 8000,
    strict: bool = True,
) -> SpEx_Plus:
    checkpoint_dir = Path(checkpoint_dir)
    nnet = SpEx_Plus(**spex_nnet_conf(spex_cfg, sample_rate))
    ckpt_path = checkpoint_dir / "best.pt.tar"
    if not ckpt_path.exists():
        ckpt_path = checkpoint_dir / "best.pt"
    if ckpt_path.exists():
        cpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = cpt.get("model_state_dict", cpt)
        # Stage-1 checkpoints wrap the vendor network under ``spex.nnet``;
        # accept both those checkpoints and raw vendor ``best.pt.tar`` files.
        if state and not any(key in state for key in nnet.state_dict()):
            prefixed = "spex.nnet."
            wrapped = {
                key[len(prefixed) :]: value
                for key, value in state.items()
                if key.startswith(prefixed)
            }
            if wrapped:
                state = wrapped
        nnet.load_state_dict(state, strict=strict)
    return nnet


class SpExPlusAdapter(nn.Module):
    """Batch-friendly SpEx+ forward."""

    def __init__(self, nnet: SpEx_Plus):
        super().__init__()
        self.nnet = nnet

    def forward(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        mixture: [B, T]
        enrollment: [B, T_aux]
        returns s1,s2,s3 [B,T] and spk_logits [B, num_spks]
        """
        s1_list, s2_list, s3_list, logit_list = [], [], [], []
        for i in range(mixture.shape[0]):
            mix = mixture[i]
            aux = enrollment[i]
            if aux.dim() == 1:
                aux = aux.unsqueeze(0)
            aux_len = torch.tensor(float(aux.shape[-1]), device=mix.device)
            o1, o2, o3, logits = self.nnet(mix, aux, aux_len)
            t = mixture.shape[-1]

            def _wave(o: torch.Tensor) -> torch.Tensor:
                return o.reshape(-1)[:t]

            s1_list.append(_wave(o1))
            s2_list.append(_wave(o2))
            s3_list.append(_wave(o3))
            logit_list.append(logits.reshape(-1) if logits.dim() > 1 else logits)
        t = mixture.shape[-1]
        s1 = torch.stack(s1_list, dim=0)
        s2 = torch.stack(s2_list, dim=0)
        s3 = torch.stack(s3_list, dim=0)
        spk_logits = torch.stack(
            [x if x.dim() > 0 else x.unsqueeze(0) for x in logit_list], dim=0
        )
        if spk_logits.dim() == 1:
            spk_logits = spk_logits.unsqueeze(0)
        return s1, s2, s3, spk_logits

    def encoder_params(self):
        enc = [
            self.nnet.encoder_1d_short,
            self.nnet.encoder_1d_middle,
            self.nnet.encoder_1d_long,
            self.nnet.ln,
            self.nnet.proj,
            self.nnet.spk_encoder,
        ]
        return enc
