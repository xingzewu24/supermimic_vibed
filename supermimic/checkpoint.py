"""Policy restoration shared by numerical evaluation and interactive playback."""

from __future__ import annotations

from supermimic.config import checkpoint_network_config, validate_control_checkpoint


def load_policy(checkpoint: dict, device: str):
    import torch

    from supermimic import networks
    from supermimic.hoi import GOAL_FRAME_DIM
    from supermimic.mdp import OBS_DIM
    from supermimic.networks import Policy

    args = checkpoint["args"]
    validate_control_checkpoint(args)
    net = checkpoint_network_config(args)
    networks.AMP_DTYPE = torch.bfloat16 if args.get("amp", "none") == "bf16" else None
    torch.backends.cuda.matmul.allow_tf32 = args.get("tf32", True)
    policy = Policy(
        OBS_DIM,
        GOAL_FRAME_DIM * args["goal_frames"],
        hist=args["hist"],
        z_dim=net["latent_dim"],
        encoder_hidden=net["encoder_hidden"],
        decoder_hidden=net["decoder_hidden"],
    ).to(device)
    policy.load_state_dict(checkpoint["pi"])
    policy.eval()
    return policy


def initialize_matching_parameters(module, saved_state: dict) -> dict:
    """Warm-start equal-shaped parameters only; new feature normalizers start from fresh data."""
    import torch

    copied, skipped = [], []
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            source = saved_state.get(name)
            if source is not None and source.shape == parameter.shape:
                parameter.copy_(source)
                copied.append(name)
            else:
                skipped.append(name)
    return {"copied": copied, "skipped": skipped}
