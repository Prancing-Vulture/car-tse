import torch

from car_tse.frontend.cabin_frontend import CabinFrontend
from scripts.eval.eval_stage1_input_scale_sweep import scale_batch


def test_mix_rms_mode_reaches_requested_batch_scale():
    multichannel = torch.ones(2, 4, 8) * 0.02
    enrollment = torch.ones(2, 8) * 0.01
    scaled_mix, unchanged_enrollment = scale_batch("mix_rms_0.1", multichannel, enrollment)

    assert torch.allclose(
        torch.sqrt(scaled_mix.square().mean(dim=(1, 2))),
        torch.full((2,), 0.1),
        atol=1.0e-5,
    )
    assert torch.equal(unchanged_enrollment, enrollment)


def test_joint_rms_mode_scales_both_streams():
    multichannel = torch.ones(1, 2, 4) * 0.02
    enrollment = torch.ones(1, 4) * 0.01
    scaled_mix, scaled_enrollment = scale_batch("joint_rms_0.05", multichannel, enrollment)

    assert torch.allclose(torch.sqrt(scaled_mix.square().mean()), torch.tensor(0.05), atol=1.0e-5)
    assert torch.allclose(
        torch.sqrt(scaled_enrollment.square().mean()), torch.tensor(0.05), atol=1.0e-5
    )


def test_raw_mode_is_identity():
    multichannel = torch.randn(2, 4, 8)
    enrollment = torch.randn(2, 8)
    scaled_mix, scaled_enrollment = scale_batch("raw", multichannel, enrollment)

    assert scaled_mix is multichannel
    assert scaled_enrollment is enrollment


def test_cabin_frontend_input_rms_target_scales_playback_consistently():
    frontend = CabinFrontend(num_mics=2, ec_hidden=4, spatial_channels=2, input_rms_target=0.1)
    frontend.eval()
    multichannel = torch.ones(1, 2, 8) * 0.02
    playback = torch.ones(1, 8) * 0.01

    with torch.no_grad():
        output = frontend(multichannel, playback)

    # The final convs are zero-initialized, so the output is the mean path;
    # this verifies the configured input normalization without depending on
    # learned frontend residual weights.
    assert torch.allclose(output, torch.full_like(output, 0.1), atol=1.0e-6)
