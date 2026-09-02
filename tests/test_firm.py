import torch

from firm import FIRMState, balanced_responsibilities


def test_projection_has_normalized_rows_and_uniform_columns():
    torch.manual_seed(1)
    probabilities = torch.softmax(torch.randn(12, 3), dim=1)
    result = balanced_responsibilities(probabilities, iterations=100)
    assert torch.allclose(result.sum(dim=1), torch.ones(12), atol=1e-5)
    assert torch.allclose(result.sum(dim=0), torch.full((3,), 4.0), atol=1e-4)


def test_firm_emits_probabilities_and_updates_state():
    torch.manual_seed(2)
    state = FIRMState(
        num_classes=3,
        feature_dim=5,
        device="cpu",
        classifier_weight=torch.randn(3, 5),
    )
    output = state.process_sample(
        fusion_feature=torch.randn(1, 5),
        audio_feature=torch.randn(1, 5),
        video_feature=torch.randn(1, 5),
        source_logits=torch.randn(1, 3),
    )
    assert set(output) == {"source", "rebalance", "firm"}
    for probability in output.values():
        assert torch.allclose(probability.sum(dim=1), torch.ones(1), atol=1e-6)
    assert state.prototype_mass.sum() > 0

