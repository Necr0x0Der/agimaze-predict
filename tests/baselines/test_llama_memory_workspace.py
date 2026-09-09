import pytest


torch = pytest.importorskip("torch")

from agimaze_predict.baselines.llama_memory.spatial_memory import (  # noqa: E402
    SpatialMemoryConfig,
    SpatialWorkspace,
)


def test_workspace_initializes_action_projection_at_construction() -> None:
    workspace = SpatialWorkspace(
        SpatialMemoryConfig(d_model=8, n_heads=2, spatial_layers=1),
        action_embedding_dim=12,
    )

    assert workspace.action_projection.in_features == 12
    assert all(parameter.numel() > 0 for parameter in workspace.parameters())
