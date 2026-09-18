from __future__ import annotations

import torch

from minimax_h3_keyless.attention import KeylessAttentionDeploy, KeylessAttentionTrain
from minimax_h3_keyless.contracts import PROVIDER_KEY
from minimax_h3_keyless.export import fold_query_route_weight


def _initialize_training_attention(attn: KeylessAttentionTrain) -> None:
    torch.manual_seed(11)
    with torch.no_grad():
        attn.q_proj.weight.normal_(0.0, 0.2)
        attn.query_route.weight.normal_(0.0, 0.2)
        attn.v_proj.weight.normal_(0.0, 0.2)
        attn.q_norm.weight.uniform_(0.7, 1.3)
        attn.route_norm.weight.uniform_(0.7, 1.3)
        attn.out_proj.weight.normal_(0.0, 0.2)


def test_folded_deploy_attention_matches_training_form() -> None:
    hidden, heads, head_dim = 6, 2, 3
    train = KeylessAttentionTrain(hidden, heads, head_dim, 1e-5, dtype=torch.float32)
    deploy = KeylessAttentionDeploy(hidden, heads, head_dim, 1e-5, dtype=torch.float32)
    _initialize_training_attention(train)
    with torch.no_grad():
        q_eff = fold_query_route_weight(train.q_proj.weight, train.query_route.weight)
        deploy.qv_proj.weight.copy_(torch.cat((q_eff, train.v_proj.weight), dim=0))
        deploy.q_norm.weight.copy_(train.q_norm.weight)
        deploy.route_norm.weight.copy_(train.route_norm.weight)
        deploy.out_proj.weight.copy_(train.out_proj.weight)
    torch.manual_seed(12)
    x = torch.randn(7, hidden)
    expected = train(x)
    actual = deploy(x)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


class _CaptureProvider:
    api = 1

    def __init__(self) -> None:
        self.v = None
        self.q = None
        self.route = None

    def __call__(self, **kwargs):
        self.v = kwargs["v"].detach().clone()
        self.q = kwargs["q"].detach().clone()
        self.route = kwargs["routing"].materialize(kwargs["v"]).detach().clone()
        return torch.zeros_like(kwargs["q"])


def test_provider_receives_raw_projected_value_not_route() -> None:
    hidden, heads, head_dim = 5, 2, 3
    attn = KeylessAttentionDeploy(hidden, heads, head_dim, 1e-5, dtype=torch.float32)
    torch.manual_seed(13)
    with torch.no_grad():
        attn.qv_proj.weight.normal_(0.0, 0.3)
        attn.q_norm.weight.fill_(1.0)
        attn.route_norm.weight.fill_(2.0)
        attn.out_proj.weight.normal_(0.0, 0.2)
    provider = _CaptureProvider()
    x = torch.randn(4, hidden)
    inner = heads * head_dim
    expected_v = torch.nn.functional.linear(x, attn.qv_proj.weight[inner:]).view(
        x.shape[0], heads, head_dim
    )
    attn(x, transformer_options={PROVIDER_KEY: provider})
    assert provider.v is not None and provider.route is not None
    torch.testing.assert_close(provider.v, expected_v)
    assert not torch.allclose(provider.route, provider.v)


def test_foreign_domain_aware_preprocessor_survives_attention_wrapping() -> None:
    class DomainAware:
        identity = "domain-aware-test"

        def __init__(self):
            self.calls = []

        def __call__(self, route):
            return route

        def apply_domain(self, route, value_domain, routing_position_domain):
            self.calls.append((value_domain, routing_position_domain))
            return route

    class SelectingProvider:
        api = 1

        def __init__(self):
            self.route = None

        def __call__(self, **kwargs):
            selected_v, selected_routing, _ = kwargs["routing"].select_value_rows(
                kwargs["v"],
                (2, 0),
                identity="selected",
            )
            self.route = selected_routing.materialize(selected_v)
            return torch.zeros_like(kwargs["q"])

    attn = KeylessAttentionDeploy(4, 2, 2, 1e-5, dtype=torch.float32)
    with torch.no_grad():
        attn.qv_proj.weight.normal_()
        attn.q_norm.weight.fill_(1.0)
        attn.route_norm.weight.fill_(1.0)
        attn.out_proj.weight.normal_()

    preprocessor = DomainAware()
    provider = SelectingProvider()
    x = torch.randn(3, 4)
    attn(
        x,
        transformer_options={
            PROVIDER_KEY: provider,
            "minimax_h3_keyless_routing_preprocessors_v1": (preprocessor,),
        },
    )

    assert provider.route is not None
    assert len(preprocessor.calls) == 1
    value_domain, routing_position_domain = preprocessor.calls[0]
    assert value_domain.indices == (2, 0)
    assert routing_position_domain.indices == (2, 0)
    assert value_domain.identity == "selected"
    assert routing_position_domain.identity == "selected"


def test_bad_provider_api_fails_closed() -> None:
    class BadProvider:
        api = 2

        def __call__(self, **kwargs):
            return kwargs["q"]

    attn = KeylessAttentionDeploy(4, 2, 2, 1e-5, dtype=torch.float32)
    with torch.no_grad():
        attn.qv_proj.weight.normal_()
        attn.q_norm.weight.fill_(1.0)
        attn.route_norm.weight.fill_(1.0)
        attn.out_proj.weight.normal_()
    x = torch.randn(2, 4)
    try:
        attn(x, transformer_options={PROVIDER_KEY: BadProvider()})
    except RuntimeError as exc:
        assert "api=1" in str(exc)
    else:
        raise AssertionError("provider api mismatch must fail closed")
