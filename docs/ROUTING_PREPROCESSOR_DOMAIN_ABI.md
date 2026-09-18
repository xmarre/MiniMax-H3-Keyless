# Domain-aware routing preprocessor ABI

Canonical `h3_keyless_core50_v1` attention keeps retrieval V and logical routing
semantics separate. Sparse providers are allowed to select/reorder the V domain
before route materialization, but the route must then be derived from exactly those
selected V rows with the corresponding routing positions.

`RoutingPreprocessor` therefore has two execution forms:

- `fn(route)`: legacy/full-domain routing-only transform;
- `domain_fn(route, value_domain, routing_position_domain)`: selected/reordered
  domain transform.

The latter is optional so existing full-domain preprocessors remain valid. A
provider that reduces/reorders V must not silently apply a legacy preprocessor
whose row semantics depend on the original packed coordinates; it must require a
domain-aware implementation or fail closed.

`RoutingSpecV1.select_value_rows(...)` is the authoritative transport. It selects
raw V once, applies the same selection to RoPE rows and log measure, and composes
both the value and routing-position domains through earlier selections. When the
selected spec later materializes route(V), domain-aware preprocessors receive those
composed domains.

The routing-position domain is the coordinate source for transforms defined on
logical key/routing positions. The value domain describes retrieval ownership and
may use a different coordinate namespace. Neither domain authorizes mutation of raw
retrieval V.

The duck-typed external preprocessor form mirrors this contract: a callable object
with an `identity` may additionally expose
`apply_domain(route, value_domain, routing_position_domain)`. Keyless wraps that
method into `RoutingPreprocessor.domain_fn` without taking attention ownership.

This is an additive API-1 capability. It does not make every legacy routing
preprocessor safe under sparse selection; selected-domain providers remain
responsible for refusing preprocessors that do not expose domain-aware semantics.
