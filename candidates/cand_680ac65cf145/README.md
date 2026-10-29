# Single-representation vocabulary layout

The exact packed GEMV and CPU codec are inherited from the prior vocabulary
candidate. A new bitwise decoder supports native multi-token fallback, while
the trusted controller exchanges native and packed backing stores between
lanes. No original dense weight is retained during a candidate request; no
packed arrays are retained during a baseline request. Conversion/validation
may transiently hold both and must report that preparation overhead separately.

Every request gets fresh native BF16 caches. Prefill preserves native linear
arithmetic if its projection is evaluated. The packed reader remains specialized
to one BF16 vector; unsupported storage, dtype or shape is rejected explicitly.
Full-model correctness, routes, tokens, caches, placement and memory accounting
remain controller responsibilities. No correctness or speedup verdict is given.
