# Evidence bundles

## Primary: real WRDS research

[`wrds_real_data/`](wrds_real_data/README.md) contains the reviewed aggregate
evidence from CRSP/Compustat. It includes the frozen specification, private-input
hash manifest, coverage, IC inference, cost-aware portfolio summaries, and
portfolio-level monthly returns. It contains no licensed security-level rows.

## Legacy: deterministic synthetic validation

The `synthetic_*` files were generated with:

```bash
python run_model.py --offline
```

They record a synthetic-universe Sharpe of 2.021 and several apparently
significant factor/horizon rows. Those figures verify execution and analysis
plumbing only. They are not historical-market evidence and should not be used as
resume performance claims.
