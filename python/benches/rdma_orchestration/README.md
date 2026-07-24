# pytokio-removal RDMA benchmark

This tool answers one question: did the pytokio-removal refactor change RDMA
performance? It is not a general RDMA benchmark or an automated regression
gate.

## Run

Use the same idle host for both revisions. Run both backends on the baseline
revision:

```bash
buck run fbcode//monarch/python/benches/rdma_orchestration:bench -- --backend tcp
buck run fbcode//monarch/python/benches/rdma_orchestration:bench -- --backend ibverbs --target nic:mlx5_0
```

Move to the refactored revision and run the same two commands. Use the same NIC
for both ibverbs runs.

Compare TCP with TCP and ibverbs with ibverbs. For each backend, inspect both:

- `cold_init`, which covers buffer setup, backend initialization, readiness,
  and the first validated read in a fresh process; and
- the warmed serial read/write and K=32 concurrent read/write statistics.

If a difference is close to ordinary run-to-run variation, repeat that command
on both revisions and inspect the spread.

The benchmark validates every transfer, requires ibverbs when that backend is
selected, prints its measurements to stdout, and writes no result files. It
rejects conflicting `MONARCH_RDMA_*` environment settings rather than measuring
a different backend than the command requested.
