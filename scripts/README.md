This directory contains the release-facing batch runner for SabreCoder benchmarks.

`run_sabrecoder_main_benchmark_parallel.sh`:

- discovers prepared LCC and CCEval prompt files
- launches one benchmark job per dataset
- assigns jobs to the GPU list you provide
- writes logs and results into timestamped subdirectories

Example:

```bash
bash scripts/run_sabrecoder_main_benchmark_parallel.sh \
  --model_name deepseek-ai/deepseek-coder-1.3b-base \
  --budgets "14336 12288" \
  --gpus "0 1"
```

Prepared prompt directories are expected at:

- `data/_cceval_rag_prompts`
- `data/_lcc_budget_prompts`

If you built prompt sets with tokenizer-specific suffixes, the runner will try to select the matching directory automatically.
