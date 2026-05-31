# Evaluating Large Language Models for Code Optimization

This repository contains the code, inference outputs, and documentation for a Master's thesis extending the [PRO-Bench framework (SWE-Pro)](https://github.com/probench-swe/SWE-Pro) with two new context granularities and five LLM models.

## Contributions Over SWE-Pro

| Dimension | SWE-Pro (original) | This Work |
|---|---|---|
| Context granularity | Full-file | + **Class-level**, **Method-level** |
| LLM models | GPT-4o, Claude, Gemini, … | MiniMax M2.7, GLM-5.1, DeepSeek V4 Pro, Kimi K2.6, GPT-5.2 |
| Retrieval strategy | Oracle + BM25 | Oracle only |
| New code | — | `prep/prompt/oracle_context_validator.py`, `inference/llm_client/zhipu_client.py` |

## Experimental Setup

**15 experiments** = 5 models × 3 granularities (full-file, class-level, method-level), evaluated on 102 performance-regression PRs from pandas, scikit-learn, and xarray.

| Model | Provider | Granularities |
|---|---|---|
| GPT-5.2 (`gpt-5-2025-08-07`) | OpenAI | full-file, class, method |
| GLM-5.1 (`glm-5.1`) | Zhipu AI | full-file, class, method |
| DeepSeek V4 Pro (`deepseek-v4-pro`) | NVIDIA NIM | full-file, class, method |
| Kimi K2.6 (`kimi-k2.6`) | NVIDIA NIM | full-file, class, method |
| MiniMax M2.7 (`minimax-m2.7`) | NVIDIA NIM | full-file, class, method |

## Results

Full evaluation results (~5 GB) are hosted on Google Drive:

**[Download Results](https://drive.google.com/file/d/1lEVWWITDcpHh0OvRufYVSrzgtV4oEr-n/view?usp=sharing)**

The results directory contains per-PR reports, performance measurements, and experiment summaries for all 15 runs plus the developer reference (anchor) baseline.

## Repository Structure

```
probench/
├── harness/          # Docker-based evaluation harness (correctness + performance)
├── inference/        # LLM client implementations
│   └── llm_client/
│       ├── zhipu_client.py           # GLM-5.1 via Zhipu AI
│       ├── nvidia_nim_client.py      ← new: DeepSeek V4 Pro, Kimi K2.6
│       ├── nvidia_nim_openai_client.py  # MiniMax M2.7
│       └── openai_chat_client.py    # GPT-5.2
├── prep/
│   └── prompt/
│       ├── oracle_context_validator.py  ← new: class- and method-level context extraction
│       └── prompt_oracle_builder.py
├── reporting/        # Experiment aggregation and summary scripts
├── scenarios/        # Per-library benchmark scenarios (pandas, sklearn, xarray)
└── utils/

data/
└── dataset.json      # 102-PR benchmark dataset

inference/            # LLM completions for all 15 experiments (15 × 102 PRs)
├── oracle_full_file__glm-5.1.../
├── oracle_class_and_function_level__glm-5.1.../
└── ...

config/config.yaml    # Measurement and analysis parameters
docker/               # Dockerfiles for reproducible evaluation environments
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/probench-granularity-thesis.git
cd probench-granularity-thesis
pip install -e .
```

Set API keys in a `.env` file:
```
OPENAI_API_KEY=...
ZHIPU_API_KEY=...
NVIDIA_API_KEY=...
```

## Reproducing Evaluation

To re-run evaluation from the stored inference outputs:

```bash
python -m probench.harness.run_evaluation \
    --inference_dir inference/oracle_full_file__glm-5.1__tNone__topNone__pd_4102f209a1355 \
    --config config/config.yaml
```

To regenerate summaries from existing results:

```bash
python -m probench.reporting.aggregate_all_experiments --results_dir results/
```

## Attribution

This work extends [SWE-Pro](https://github.com/probench-swe/SWE-Pro) (Mozilla Public License 2.0).  
Original framework by Siemens AG. All original files retain their MPL-2.0 license headers.

## License

Mozilla Public License 2.0 — see [LICENSE](LICENSE).
