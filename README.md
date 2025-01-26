# LLM-Math-Eval

## Overview

LLM-Math-Eval provides a standardized testing environment to assess how well language models perform on various mathematical tasks, from basic arithmetic to advanced problem-solving.

## Key Features

### Supported Benchmarks
- **GSM8K** (Grade School Math 8K) - Elementary to middle school word problems
- **MATH** - Advanced mathematical problems across various topics
- **MMLU-STEM** - Science, Technology, Engineering, and Mathematics questions
- **AGIEval** - Comprehensive evaluation of mathematical reasoning
- **SAT** - Standard college admission test problems


### Model Compatibility
Validated against leading LLM implementations:
- Deepseek
- Qwen
- Other popular open-source models

## Getting Started

### Usage

1. Run evaluation jobs:
```bash
python submit_eval_jobs.py
```

2. Summarize evaluation results:
```bash
python summerize_eval_results.py
```

## Citation

If you use LLM-Math-Eval in your research, please cite our work using the following BibTeX entry:

```bibtex
@misc{llm-math-eval,
  author = {Your Name},
  title = {LLM-Math-Eval: A Comprehensive Framework for Evaluating Mathematical Reasoning in Large Language Models},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/LLM-Math-Eval}}
}
```

## Acknowledgments

- Special thanks to [DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math) team 
.
