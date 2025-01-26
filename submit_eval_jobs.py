import os
import argparse

configs = [
    {
        'output-dir': "outputs/Qwen2.5-Math-7B",
        'model-path': "Qwen/Qwen2.5-Math-7B",
        'tokenizer-path': "Qwen/Qwen2.5-Math-7B",
        'model-size': "7b",
        'overwrite': True,
        'use-vllm': True,
        'no-markup-question': True,
        'test-conf': "configs/qwen7b.json",
        'prompt_format': 'few_shot',
        'few_shot_prompt': 'CoTGSMPrompt',
        'expname': 'eval-Qwen2.5-Math-7B'
    }
]

base_conf = configs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n-gpus", type=int, default=4)
    args = parser.parse_args()

    conf = base_conf[0]
    cmd = "python run_subset_parallel.py"
    for key, val in conf.items():
        if key == 'expname':
            continue
        if isinstance(val, str):
            cmd += f" --{key} '{val}'"
        elif val:
            cmd += f" --{key}"
    cmd += f" --test-conf {conf['test-conf']}"
    cmd += f" --n-repeats {args.n_repeats}"
    cmd += f" --temperature {args.temperature}"
    cmd += f" --ngpus {args.n_gpus}"
    cmd += f" --rank {0}"
    print(cmd, flush=True)
    os.system(cmd)

if __name__ == '__main__':
    main()
