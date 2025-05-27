import os
import re
import json
import argparse
from typing import List, Dict, Any

# 匹配并去掉 ANSI 颜色码及进程前缀，例如：\x1b[36m(main_task pid=11227)\x1b[0m
ANSI_PREFIX = re.compile(r'^\x1b\[\d+m\([^)]*\)\x1b\[0m\s*')
PID_PREFIX = re.compile(r'\(main_task pid=\d+\)\s*')

def sorted_log_files(log_dir: str, pattern: str = r'(.+?)(?:_(\d+))?\.out$') -> List[str]:
    files = []
    for fn in os.listdir(log_dir):
        m = re.match(pattern, fn)
        if m:
            idx = int(m.group(2)) if m.group(2) else 0
            files.append((idx, fn))
    files.sort(key=lambda x: x[0])
    return [fn for _, fn in files]


def parse_summary_line(line: str) -> Dict[str, Any]:

    parts = [p.strip() for p in line.split(' - ')]
    d: Dict[str, Any] = {}
    # 第一部分 "step:N"
    step = int(parts[0].split(':', 1)[1])
    d['step'] = step
    # 解析 key:val
    for kv in parts[1:]:
        if ':' not in kv:
            continue
        key, val = kv.split(':', 1)
        try:
            num = float(val)
        except ValueError:
            num = val
        d[key] = num
    return d


def extract_summary_from_logs(
    log_dir: str,
    out_summary: str,
) -> None:

    log_files = sorted_log_files(log_dir)

    step_summaries: Dict[int, Dict[str, Any]] = {}


    in_sample = False
    sample_lines: List[str] = []

    for fname in log_files:
        path = os.path.join(log_dir, fname)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:

                line = raw.rstrip('\n')

                # 去掉 ANSI 前缀
                content = ANSI_PREFIX.sub('', line)
                content = PID_PREFIX.sub('', content)
                stripped = content.lstrip()

                # 检测 summary 行，仅当去除前缀后以 step: 开头
                if stripped.startswith('step:'):
                    summary = parse_summary_line(stripped)

                    step = summary['step']
                    # 后续日志覆盖前面相同 step
                    step_summaries[step] = summary


    # 将结果按 step 升序输出
    summaries_list = [step_summaries[s] for s in sorted(step_summaries)]

    with open(out_summary, 'w', encoding='utf-8') as f:
        json.dump(summaries_list, f, ensure_ascii=False, indent=2)

    return summaries_list


def extract_example_from_logs(
    log_dir: str,

    out_samples: str
) -> None:

    log_files = sorted_log_files(log_dir)

    step_samples = {}

    sample_lines: List[str] = []

    for fname in log_files:
        path = os.path.join(log_dir, fname)

        current_step = None
        ground_truths, ground_truths_reward, response_token_len = None, None, None

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                line = raw.rstrip('\n')
                # 去掉 ANSI 前缀
                content = ANSI_PREFIX.sub('', line)
                stripped = content.lstrip()

                step_match = re.findall(r'step:(\d+)', stripped) # 拿到当前的 step

                if step_match:
                    current_step = int(step_match[0])+1 # 一个step 开始了

                    if current_step in step_samples: # 已经有了 current_step 要清空
                        step_samples[current_step] = []

                    else: # 没有 current_step 要新建
                        step_samples[current_step] = []

                # 要取到的字段
                if current_step is not None:

                    if not ground_truths:
                        ground_truths = re.findall(r'"ground_truths":\s*"(.*?)(?<!\\)"', stripped)
                    if not ground_truths_reward:
                        ground_truths_reward = re.findall(r'"ground_truths_reward":\s*([0-9.]+)', stripped)
                    if not response_token_len:
                        response_token_len = re.findall(r'"response_token_len":\s*(\d+)', stripped)

                    if ground_truths and ground_truths_reward and response_token_len:

                        step_samples[current_step].append({
                            "ground_truths": ground_truths[0],
                            "ground_truths_reward": float(ground_truths_reward[0]),
                            "response_token_len": int(response_token_len[0])
                        })

                        ground_truths, ground_truths_reward, response_token_len = None, None, None


    # 将结果按 step 升序输出
    samples_dict = {str(s): step_samples[s] for s in sorted(step_samples)}

    with open(out_samples, 'w', encoding='utf-8') as f:
        json.dump(samples_dict, f, ensure_ascii=False, indent=2)

    return samples_dict


if __name__ == '__main__':


    log_dir = '/Users/grayson/Desktop/logs/grpo-deepscaler_17k-7b_math-bs_128-3k-zs_qwen-r_f_a'

    summary_out = '/Users/grayson/Desktop/logs/grpo-deepscaler_17k-7b_math-bs_128-3k-zs_qwen-r_f_a/summary.json'

    samples_out = '/Users/grayson/Desktop/logs/grpo-deepscaler_17k-7b_math-bs_128-3k-zs_qwen-r_f_a/samples.json'

    extract_summary_from_logs(log_dir, summary_out)
    print(f"✔ 已生成：\n  • 摘要文件: {summary_out}\n ")

    # extract_example_from_logs(log_dir, samples_out)
    # print(f"✔ 已生成：\n  • 样本文件: {samples_out}\n ")


