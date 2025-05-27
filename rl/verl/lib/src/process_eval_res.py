# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
import csv

import numpy as np

import os
from tabulate import tabulate

from src.eval.evaluator import MATHEvaluator_Base, MATHEvaluator


import pandas as pd

eval_obj = MATHEvaluator()

def ground_truths_reward_func(response, ground_truths):
    """
    评估 模型生成的答案是否和参考答案一致，如果是 则奖励 1，否则奖励 0。

    """
    if ground_truths is None:
        return 0.0

    answer = eval_obj.parse_answer(response)

    if answer is None:
        answer = ""

    score = 0.0

    if eval_obj.is_equiv(answer, ground_truths):
        score = 1.0

    return score


def answer_format_reward_func(response):
    """
    返回的内容中能提取出答案，并且检测答案是否合法

    """

    answer = eval_obj.parse_answer(response)

    score = 0

    if len(answer) > 0 and eval_obj.val_by_sympy(answer):
        score += 1.0

    return score

def find_files(root_dir, suffix='parquet'):
    """
    递归查找目录下所有文件（可过滤后缀）
    :param root_dir: 根目录路径
    :param suffix: 文件后缀（如 ".txt"），默认不过滤
    :return: 文件路径列表
    """
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if suffix is None or file.endswith(suffix):
                file_list.append(os.path.join(root, file))
    return file_list


def main():
    from pprint import pprint

    # 评测结果文件

    # paths = find_files(
    #     "../data/llm_outputs/eval/Qwen2.5-Math-7B",
    #     ".parquet")


    paths = find_files(
        "../data/llm_outputs/eval/ml-3k/qwen-math-template/Qwen2.5-Math-7B-Instruct",
        ".parquet")



    for path in paths:
        # print( )
        # print()
        dataset = pd.read_parquet(path)
        # output_dir = os.path.dirname('/global_data/pretrain/wzc/deepscaler/output/sft_saved_model/Qwen2.5-1.5B-hp-dpo-tldr-20250116_dpo_epvi_mod4/checkpoint-20/aime.parquet')
        # print("output_dir:",output_dir)
        # Compute evaluation metrics
        prompts = dataset['prompt']
        responses = dataset['responses']  # Using the generated responses
        data_sources = dataset['data_source']
        reward_model_data = dataset['reward_model']
        # print("data_sources:",data_sources,reward_model_data,prompts)
        passes = 0
        total = len(dataset)
        total_scores = []

        for i in range(total):
            response_lst = responses[i]
            data_source = data_sources[i]
            prompt = prompts[i]
            reward_data = reward_model_data[i]

            ground_truth = reward_data['ground_truth']
            score_lst = []

            for r in response_lst:

                score = ground_truths_reward_func(str(r), str(ground_truth))
                # score = answer_format_reward_func(str(r))

                score_lst.append(score)

            max_score = np.max(score_lst)
            total_scores.append(score_lst)
            if max_score == 1:
                passes += 1

        n_samples = 64
        pass_at_n = passes / total
        pass_at_1 = np.mean(total_scores)

        row_data = {
            # 'model_path': config.model.path,
            # 'dataset': dataset_name,
            'pass@1': float(pass_at_1),
            f'pass@{n_samples}': pass_at_n
        }
        print(''.join(path.split('/')[-1:]))
        print(row_data)





if __name__ == '__main__':
    main()

