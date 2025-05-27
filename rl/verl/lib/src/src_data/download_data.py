from huggingface_hub import snapshot_download
from datasets import load_dataset, Dataset
import os
import json
import random



def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def extract_math_answer(text):

    # 找到 \boxed 的起始位置
    start = text.find("\\boxed")
    if start == -1:
        return None  # 如果没有找到 \boxed，则返回 None

    # print(text[start:])

    first = start + len("\\boxed")
    stack = []

    for i in range(first, len(text)):
        char = text[i]
        # print('char:', char)

        if char == '{':
            stack.append(i)  # 遇到 {，入栈

        elif char == '}':  # 遇到 }，出栈

            if len(stack) > 0:

                left = stack.pop()
                if left == first:

                    return text[first+1:i]  # 返回 \boxed{} 内的内容

    return None  # 如果没有找到匹配的括号，返回 None


def download_eleutherAI___hendrycks_by_huggingface(cache_directory, out_json_path):
    """
    下载 EleutherAI/hendrycks_math 数据集

    用 ___ 连接名字 作为目录

    EleutherAI/hendrycks_math -> EleutherAI___hendrycks_math

    """

    # 判断缓存目录是否存在，如果不存在则创建
    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)

    snapshot_download(repo_id="EleutherAI/hendrycks_math",
                      local_dir=os.path.join(cache_directory, "EleutherAI___hendrycks_math"),
                      local_dir_use_symlinks=False, repo_type="dataset")

    partitions = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory',
                  'prealgebra', 'precalculus']

    # 用于存储所有分区的训练数据
    all_train_data = []

    for partition in partitions:
        try:
            # 加载特定分区的数据集
            dataset = load_dataset("EleutherAI/hendrycks_math", partition, cache_dir=cache_directory,
                                   download_mode="reuse_cache_if_exists")
            # 获取训练集数据
            train_data = dataset["train"]
            # 将当前分区的训练数据添加到总数据列表中
            all_train_data.extend(train_data)

        except Exception as e:
            print(f"加载 {partition} 分区时出现错误: {e}")

    # 打印选取的数据数量
    print(f"数据数量: {len(all_train_data)}")

    for ele in all_train_data:
        ele['answer'] = extract_math_answer(ele['solution'])

    print('显示1条数据:', all_train_data[0])

    save_json(out_json_path, Dataset.from_list(all_train_data).to_list())
    print('数据生成成功:', out_json_path)

if __name__ == '__main__':

    download_eleutherAI___hendrycks_by_huggingface('train/', 'train/math_train.json')

