import json
import math
import os
import re
import string
from datasets import load_dataset
from collections import defaultdict
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm
from json_utils import *
from repeated_method import *
def count_boxed_responses(json_path):
    """
    统计文本中 boxed 的个数
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = 0
    boxed_count = 0

    for item in data:
        # 确保 item 是字典且包含 response 字段
        if isinstance(item, dict) and 'response' in item:
            total += 1
            if '\\boxed{' in item['response']:
                boxed_count += 1

    if total == 0:
        print("没有找到任何 response 字段。")
        return

    ratio = boxed_count / total
    print(f"总共有 {total} 个 response，其中 {boxed_count} 个包含 \\boxed{{。")
    print(f"包含 \\boxed{{ 的比例是: {ratio:.2%}")



# rethink_keywords=['wait', 'recheck', 'alternatively', 'retry', 'however', 'rethink','but',
#                   'another','more','different','try','incorrect','might','re-evaluate', 'perhaps', 'first',
#                   'step-by-step', 'step', 'recall', 'check']

rethink_keywords=['wait', 'recheck', 'alternatively', 'retry', 'however', 'but', 'verify',
                  'step-by-step', 'step', 'recall', 'check', 'since']


def count_keywords_in_json(json_file_path, keywords=rethink_keywords):
    """
    统计关键词的数量

    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        keyword_counts = {keyword: 0 for keyword in keywords}
        for item in data:

            response = item.get('response', '').lower()

            for keyword in keywords:
                if keyword.lower() in response:
                    keyword_counts[keyword] += 1

        print('keyword sum:', sum(keyword_counts.values()))

        return keyword_counts
    except FileNotFoundError:
        print("错误: 文件未找到!")
    except json.JSONDecodeError:
        print("错误: 无法解析JSON文件!")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")
    return None


def download_math500(output_path, format='json') -> None:
    # 1. 下载 MATH-500 数据集（split="test" 包含所有 500 条题目）
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    if format == 'jsonl':

        # 2. 打开输出文件，用 utf-8 编码
        with open(output_path, "w", encoding="utf-8") as f:
            # 3. 遍历 Dataset 中的每一条记录，以 JSON Lines 形式写入
            for example in ds:
                # example 本身就是一个 dict，键为 “question”、“answer”、“category”、“file”等
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    else:
        # 2. 转为 Python 列表（每个元素都是一个 dict）
        data_list = list(ds)  # 也可写成 ds[:]

        # 3. 写入 JSON 文件
        #    ensure_ascii=False 保留中文／特殊字符，indent=2 美化缩进
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)


    print(f"已将 {len(ds)} 条记录保存到 {output_path}")


def tokenize_and_stats(text, tokenizer=None):
    """
    对输入文本做 tokenize，并统计 token 的相关信息。

    Args:
        text: 待处理的原始文本。
        tokenizer: 已加载的 HuggingFace Tokenizer 实例，例如
                   `AutoTokenizer.from_pretrained("gpt2")`。

    Returns:
        包含以下键的字典：
          - "tokens": List[str] 分词后的 token 文本列表
          - "token_ids": List[int] 对应 token 的 ID 列表
          - "total_tokens": int 文本被分成多少个 token
          - "token_lengths": List[int] 每个 token 文本的字符长度
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("/Users/grayson/Desktop/xrh/python_projects/llm_pre_post_train/model_zoo/Qwen2.5-Math-1.5B")

    # 1. 用 tokenizer 分词（得到 token 文本列表）
    tokens: List[str] = tokenizer.tokenize(text)

    # 2. 将 token 文本 转成 token ID
    token_ids: List[int] = tokenizer.convert_tokens_to_ids(tokens)

    # 3. 统计总 token 数
    total_tokens: int = len(tokens)


    return {
        "tokens": tokens,
        "token_ids": token_ids,
        "total_tokens": total_tokens,
    }

def stats_solution_length_by_level(json_path,
                                   mode='token_length',
                                   print_log=False,
                                   stats_mode='',
                                   level_range=[1,5],
                                   ):
    """
    读取 math500.json 文件，统计不同难度(level)的 solution 字段的平均长度。

    Args:
        json_path: math500.json 文件路径

    Returns:
        一个字典，键为 level，值为 (average_length, count)，
        其中 average_length 是该等级所有 solution 长度的平均值，
        count 是该等级下包含 solution 的记录数。
    """
    # 读入 JSON 数组
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 累加各 level 的 solution 长度和计数
    sums = defaultdict(int)
    counts = defaultdict(int)

    # 初始化一下
    for i in range(level_range[0], level_range[1]+1):
        sums[i] = 0

    total_len = 0
    total_count = 0

    for entry in data:

        if 'level' in entry:
            level = entry.get("level", "unknown")

        elif 'difficulty' in entry:
            level = entry.get("difficulty", "unknown")


        try:
            lvl_num = float(level)
            lvl_key = math.floor(lvl_num) # 向下取整

        except Exception as e:
            print(e)
            lvl_key = "unknown"

        if 'response' in entry:
            response = entry.get("response")

        elif 'solution' in entry:
            response = entry.get("solution")

        else:
            raise Exception('key response not in entry')

        if isinstance(response, str) and response:

            if mode == 'token_length':

                length = entry.get("response_token_len")

            else:
                length = len(response)

            sums[lvl_key] += length
            counts[lvl_key] += 1

            total_len += length
            total_count += 1

    # 计算整体的平均值
    overall = total_len / total_count

    if print_log:
        print("整体的平均长度: {}, 样本总个数: {}".format(overall, total_count))

    # 计算不同 level 的平均值
    result = {}
    by_level = {}
    for level, total_len in sums.items():
        cnt = counts[level]
        by_level[level] = total_len / cnt

        result[level] = (total_len / cnt, cnt)

    if print_log:
        for lvl, (avg_len, cnt) in sorted(result.items()):
            print(f"Level {lvl!r}: 平均长度 {avg_len:.2f} ，样本数 {cnt}")

    return by_level, overall


def clean_prompt(prompt: str) -> str:
    prefix = "system\nPlease reason step by step, and put your final answer within \\boxed{}.\nuser\n"
    suffix = "\nassistant\n"
    if prompt.startswith(prefix):
        prompt = prompt[len(prefix):]
    if prompt.endswith(suffix):
        prompt = prompt[:-len(suffix)]
    return prompt.strip()


def merge_result_and_problem(result_path, problem_path):
    """合并结果文件和问题文件，输出到 res_all_xxx.json"""


    # 读取两个文件
    result_data = load_json(result_path)
    problem_data = load_json(problem_path)

    # 构建问题的映射：problem_text -> problem_info
    problem_map = { item['problem'].strip(): item for item in problem_data }

    merged_data = []

    for item in result_data:
        cleaned_prompt = clean_prompt(item['prompt'])
        problem_info = problem_map.get(cleaned_prompt)

        if problem_info is None:
            print(f"Warning: 没找到匹配的问题: {cleaned_prompt[:50]}...")
            continue


        if 'level' in problem_info:
            difficulty = problem_info["level"]

        elif 'difficulty' in problem_info:
            difficulty = problem_info["difficulty"]

        else:
            raise Exception('key difficulty not in problem info')

        response = item['response']
        response_token_len =tokenize_and_stats(response)['total_tokens'] # 分词并统计耗时较长

        merged_item = {
            **item,
            **{
                "problem": problem_info.get("problem", ""),
                "solution": problem_info.get("solution", ""),
                "difficulty": difficulty,
                "response_token_len": response_token_len,
                # "year": problem_info.get("year"),
                # "subject": problem_info.get("subject"),
                # "problem_number": problem_info.get("problem_number"),
                # "difficulty": problem_info.get("difficulty"),
            }
        }


        merged_data.append(merged_item)

    # 输出文件路径
    dirname, basename = os.path.split(result_path)
    if basename.startswith("res_"):
        output_basename = basename.replace("res_", "res_all_", 1)
    else:
        output_basename = "res_all_" + basename

    output_path = os.path.join(dirname, output_basename)

    save_json(output_path, merged_data)

    print(f"成功保存到: {output_path}")


def all_merge_result_and_problem(result_path_root, problem_path, datasetname='AIME2024'):
    """
    遍历指定路径下所有以 result_suffix 为文件名后缀的结果文件，
    与问题文件进行关联，生成结果文件。
    """

    # 将 os.walk 包装为列表，以便 tqdm 能显示总进度
    walk_items = list(os.walk(result_path_root))

    # 遍历路径下所有文件
    for root, dirs, files in tqdm(walk_items):
        for file in files:
            # 检查文件名是否以 "AIME2024.json" 结尾
            if file.endswith('res_{}.json'.format(datasetname)):
                result_path = os.path.join(root, file)
                print(f"Processing {result_path}...")

                # 生成新的文件路径
                merge_result_and_problem(result_path, problem_path)



def stats_score_by_level_and_overall(json_path: str,
                                     print_log=False,
                                     stats_mode='',
                                     level_range=[1,5],
                                     ) :
    """
    读取评测结果文件，统计不同难度(level)的记录的 score 字段的平均得分，
    并计算整个数据集的平均得分。

    Args:
        json_path: math500.json 文件路径

    Returns:
        by_level: 一个字典，键为 level（向下取整为整数）或 "unknown"，值为 (average_score, count)
        overall: 一个元组，包含整体平均分和记录数
    """
    # 读入 JSON 数组
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 累加各 level 的 score 长度和计数，同时统计全局得分
    sums = defaultdict(float)
    counts = defaultdict(int)

    # 初始化一下
    for i in range(level_range[0], level_range[1]+1):
        sums[i] = 0

    total_score = 0
    total_count = 0

    for entry in data:
        score = entry.get("score", None)
        if score is None:
            continue  # 如果没有 score 字段，则跳过该记录

        if 'level' in entry:
            level = entry.get("level", "unknown")

        elif 'difficulty' in entry:
            level = entry.get("difficulty", "unknown")

        try:
            lvl_num = float(level)
            lvl_key = int(lvl_num)  # 向下取整
        except Exception:
            lvl_key = "unknown"


        # 累计每个 level 的得分总和和样本数
        sums[lvl_key] += score
        counts[lvl_key] += 1

        # 统计全局的得分和样本数
        total_score += score
        total_count += 1

    overall = total_score / total_count

    if print_log:
        print("整体的平均分: {}, 样本总个数: {}".format(overall, total_count))

    # 计算不同 level 的平均值
    result = {}
    by_level = {}
    for level, total_score in sums.items():
        cnt = counts[level]
        by_level[level] = total_score / cnt

        result[level] = (total_score / cnt, cnt)

    if print_log:
        for lvl, (avg_score, cnt) in sorted(result.items()):
            print(f"Level {lvl!r}: 平均分 {avg_score:.2f} ，样本数 {cnt}")

    return by_level, overall

def tokenize(text: str) -> list[str]:
    """
    将英文文本按照空格和常见标点（不包括连字符 '-'）拆分，返回非空的 token 列表。
    """
    # 获取所有常见标点
    punct = string.punctuation
    # 去掉连字符 '-'
    punct = punct.replace('-', '')
    # 加入所有空白字符
    sep_chars = punct + string.whitespace
    # 构造正则：匹配一个或多个分隔符
    pattern = rf"[{re.escape(sep_chars)}]+"
    raw_tokens = re.split(pattern, text)
    return [tok for tok in raw_tokens if tok]

def stats_rethink_word_by_level_and_overall(json_path: str,
                                            print_log=False,
                                            keywords=rethink_keywords,
                                            stats_mode='avg_rethink_num',
                                            level_range=[1,5],
                                            ) :
    """
    读取评测结果文件，统计不同难度(level)的记录的 rethink 关键词的个数，
    并计算整个数据集的平均得分。

    Args:
        json_path: math500.json 文件路径
        return_ratio: 返回的是个数还是比例

    Returns:
        by_level: 一个字典，键为 level（向下取整为整数）或 "unknown"，值为 (average_score, count)
        overall: 一个元组，包含整体平均分和记录数
    """
    # 读入 JSON 数组
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 累加各 level 的 关键词个数，同时统计全局的个数
    sums = defaultdict(float)
    counts = defaultdict(int)

    # 初始化一下
    for i in range(level_range[0], level_range[1]+1):
        sums[i] = 0

    total_sum = 0
    total_count = 0

    for entry in data:

        response = entry.get("response", None)

        if 'level' in entry:
            level = entry.get("level", "unknown")

        elif 'difficulty' in entry:
            level = entry.get("difficulty", "unknown")

        try:
            lvl_num = float(level)
            lvl_key = int(lvl_num)  # 向下取整
        except Exception:
            lvl_key = "unknown"

        response_list = tokenize(response)

        keyword_counts = {keyword: 0 for keyword in keywords}
        for keyword in keywords:
            if keyword.lower() in response_list:
                keyword_counts[keyword] += 1

        if print_log:
            print(keyword_counts)

        if stats_mode=='rethink_word_ratio': # 反思词占所有词的比例

            rethink_num = sum(keyword_counts.values())
            word_num =  len(response.split(' '))

            # 累计每个 level 的 rethink 单词的个数和总的单词个数
            sums[lvl_key] += rethink_num

            counts[lvl_key] += word_num

            # 统计全局的
            total_sum += rethink_num
            total_count += word_num

        elif stats_mode == 'avg_rethink_num': # 每个doc 的平均反思词的个数

            rethink_num = sum(keyword_counts.values())

            # 累计每个 level 的 rethink 单词的个数
            sums[lvl_key] += rethink_num

            counts[lvl_key] += 1

            # 统计全局的
            total_sum += rethink_num
            total_count += 1



    # 计算整体
    overall = total_sum / total_count if total_count > 0 else 0

    # 计算各 level
    by_level: Dict[Union[int, str], Tuple[float, int]] = {}
    for lvl_key, total_score in sums.items():
        count = counts[lvl_key]
        avg_score = total_score / count if count > 0 else 0
        by_level[lvl_key] = avg_score


    if print_log:

        print("各难度平均反思词的比例：")
        for lvl, avg_score in sorted(by_level.items(), key=lambda x: (isinstance(x[0], int), x[0])):
            print(f"  Level {lvl!r}: 平均得分 {avg_score:.4f}")

        print(f"\n整体的反思词的比例: {overall:.4f}，词的总个数 {total_count}")

    return by_level, overall

from typing import List



def get_repetition_degree(ngram_size: int, generation: str) -> float:
    """
    ngram_size = 40

    """

    ngrams = {}

    ngram_cnt = 0 # ngram 短语的总个数

    for ng in zipngram(generation, ngram_size): # 生成所有的 n-gram 短句

        if ng in ngrams:
            ngrams[ng] += 1

        else:
            ngrams[ng] = 0

        ngram_cnt += 1

    ngrams_dict = {}
    repetition_ngram_num = 0

    for ng in ngrams.keys():

        if ngrams[ng] > 1: # 只有重复度 大于1次 的才被记录

            ngrams_dict[' '.join(ng)] = ngrams[ng]
            repetition_ngram_num += 1

    return ngrams_dict, repetition_ngram_num





def stats_repetition_by_level_and_overall(json_path: str,
                                            print_log=False,
                                            keywords=rethink_keywords,
                                            stats_mode='avg_n_gram_rep_num',
                                            level_range=[1,5],
                                            ) :
    """
    读取评测结果文件，统计不同难度(level)的记录的 n-gram 的重复度，
    并计算整个数据集的平均得分。

    Args:
        json_path: math500.json 文件路径
        return_ratio: 返回的是个数还是比例

    Returns:
        by_level: 一个字典，键为 level（向下取整为整数）或 "unknown"，值为 (average_score, count)
        overall: 一个元组，包含整体平均分和记录数
    """
    # 读入 JSON 数组
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 累加各 level 的 关键词个数，同时统计全局的个数
    sums = defaultdict(float)

    # 初始化一下
    for i in range(level_range[0], level_range[1]+1):
        sums[i] = 0

    counts = defaultdict(int)

    total_sum = 0
    total_count = 0

    for entry in data:

        response = entry.get("response", None)

        if 'level' in entry:
            level = entry.get("level", "unknown")

        elif 'difficulty' in entry:
            level = entry.get("difficulty", "unknown")

        try:
            lvl_num = float(level)
            lvl_key = int(lvl_num)  # 向下取整
        except Exception:
            lvl_key = "unknown"


        if stats_mode == 'avg_n_gram_rep_num': # 每个doc 的平均 N-gram 重复的个数

            ngrams, ten_gram_rep_num = get_repetition_degree(40, response)

            if print_log:
                if len(ngrams) >0:
                    print(ngrams)

            # 累计每个 level 的 N-gram 重复的个数
            sums[lvl_key] += ten_gram_rep_num

            counts[lvl_key] += 1 # 每个 level 的 doc 的个数

            # 统计全局的
            total_sum += ten_gram_rep_num
            total_count += 1

        elif stats_mode == 'longest_repeated_substring_ratio': # 最长重复子串的比例

            max_substr, max_substr_cnt = find_longest_repeated_substring(response, rep_length_threshold=200)

            if max_substr_cnt > 0:
                if print_log:
                    print({max_substr: max_substr_cnt})

                # 累计每个 level 的重复的子串的总长度
                sums[lvl_key] += (len(max_substr) * max_substr_cnt)
                counts[lvl_key] += len(response) # 每个 level 的response的总长度

                # 统计全局的
                total_sum += (len(max_substr) * max_substr_cnt)
                total_count += len(response)

            else:

                counts[lvl_key] += len(response)
                total_count += len(response)



    # 计算整体
    overall = total_sum / total_count if total_count > 0 else 0

    # 计算各 level
    by_level: Dict[Union[int, str], Tuple[float, int]] = {}
    for lvl_key, total_score in sums.items():
        count = counts[lvl_key]
        avg_score = total_score / count if count > 0 else 0
        by_level[lvl_key] = avg_score


    if print_log:

        print("各难度下，平均一个doc中 n-gram重复的个数")
        for lvl, avg_score in sorted(by_level.items(), key=lambda x: (isinstance(x[0], int), x[0])):
            print(f"  Level {lvl!r}: 平均一个doc中 n-gram重复的个数 {avg_score:.4f}")

        print(f"\n整体的平均一个doc中 n-gram重复的个数: {overall:.4f}，doc的总个数 {total_count}")

    return by_level, overall




def level_func_trends(base_path, stats_func, dataset_suffix="AIME2024", start=0, end=float('inf'), stats_mode='', level_range=[]):
    """
    遍历指定路径下所有以指定数据集名为文件名后缀的结果文件，
    利用 `stats_func` 函数统计

    Args:
        base_path: 存放训练结果文件的路径
        dataset_suffix: 文件名后缀（例如 "AIME2024"）
    """
    steps = []
    level_scores = defaultdict(list)
    overall_scores = []

    # 遍历路径下所有以 dataset_suffix 为文件名后缀的文件
    for root, _, files in os.walk(base_path):

        for file in files:

            # print('file:', file)

            if file.endswith("all_" + dataset_suffix + ".json"):
                # 提取训练步数（从文件名中获取）
                try:
                    step = int(root.split("/")[-1])  # 从路径中提取训练步数

                except ValueError:
                    continue  # 如果路径解析错误，跳过该文件

                if step >= start and step <= end:

                    steps.append(step)

                    # 获取文件完整路径
                    file_path = os.path.join(root, file)

                    # 统计当前文件的分数
                    by_level, overall = stats_func(file_path, stats_mode=stats_mode, level_range=level_range)

                    # 按每个 level 统计得分
                    for level, avg_score in by_level.items():
                        level_scores[level].append((step, avg_score))

                    # 记录整体平均得分
                    overall_scores.append((step, overall))

    # 对 level 进行排序，确保图例按从 1 到 5 排序
    sorted_levels = sorted(level_scores.keys())

    overall_scores = sorted(overall_scores, key=lambda x: x[0])  # 按训练步数排序

    level_score_dict = {}

    for level in sorted_levels:
        scores = level_scores[level]
        scores = sorted(scores, key=lambda x: x[0])  # 按训练步数排序
        steps_sorted, avg_scores = zip(*scores)

        level_score_dict[level] = (steps_sorted, avg_scores)

    return level_score_dict, overall_scores

def plot_score_trends(base_path, dataset_suffix = "AIME2024", save_image = True, image_filename = "score_trends.png") -> None:
    """
    遍历指定路径下所有以指定数据集名为文件名后缀的结果文件，
    利用 `stats_score_by_level_and_overall` 函数统计并绘制训练步数与得分的折线图。

    Args:
        base_path: 存放训练结果文件的路径
        dataset_suffix: 文件名后缀（例如 "AIME2024"）
    """
    steps = []
    level_scores = defaultdict(list)
    overall_scores = []

    # 遍历路径下所有以 dataset_suffix 为文件名后缀的文件
    for root, _, files in os.walk(base_path):

        for file in files:

            # print('file:', file)

            if file.endswith("all_"+dataset_suffix + ".json"):
                # 提取训练步数（从文件名中获取）
                try:
                    step = int(root.split("/")[-1])  # 从路径中提取训练步数

                except ValueError:
                    continue  # 如果路径解析错误，跳过该文件

                steps.append(step)

                # 获取文件完整路径
                file_path = os.path.join(root, file)

                # 统计当前文件的分数
                # print('step:', step)
                by_level, overall = stats_score_by_level_and_overall(file_path)


                # 按每个 level 统计得分
                for level, (avg_score, _) in by_level.items():
                    level_scores[level].append((step, avg_score))

                # 记录整体平均得分
                overall_scores.append((step, overall))

                # print('='*20)


    # 对 level 进行排序，确保图例按从 1 到 5 排序
    sorted_levels = sorted(level_scores.keys())

    # 绘制折线图
    plt.figure(figsize=(10, 6))

    for level in sorted_levels:
        scores = level_scores[level]
        scores = sorted(scores, key=lambda x: x[0])  # 按训练步数排序
        steps_sorted, avg_scores = zip(*scores)
        plt.plot(steps_sorted, avg_scores, marker='o', markersize=4, linestyle='-',label=f"Level {level}")

    # 绘制整体平均得分曲线
    overall_scores = sorted(overall_scores, key=lambda x: x[0])  # 按训练步数排序
    steps_sorted, avg_overall_scores = zip(*overall_scores)
    plt.plot(steps_sorted, avg_overall_scores, label="Overall Average", linestyle="--", color="black", linewidth=2)


    plt.xlabel("Training Steps", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
    plt.title(dataset_suffix, fontsize=16)
    plt.legend(title="Level")
    plt.grid(True)

    # 保存图像
    if save_image:

        image_path = os.path.join(base_path, '{}_{}'.format(dataset_suffix,image_filename))

        plt.savefig(image_path)
        print(f"图像已保存到：{image_path}")

    plt.show()


def plot_length_trends(base_path: str, dataset_suffix: str = "AIME2024", save_image: bool = True, image_filename: str = "length_trends.png") -> None:
    """
    遍历指定路径下所有以指定数据集名为文件名后缀的结果文件，
    利用 `stats_solution_length_by_level` 函数统计并绘制训练步数与平均长度的折线图。

    Args:
        base_path: 存放训练结果文件的路径
        dataset_suffix: 文件名后缀（例如 "AIME2024"）
    """
    steps = []
    level_scores = defaultdict(list)
    overall_scores = []

    # 遍历路径下所有以 dataset_suffix 为文件名后缀的文件
    for root, _, files in os.walk(base_path):

        for file in files:

            # print('file:', file)

            if file.endswith("all_"+dataset_suffix + ".json"):
                # 提取训练步数（从文件名中获取）
                try:
                    step = int(root.split("/")[-1])  # 从路径中提取训练步数

                except ValueError:
                    continue  # 如果路径解析错误，跳过该文件

                steps.append(step)

                # 获取文件完整路径
                file_path = os.path.join(root, file)

                # 统计当前文件的分数
                # print('step:', step)

                by_level, overall = stats_solution_length_by_level(file_path)

                # 按每个 level 统计得分
                for level, (avg_score, _) in by_level.items():
                    level_scores[level].append((step, avg_score))

                # 记录整体平均得分
                overall_scores.append((step, overall))

                # print('='*20)

    # 对 level 进行排序，确保图例按从 1 到 5 排序
    sorted_levels = sorted(level_scores.keys(), reverse=False)

    # 绘制折线图
    plt.figure(figsize=(10, 6))

    for level in sorted_levels:
        scores = level_scores[level]
        scores = sorted(scores, key=lambda x: x[0])  # 按训练步数排序
        steps_sorted, avg_scores = zip(*scores)
        plt.plot(steps_sorted, avg_scores, marker='o', markersize=4, linestyle='-', label=f"Level {level}")


    # 绘制整体平均得分曲线
    overall_scores = sorted(overall_scores, key=lambda x: x[0])  # 按训练步数排序
    steps_sorted, avg_overall_scores = zip(*overall_scores)
    plt.plot(steps_sorted, avg_overall_scores, label="Overall Average",  marker='o', markersize=4, linestyle="--", color="black", linewidth=2)


    plt.xlabel("Training Steps", fontsize=14, fontweight='bold')
    plt.ylabel("Average Length", fontsize=14, fontweight='bold')
    plt.title(dataset_suffix, fontsize=16)
    plt.legend(title="Level")
    plt.grid(True)

    # 保存图像
    if save_image:
        image_path = os.path.join(base_path, '{}_{}'.format(dataset_suffix,image_filename))
        plt.savefig(image_path)
        print(f"图像已保存到：{image_path}")

    plt.show()

def compute_accuracy_avg(data, start=0, end=float('inf'), mode = 'arithmetic_avg'):
    """
    将 3个评测集合并到一起计算平均分

    # start , end 控制输出的步数

    """

    steps = []
    acc = []

    for entry in data:

        step = entry['step']

        if step >= start and step <= end:

            if "val/test_score/MATH500" in entry:
                steps.append(int(step))

                if mode == 'weighted_avg':

                    avg_score = (entry['val/test_score/MATH500'] * 500 + entry['val/test_score/AIME2024'] * 30 + entry[
                        'val/test_score/AMC2023'] * 83) / (500 + 30 + 83)

                elif mode == 'arithmetic_avg':

                    avg_score = (entry['val/test_score/MATH500'] + entry['val/test_score/AIME2024'] + entry[
                        'val/test_score/AMC2023'] ) / 3

                elif mode == 'avg_2':

                    avg_score1 = (entry['val/test_score/MATH500'] * 500 + entry['val/test_score/AIME2024'] * 30 + entry[
                        'val/test_score/AMC2023'] * 83) / (500 + 30 + 83)

                    avg_score2 = (entry['val/test_score/MATH500'] + entry['val/test_score/AIME2024'] + entry[
                        'val/test_score/AMC2023'] ) / 3

                    avg_score= (avg_score1+avg_score2)/2


                acc.append(avg_score)

    # 排序并解包
    sorted_pairs = sorted(zip(steps, acc))
    steps, acc = map(list, zip(*sorted_pairs))

    return steps, acc


def weighted_average_with_step(*lists, weights):
    """
    接收多个 [(step, value)] 格式的列表和对应权重，输出 [(step, weighted_value)]。
    """
    if not lists:
        raise ValueError("必须至少提供一个列表。")

    num_lists = len(lists)
    length = len(lists[0])

    if any(len(lst) != length for lst in lists):
        raise ValueError("所有输入的列表必须等长。")
    if len(weights) != num_lists:
        raise ValueError("权重数量必须与列表数量一致。")

    weight_sum = sum(weights)
    norm_weights = [w / weight_sum for w in weights]

    result = []
    for i in range(length):
        step = lists[0][i][0]  # 所有列表的 step 应一致
        weighted_value = sum(norm_weights[j] * lists[j][i][1] for j in range(num_lists))
        result.append((step, weighted_value))

    return result


if __name__ == '__main__':


    # all_merge_result_and_problem(result_path_root='../data/llm_outputs/grpo-deepscaler_17k-7b_math-bs_128-3k-zs_qwen-r_a_g_f/', problem_path='../data/download/aime2024.json', datasetname='AIME2024')


    # all_merge_result_and_problem(
    #     result_path_root='../data/llm_outputs/grpo-deepscaler_17k-7b_math-bs_128-3k-zs_qwen-r_f_l_v3_token5_a-v3/',
    #     problem_path='../data/download/aime2024.json', datasetname='AIME2024')


    # all_merge_result_and_problem(
    #     result_path_root='../data/llm_outputs/grpo-deepscaler_17k-7b_math-bs_128-3k-zs_qwen-r_f_a/',
    #     problem_path='../data/download/aime2024.json', datasetname='AIME2024')

    # all_merge_result_and_problem(
    #     result_path_root='../data/llm_outputs/grpo-math-7b_math-bs_128-3k-zs_qwen-r_f_l_v3_token5_a-v3/',
    #     problem_path='../data/download/aime2024.json', datasetname='AIME2024')

    all_merge_result_and_problem(
        result_path_root='../data/llm_outputs/grpo-math-7b_math-bs_128-3k-zs_qwen-r_a_g_f/',
        problem_path='../data/download/aime2024.json', datasetname='AIME2024')


    # all_merge_result_and_problem(result_path_root='../data/llm_outputs/grpo-deepscaler_17k-7b_math-bs_128-3k-zs_qwen-r_a_g_f/', problem_path='../data/download/math500.json', datasetname='MATH500')

    # all_merge_result_and_problem(result_path_root='../data/llm_outputs/grpo-deepscaler_17k-7b_math-bs_128-3k-zs_qwen-r_f_a/',
    #         problem_path='../data/download/math500.json', datasetname='MATH500')

    # all_merge_result_and_problem(result_path_root='../data/llm_outputs/grpo-deepscaler_17k-7b_math-bs_128-3k-zs_qwen-r_f_l_v3_token5_a-v3/',
    #         problem_path='../data/download/math500.json', datasetname='MATH500')








