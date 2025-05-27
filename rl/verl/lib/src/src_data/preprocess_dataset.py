
import os
from typing import Dict, List, Optional, Any

import pandas as pd

from src.src_data.utils import load_dataset
from src.src_data.dataset_types import TrainDataset, TestDataset




SYSTEM_PROMPT_qwen25 = "Please reason step by step, and put your final answer within \\boxed{}."




def make_map_fn(split: str, system_prompt, data_source):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """

    def process_fn_zs(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """
        zero shot 


        """

        if 'problem' in  example:

            question = example.pop('problem')
        
        elif 'prompt' in  example:
            question = example.pop('prompt')
        
        else:
            raise ValueError("数据中没有 problem / prompt 字段")

        # instruction = "Let's think step by step and output the final answer within \\boxed{}."

        answer = str(example.pop('answer'))

        if answer is not None and len(answer) > 0 :

            data = {
                "data_source": data_source,
                "prompt": [

                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': question}
                
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data
        
        else:
            return None



    return process_fn_zs


def process_train_data(local_dir, system_prompt, sample_n=-1):
    """
    处理训练数据
    """

    # train_datasets = [TrainDataset.MATH]

    train_datasets = [TrainDataset.DEEPSCALER]

    train_dataset = load_dataset(train_datasets[0])

    # Process training data
    train_data: List[Dict[str, Any]] = []

    process_fn = make_map_fn('train', system_prompt, 'math')

    for idx, example in enumerate(train_dataset):

        processed_example = process_fn(example, idx)

        if processed_example is not None:
            train_data.append(processed_example)

     # Save training dataset
    print("train data size:", len(train_data))

    train_df = pd.DataFrame(train_data)

    if sample_n > 0:

        train_df = train_df.sample(n=sample_n, random_state=42)

    print("sample data size:", len(train_df))

    train_df.to_parquet(os.path.join(local_dir, 'deepscaler_17k.parquet'))

def process_test_data(local_dir, system_prompt):
    """
    处理测试数据
    """

    # test_datasets = [TestDataset.AIME, TestDataset.AMC, TestDataset.MATH, TestDataset.MINERVA, TestDataset.OLYMPIAD_BENCH]
    
    test_datasets = [TestDataset.MATH, TestDataset.AIME, TestDataset.AMC]


    test_datasets_data = [load_dataset(d) for d in test_datasets]

    # Process and save each test dataset separately
    for test_dataset, test_data_list in zip(test_datasets, test_datasets_data):

        test_data: List[Dict[str, Any]] = []

        process_fn = make_map_fn('test', system_prompt, test_dataset.value)

        for idx, example in enumerate(test_data_list):
            processed_example = process_fn(example, idx)
            if processed_example is not None:
                test_data.append(processed_example)

        dataset_name = test_dataset.value.lower()
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(local_dir, f'{dataset_name}.parquet'))
        print(f"{dataset_name} test data size:", len(test_data))


def arrow_to_json(arrow_file_path):
    # 检查文件是否存在
    if not os.path.exists(arrow_file_path):
        raise FileNotFoundError(f"找不到文件：{arrow_file_path}")

    # 自动生成 .json 文件路径（同名）
    base_name = os.path.splitext(arrow_file_path)[0]
    json_file_path = base_name + '.json'

    # 打开并读取 .arrow 文件
    with open(arrow_file_path, 'rb') as f:
        reader = ipc.RecordBatchFileReader(f)
        table = reader.read_all()

    # 转为 pandas DataFrame，再转字典
    df = table.to_pandas()
    records = df.to_dict(orient='records')

    # 写入 JSON 文件
    with open(json_file_path, 'w', encoding='utf-8') as out_json:
        json.dump(records, out_json, ensure_ascii=False, indent=2)

    print(f"✅ 成功转换：{arrow_file_path} → {json_file_path}")


def arrow_to_json_auto(arrow_file_path):
    base_name = os.path.splitext(arrow_file_path)[0]
    json_file_path = base_name + '.json'

    with open(arrow_file_path, 'rb') as f:
        try:
            reader = ipc.RecordBatchFileReader(f)
        except pa.lib.ArrowInvalid:
            f.seek(0)
            reader = ipc.RecordBatchStreamReader(f)

        table = reader.read_all()

    df = table.to_pandas()

    # 转为 list，并处理 ndarray 等不能直接 JSON 序列化的对象
    def safe_convert(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    # 应用转换
    records = df.applymap(safe_convert).to_dict(orient='records')

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(records, json_file, ensure_ascii=False, indent=2)

    print(f"✅ 成功转换：{arrow_file_path} → {json_file_path}")





if __name__ == '__main__':


    local_dir = '../../data/qwen_template'

    process_train_data(local_dir, SYSTEM_PROMPT_qwen25, 8500*2)

    # process_test_data(local_dir, SYSTEM_PROMPT_qwen25)

