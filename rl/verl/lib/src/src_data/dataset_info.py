
import json
import os 
import numpy as np
import pandas as pd

import pyarrow.ipc as ipc
import pyarrow as pa
import pandas as pd
import json
import os

def default_converter(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def parquet_to_json(parquet_file):
    # 自动生成与 Parquet 文件同名的 JSON 文件名
    base_name, _ = os.path.splitext(parquet_file)
    json_file = base_name + ".json"
    
    # 读取 Parquet 文件
    df = pd.read_parquet(parquet_file)
    
    # 将 DataFrame 转换为字典列表
    data = df.to_dict(orient="records")

    print('记录数：', len(data))
    
    # 写入 JSON 文件，并转换 ndarray 为 list
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=default_converter)
    
    print(f"转换成功，生成文件：{json_file}")

def read_parquet_and_print(file_path):
    try:
        # 读取 Parquet 文件
        df = pd.read_parquet(file_path)

        # 设置显示选项，避免省略信息
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.max_rows', 100)  # 避免输出过长
        pd.set_option('display.width', 10000)
        
        print(f"DataFrame 共有 {len(df)} 条记录")

        print('columns:', df.columns)

        # print()

        # 打印前 1 条数据
        print(df[['prompt']].head(2).to_string(index=False))

        print(df[['data_source']].head(1).to_string(index=False))
        
        return df
    except Exception as e:
        print(f"Error reading parquet file: {e}")





if __name__ == '__main__':


    data_path = '../../data/qwen_template/deepscaler_17k.parquet'

    parquet_to_json(data_path)



