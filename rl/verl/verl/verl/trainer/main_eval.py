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
import ray

import hydra

import json
from omegaconf import ListConfig
from typing import List


from verl.workers.fsdp_workers import ActorRolloutRefWorker

import os

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs




class RayPPOPredictor(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.

    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.val_reward_fn = val_reward_fn

        self.init_workers(config)

        self._create_dataloader()

    def init_workers(self, config):
        """Init resource pool and worker group"""

        ray_cls_with_init = RayClassWithInitArgs(
            cls=ray.remote(ActorRolloutRefWorker),
            # cls=ray.remote(num_cpus=4, num_gpus=1)(ActorRolloutRefWorker),
            config=config,
            role='actor_rollout')

        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        
        self.actor_rollout_wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg.init_model()

        if self.config.model.ckpt_path is not None:

            print('model.ckpt_path:', self.config.model.ckpt_path)

            actor_path = os.path.join(self.config.model.ckpt_path, 'actor')

            self.actor_rollout_wg.load_checkpoint(actor_path,
                                                  del_local_after_load=False)


    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        # 对不同的评测集分别创建 dataloader 
        self.val_dataloader_list = []

        if not isinstance(self.config.data.val_files, (List, ListConfig)):
            self.config.data.val_files = [self.config.data.val_files]

        for val_file in self.config.data.val_files:

            print('create val_dataloader from val_file:', val_file)

            val_dataset = RLHFDataset(parquet_files=val_file,
                                        tokenizer=self.tokenizer,
                                        prompt_key=self.config.data.prompt_key,
                                        max_prompt_length=self.config.data.max_prompt_length,
                                        filter_prompts=True,
                                        return_raw_chat=self.config.data.get('return_raw_chat', False),
                                        truncation='error')
            
            print('Lenght of val_datase:', len(val_dataset))

            val_dataloader = DataLoader(dataset=val_dataset,
                                            batch_size=self.config.data.batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            collate_fn=collate_fn)


            assert len(val_dataloader) >= 1

            print(f'batch of val dataloader: {len(val_dataloader)}')

            self.val_dataloader_list.append(val_dataloader)

        

    def validate(self):

        metric_dict = {}

        for val_dataloader in self.val_dataloader_list:

            reward_tensor_lst = []
            data_source_lst = []

            # Lists to collect samples for the table
            sample_inputs = []
            sample_outputs = []
            sample_scores = []
            ground_truths = []

            for test_data in val_dataloader:

                test_batch = DataProto.from_single_dict(test_data)

                # Store original inputs
                input_ids = test_batch.batch['input_ids']

                print('batch size:', len(input_ids))

                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                print('world_size:', self.actor_rollout_wg.world_size)

                # pad to be divisible by dp_size
                world_size = self.actor_rollout_wg.world_size

                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, world_size)

                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad

                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

                print('validation generation end')

                # Store generated outputs
                output_ids = test_output_gen_batch.batch['responses']
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)

                ground_truths.extend([item['ground_truth'] for item in test_batch.non_tensor_batch['reward_model']])

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                reward_tensor = self.val_reward_fn(test_batch)

                # Store scores
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)

                reward_tensor_lst.append(reward_tensor)

                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))




            reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
            data_sources = np.concatenate(data_source_lst, axis=0)



            data_source = data_sources[0]

            metric_dict[f'val/test_score/{data_source}'] = reward_tensor.mean().item()
        

            # 组合每一行数据成一个字典，并添加到一个列表中
            combined_data = []
            for inp, out, label, score, source in zip(sample_inputs, sample_outputs, ground_truths, sample_scores, data_sources):
                entry = {
                    "prompt": inp,
                    "response": out,
                    "ground_truth":label,
                    "score": score,
                    "data_source": source
                }
                combined_data.append(entry)


            if not os.path.exists(self.config.data.output_path):
                os.makedirs(self.config.data.output_path)

            # 将组合后的数据写入 JSON 文件
            out_path = os.path.join(self.config.data.output_path ,f"res_{data_source}.json")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(combined_data, f, ensure_ascii=False, indent=4)

            print(f"评测结果写入文件{out_path} ")



        return metric_dict




@hydra.main(config_path='config', config_name='generation_xrh', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # local_path = copy_local_path_from_hdfs(config.model.path)

    local_path = config.model.path
    
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(config.model.path)

    from verl.workers.reward_manager.xrh_math import  RewardManagerVal
    reward_manager_cls_val = RewardManagerVal

    # Note that we always use function-based RM for validation
    val_reward_fn = reward_manager_cls_val(tokenizer=tokenizer, num_examine=1)

    trainer = RayPPOPredictor(config=config,
                            tokenizer=tokenizer,

                            val_reward_fn=val_reward_fn)
    
    print(trainer.validate())


if __name__ == '__main__':
    main()
