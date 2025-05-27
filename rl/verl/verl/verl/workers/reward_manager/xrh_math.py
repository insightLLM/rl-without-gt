from verl import DataProto
import torch


from verl.utils.reward_score import gsm8k, math


from src.rewards.math_reward import xrh_math_reward_fn
from src.eval.evaluator import MATHEvaluator

def _select_rm_score_fn(data_source):

    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    
    else:
        return xrh_math_reward_fn


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}


        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # sequences = torch.cat((valid_prompt_ids, valid_response_ids)) # 将 prompt 和 response 拼起来
            
            sequences_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']


            score = xrh_math_reward_fn(solution_str=sequences_str, ground_truth=ground_truth, response_token_len=int(valid_response_length.item()))

            
            return i, score, valid_response_length



        results = []
        for i in range(len(data)):

            results.append(process_item((i, data[i], already_print_data_sources)))

        # Fill reward tensor with results
        for i, score, valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor


class RewardManagerVal():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

        self.evaluator = MATHEvaluator()

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']


            score = self.evaluator.score(prediction=sequences_str, reference=ground_truth)

            
            return i, score, valid_response_length



        results = []
        for i in range(len(data)):
            results.append(process_item((i, data[i], already_print_data_sources)))

        # Fill reward tensor with results
        for i, score, valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor
