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

if __name__ == "__main__":

    response4_1 = "Let's solve the problem step by step.\n\n### Step 1: Determine the parity of the function \\( f(x) \\)\nThe function \\( f(x) \\) satisfies the functional equation \\( f(x+y) = f(x) + f(y) \\) for any \\( x, y \\in \\mathbb{R} \\). This is known as the Cauchy functional equation. One of the properties of the solutions to this equation is that they are linear when the domain is restricted to rational numbers, i.e., \\( f(x) = cx \\) for some constant \\( c \\). However, we need to verify if \\( f(x) \\) is odd or even.\n\n#### Step 1.1: Check the parity\nTo determine the parity of \\( f(x) \\), we need to check if \\( f(-x) = f(x) \\) (even function) or \\( f(-x) = -f(x) \\) (odd function). Let's substitute \\( y = -x \\) into the functional equation:\n\\[ f(x + (-x)) = f(x) + f(-x) \\]\nSince \\( x + (-x) = 0 \\), we have:\n\\[ f(0) = f(x) + f(-x) \\]\nWe need to find \\( f(0) \\). Let's substitute \\( x = 0 \\) and \\( y = 0 \\) into the functional equation:\n\\[ f(0 + 0) = f(0) + f(0) \\]\n\\[ f(0) = 2f(0) \\]\nThis implies:\n\\[ f(0) = 0 \\]\nSo, we have:\n\\[ 0 = f(x) + f(-x) \\]\nThis implies:\n\\[ f(-x) = -f(x) \\]\nTherefore, \\( f(x) \\) is an odd function.\n\n### Step 2: Determine if \\( f(x) \\) has an extreme value on the interval \\([-3, 3]\\)\nGiven that \\( f(1) = -2 \\), we can determine the function \\( f(x) \\) for any rational number \\( x \\). Since \\( f(x) = cx \\), we can find \\( c \\) using \\( f(1) = -2 \\):\n\\[ f(1) = c \\cdot 1 = -2 \\]\nSo, \\( c = -2 \\). Therefore, \\( f(x) = -2x \\).\n\nWe need to check if \\( f(x) = -2x \\) has an extreme value on the interval \\([-3, 3]\\). Since \\( f(x) \\) is a linear function, it does not have any local extreme values (maximum or minimum). However, we need to check the values at the endpoints of the interval \\([-3, 3]\\).\n\n#### Step 2.1: Evaluate \\( f(x) \\) at the endpoints\n\\[ f(-3) = -2 \\cdot (-3) = 6 \\]\n\\[ f(3) = -2 \\cdot 3 = -6 \\]\nThe function \\( f(x) = -2x \\) is a decreasing function on the interval \\([-3, 3]\\), so the maximum value is \\( 6 \\) at \\( x = -3 \\) and the minimum value is \\( -6 \\) at \\( x = 3 \\).\n\n### Conclusion\n1. The function \\( f(x) \\) is an odd function.\n2. The function \\( f(x) = -2x \\) has an extreme value on the interval \\([-3, 3]\\). The maximum value is \\( 6 \\) at \\( x = -3 \\) and the minimum value is \\( -6 \\) at \\( x = 3 \\).\n\nLet's write the final answer in the boxed format:\n\n1. The function \\( f(x) \\) is an odd function.\n2. The function \\( f(x) \\) has an extreme value on the interval \\([-3, 3]\\). The maximum value is \\(\\boxed{6}\\) at \\( x = -3 \\) and the minimum value is \\(\\boxed{-6}\\) at \\( x = 3 \\).<|endoftext|>"
    ground_truths4_1 = "-6"

    xrh_math_reward_fn(solution_str=response4_1, ground_truth=ground_truths4_1, response_token_len=100)