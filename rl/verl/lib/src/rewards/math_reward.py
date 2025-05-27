

from src.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType


from src.eval.evaluator import MATHEvaluator_Base, MATHEvaluator

import re 
import logging
import json
import threading
import math_evaluation

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

eval_obj = MATHEvaluator()

def answer_contain_reward_func(response, ground_truths):
    """
    评估 参考答案 ground_truths（一般比较短） 是否在 模型生成的答案中，如果是 则奖励 1，否则奖励 0。
    
    """
    if ground_truths is None:
        return 0.0

    answer = eval_obj.parse_answer(response)

    if answer is None:
        answer= ""

    score = 0.0

    try:
        answer = eval_obj._strip_string(answer)
        ground_truths = eval_obj._strip_string(ground_truths)

    except:  
        pass


    if ground_truths in answer: # ground_truths 在 answer 中，answer 包含了 ground_truths
                                
        score = 1.0

    return score

def ground_truths_contain_reward_func(response, ground_truths):
    """
    评估 模型生成的答案 是否在 参考答案中，如果是 则奖励 1，否则奖励 0。

    考虑到有的题目是多个答案的，模型做出来一个就给奖励
    
    """
    if ground_truths is None:
        return 0.0

    answer = eval_obj.parse_answer(response)

    if answer is None:
        answer= ""

    score = 0.0

    try:
        answer = eval_obj._strip_string(answer)
        ground_truths = eval_obj._strip_string(ground_truths)

    except:
        pass


    logger.debug('answer: {}'.format(answer))

    if len(answer)>0 and answer in ground_truths: # answer 在 ground_truths 中，ground_truths 包含了 answer

        score = 1.0

    return score

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
    

def soft_format_reward_func(response):
    """
    检查模型输出是否包含 <think>...</think> 和 <answer>...</answer> 这两个标签，如果包含，奖励 1，否则奖励 0。
    
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>" # 允许中间有空格

    match = re.search(pattern, response, re.DOTALL)  # 检测格式是否匹配
    
    return 1.0 if match else 0.0 

def soft_format_reward_func_v2(response):
    """
    检查模型输出是否包含 <think>...</think> ，如果包含，奖励 1，否则奖励 0。
    
    """
    pattern = r"<think>.*?</think>"

    match = re.search(pattern, response, re.DOTALL)  # 检测格式是否匹配
    
    return 1.0 if match else 0.0 

def strict_format_reward_func(response):
    """
    比 soft_format_reward_func 更严格，要求完整的 XML 结构 且 不能有多余内容。如果格式完全符合，则奖励 4，否则奖励 0。
    
    """
    pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$" # 必须是完整的 XML

    match = re.search(pattern, response, re.DOTALL) 
    
    return 2.0 if match else 0.0



def length_reward_func(response):
    """
    response 长度奖励
    
    """
    max_length = 4096
    decay1_threshold = 0.3
    decay2_threshold = 0.6

    ratio = len(response)/max_length

    if ratio < decay1_threshold:

        score = ratio

    elif ratio >= decay1_threshold and ratio < decay2_threshold: # 说的越长 分数越低

        score = 1 - ratio
    
    else: #  len(response) >= (max_length*stage2_threshold) # 拿到负数的分数
        score = -ratio


    return score


def length_reward_func_v2(response):
    """
    response 长度奖励

    x = len(response)/max_length

    x 属于 [0, 1]

    当 x 属于 [0, 0.5] y=x
    当 x 属于 [0.5, 1] y = -2x+1.5
    
    """
    max_length = 4096

    decay1_threshold = 0.5

    ratio = len(response)/max_length

    if ratio < decay1_threshold:

        score = ratio

    elif ratio >= decay1_threshold: # 说的越长 分数越低

        score = (-2)*ratio + 1.5
    
    else:
        score = 0

    return score


def length_reward_func_base(response_token_len, max_length):
    """
    response 长度奖励

    """

    score = response_token_len / max_length

    return score

def length_reward_func_v3(response_token_len, decay_threshold=0.25, max_length=4096):
    """
    response 长度奖励

    \[
    f(x; p)=
    \begin{cases}
    1 - \bigl(1 - \frac{x}{p}\bigr)^2, & 0 \le x \le p,\\[0.5em]
    1 - 2\left(\frac{x-p}{1-p}\right)^2, & p < x \le 1,
    \end{cases}
    \]

    - **[0, p] 区间**：一开始快速上升（大斜率），逐渐趋于平缓，在 \(x=p\) 达到 1  
    - **[p, 1] 区间**：从 1 开始下跌，且下跌速度（绝对斜率）随着 \(x\) 增加而加快，至 \(x=1\) 到达 \(-1\)

    """

    p = decay_threshold # 阈值

    x = response_token_len/max_length


    if x<=p:

        y = 1 - (1 - x / p) ** 2

    elif x>p:

        y = 1 - 2 * ((x - p) / (1 - p)) ** 2

    # else:
    #     y = -1

    return y


def reasoning_action_reward_func(text):
    """
    统计给定文本中关键词的总出现次数，包括：
    "however", "but", "wait", "verify", "alternatively"
    
    参数：
    text (str): 输入文本
    
    返回：
    int: 总的出现次数
    """
    # 定义需要匹配的关键词列表
    reasoning_keywords = ["however", "but", "wait", "verify", "alternatively"]
    # 构造正则表达式，\b 确保是完整单词，使用 | 连接关键词
    pattern = r'\b(?:' + '|'.join(reasoning_keywords) + r')\b'
    
    # 使用 re.findall 查找所有匹配项，忽略大小写
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    count = len(matches)

    count_max = 20

    score = min(count, count_max) / count_max
    
    return score*0.5



# xrh 基于 label 的reward

def get_repetition_penalty(ngram_size: int, max_penalty: float, generation: str) -> float:
    """
    ngram_size = 40
    max_penalty = -0.05

    """

    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if max_penalty == 0:
        return 0

    ngrams = set()
    total = 0
    for ng in zipngram(generation, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    return scaling * max_penalty


# Source:
# https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
# def zipngram(text: str, ngram_size: int):
#     words = text.lower().split()
#     return zip(*[words[i:] for i in range(ngram_size)])
def zipngram(text: str, ngram_size: int):
    words = text.lower().split()
    for i in range(len(words) - ngram_size + 1):
        yield tuple(words[i:i + ngram_size])




class RewardMathFn_xrh(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:

        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        response_token_len = input.response_token_len

        ground_truths = input.ground_truth.get("answer", None)
        
        score = 0

        ground_truths_contain_reward = ground_truths_contain_reward_func(model_response, ground_truths)

        ground_truths_reward = ground_truths_reward_func(model_response, ground_truths)

        soft_format_reward = soft_format_reward_func_v2(model_response)

        # strict_format_reward = strict_format_reward_func(model_response)

        max_response_length = 3072

        length_reward = length_reward_func_v3(response_token_len, 0.50, max_response_length)


        # length_reward = length_reward_func_base(response_token_len, 3072)

        # reasoning_action_reward = reasoning_action_reward_func(model_response)

        answer_format_reward = answer_format_reward_func(model_response)


        # Correctnes (baseline)
        score = (soft_format_reward) * 0.5 + ground_truths_reward

        # Format-Only
        score = (soft_format_reward + answer_format_reward) * 0.5


        # Format-Length
        if answer_format_reward > 0:
            score = (soft_format_reward + answer_format_reward) * 0.5 + length_reward
        else:
            score = min(0.0, length_reward)


        data = {
        "model_response": model_response,
        "reward": {  
            "score" : score,    
            "ground_truths_contain_reward": ground_truths_contain_reward,
            "ground_truths_reward": ground_truths_reward,
            "ground_truths":ground_truths,
            "soft_format_reward": soft_format_reward,
            # "strict_format_reward": strict_format_reward
            "length_reward": length_reward,
            "response_token_len":response_token_len, 
            "response_len": len(model_response),
            "answer_format_reward":answer_format_reward,
            # "reasoning_action_reward":reasoning_action_reward
            },

        }
        json_output = json.dumps(data, indent=4, ensure_ascii=False)

        # Thread-safe dict for tracking printed data sources
        print_lock = threading.Lock()

        with print_lock:

            print(json_output)

        # RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        # RewardOutput(reward=self.config.correct_reward, is_correct=True)

        if score <= 0:
            return RewardOutput(reward=score, is_correct=False)

        else:
            return RewardOutput(reward=score, is_correct=True)
        

def xrh_math_reward_fn(solution_str, ground_truth, response_token_len, enable_llm = False):
    
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    
    reward_fn = RewardMathFn_xrh(reward_config)
    
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, response_token_len=response_token_len, ground_truth={"answer": ground_truth}))
    
    return reward_response.reward



if __name__ == "__main__":

    reward = RewardMathFn_xrh(RewardConfig)
    
    input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", response_token_len=2000, ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    
    input2 = RewardInput(problem="What is the value of $9^3 + 3(9^2) + 3(9) + 1$?", problem_type=RewardType.MATH, model_response="\n<think>\nThe given expression is the expansion of $(9+1)^3$.  In general, the cube of $(x+y)^3$ is \\[(x+y)^3=1x^3+3x^2y+3xy^2+1y^3.\\]   The first and last terms in the given expression are cubes and the middle two terms both have coefficient 3, giving us a clue that this is a cube of a binomial and can be written in the form \\[(x+y)^3\\]In this case, $x=9$ and $y=1$, so our answer is\\[(9+1)^3\\ = 10^3 = \\boxed{1000}\\]\n</think>\n<answer>\n \\boxed{ 1000 } \\ \n</answer>\n", response_token_len=4000, ground_truth={"answer": "1000"})

    # 格式对了，答案 正/负
    response3 = "To solve this problem, we need to find the number of distinct four-digit positive integers where the product of their digits equals 18. Let's break down the solution step by step.\n\n1. **Understanding the problem**:\n   - We are looking for four-digit numbers \\( \\overline{abcd} \\) (where \\(a\\), \\(b\\), \\(c\\), and \\(d\\) are digits and \\(a \\neq 0\\)) such that \\(a \\times b \\times c \\times d = 18\\).\n   - The digits \\(a, b, c, d\\) must be integers between 1 and 9 because they are part of a four-digit number and the digit 0 cannot be the leading digit.\n\n2. **Factorization of 18**:\n   - We need to find all combinations of four digits (where the first digit is from 1 to 9 and the others are from 0 to 9) whose product is 18.\n   - The prime factorization of 18 is \\(2 \\times 3^2\\). Therefore, the digits we are looking for must be a combination of the factors 1, 2, 3, 6, 9 that multiply to give 18.\n\n3. **Possible combinations**:\n   - The possible sets of digits (considering permutations) that multiply to 18 are: \\((1, 1, 2, 9)\\), \\((1, 1, 3, 6)\\), \\((1, 2, 3, 3)\\), and \\((2, 3, 3, 1)\\) (the last one is the same as the third one but in a different order, so it's not a new combination).\n   - We need to consider all permutations of these sets to form valid four-digit numbers.\n\n4. **Counting the permutations**:\n   - For the combination \\((1, 1, 2, 9)\\), the number of distinct permutations is \\(\\frac{4!}{2!} = 12\\) (since 1 is repeated twice).\n   - For the combination \\((1, 1, 3, 6)\\), the number of distinct permutations is also \\(\\frac{4!}{2!} = 12\\) (since 1 is repeated twice).\n   - For the combination \\((1, 2, 3, 3)\\), the number of distinct permutations is \\(\\frac{4!}{2!} = 12\\) (since 3 is repeated twice).\n\n5. **Summing up all the permutations**:\n   - We have 12 permutations for \\((1, 1, 2, 9)\\), 12 permutations for \\((1, 1, 3, 6)\\), and 12 permutations for \\((1, 2, 3, 3)\\).\n   - Therefore, the total number of distinct four-digit positive integers where the product of their digits equals 18 is \\(12 + 12 + 12 = 36\\).\n\nLet's verify this with Python code to ensure the accuracy of our result.\n```python\nfrom itertools import permutations\n\n# Function to get all distinct four-digit numbers with a given product of digits\ndef count_distinct_numbers(product):\n    count = 0\n    # Generate all possible combinations of four digits (1-9 for the first digit, 0-9 for the rest)\n    for a in range(1, 10):\n        for b in range(1, 10):\n            for c in range(1, 10):\n                for d in range(1, 10):\n                    if a * b * c * d == product:\n                        # Create the number and count distinct permutations\n                        number = (a, b, c, d)\n                        # Generate all permutations and count distinct ones\n                        perm_set = set(permutations(number))\n                        count += len(perm_set)\n    return count\n\n# Calculate the number of distinct four-digit numbers with the product of digits equal to 18\nresult = count_distinct_numbers(18)\nprint(result)\n```\n```output\n36\n```\nThe number of distinct four-digit positive integers where the product of their digits equals 18 is \\(\\boxed{36}\\).<|endoftext|>"
    ground_truth3 = "1000"
    # ground_truth3 = "36"
    input3 = RewardInput(problem="What is the value of $9^3 + 3(9^2) + 3(9) + 1$?", problem_type=RewardType.MATH, model_response=response3, response_token_len=4000, ground_truth={"answer": ground_truth3})

    # 格式完全错
    response4 = "To solve this problem,  \\boxed their digits equals 18 is \\(\\boxed{36}\\).<|endoftext|>"
    ground_truth4 = "1000"
    input4 = RewardInput(problem="What is the value of $9^3 + 3(9^2) + 3(9) + 1$?", problem_type=RewardType.MATH, model_response=response4, response_token_len=4000, ground_truth={"answer": ground_truth4})

    response4_1 = "Let's solve the problem step by step.\n\n### Step 1: Determine the parity of the function \\( f(x) \\)\nThe function \\( f(x) \\) satisfies the functional equation \\( f(x+y) = f(x) + f(y) \\) for any \\( x, y \\in \\mathbb{R} \\). This is known as the Cauchy functional equation. One of the properties of the solutions to this equation is that they are linear when the domain is restricted to rational numbers, i.e., \\( f(x) = cx \\) for some constant \\( c \\). However, we need to verify if \\( f(x) \\) is odd or even.\n\n#### Step 1.1: Check the parity\nTo determine the parity of \\( f(x) \\), we need to check if \\( f(-x) = f(x) \\) (even function) or \\( f(-x) = -f(x) \\) (odd function). Let's substitute \\( y = -x \\) into the functional equation:\n\\[ f(x + (-x)) = f(x) + f(-x) \\]\nSince \\( x + (-x) = 0 \\), we have:\n\\[ f(0) = f(x) + f(-x) \\]\nWe need to find \\( f(0) \\). Let's substitute \\( x = 0 \\) and \\( y = 0 \\) into the functional equation:\n\\[ f(0 + 0) = f(0) + f(0) \\]\n\\[ f(0) = 2f(0) \\]\nThis implies:\n\\[ f(0) = 0 \\]\nSo, we have:\n\\[ 0 = f(x) + f(-x) \\]\nThis implies:\n\\[ f(-x) = -f(x) \\]\nTherefore, \\( f(x) \\) is an odd function.\n\n### Step 2: Determine if \\( f(x) \\) has an extreme value on the interval \\([-3, 3]\\)\nGiven that \\( f(1) = -2 \\), we can determine the function \\( f(x) \\) for any rational number \\( x \\). Since \\( f(x) = cx \\), we can find \\( c \\) using \\( f(1) = -2 \\):\n\\[ f(1) = c \\cdot 1 = -2 \\]\nSo, \\( c = -2 \\). Therefore, \\( f(x) = -2x \\).\n\nWe need to check if \\( f(x) = -2x \\) has an extreme value on the interval \\([-3, 3]\\). Since \\( f(x) \\) is a linear function, it does not have any local extreme values (maximum or minimum). However, we need to check the values at the endpoints of the interval \\([-3, 3]\\).\n\n#### Step 2.1: Evaluate \\( f(x) \\) at the endpoints\n\\[ f(-3) = -2 \\cdot (-3) = 6 \\]\n\\[ f(3) = -2 \\cdot 3 = -6 \\]\nThe function \\( f(x) = -2x \\) is a decreasing function on the interval \\([-3, 3]\\), so the maximum value is \\( 6 \\) at \\( x = -3 \\) and the minimum value is \\( -6 \\) at \\( x = 3 \\).\n\n### Conclusion\n1. The function \\( f(x) \\) is an odd function.\n2. The function \\( f(x) = -2x \\) has an extreme value on the interval \\([-3, 3]\\). The maximum value is \\( 6 \\) at \\( x = -3 \\) and the minimum value is \\( -6 \\) at \\( x = 3 \\).\n\nLet's write the final answer in the boxed format:\n\n1. The function \\( f(x) \\) is an odd function.\n2. The function \\( f(x) \\) has an extreme value on the interval \\([-3, 3]\\). The maximum value is \\(\\boxed{6}\\) at \\( x = -3 \\) and the minimum value is \\(\\boxed{-6}\\) at \\( x = 3 \\).<|endoftext|>"
    ground_truths4_1 = "-6"
    input4_1 = RewardInput(problem="What is the value of $9^3 + 3(9^2) + 3(9) + 1$?", problem_type=RewardType.MATH, model_response=response4, response_token_len=4000, ground_truth={"answer": ground_truth4})

    # 没说完
    response5 = "To solve this problem,  their digits equals 18 is \\(36\\)."
    ground_truth5 = "1000"
    input5 = RewardInput(problem="What is the value of $9^3 + 3(9^2) + 3(9) + 1$?", problem_type=RewardType.MATH, model_response=response5, response_token_len=4000, ground_truth={"answer": ground_truth5})


    output = reward(input4_1)

    print(output)

    response = "^user\n ^user\n ^user\n ^user\n ^user\n"

    ground_truths = "2"

    # print(ground_truths_contain_reward_func(response, ground_truths))


