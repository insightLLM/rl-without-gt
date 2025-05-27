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
# from . import gsm8k, math, prime_math, prime_code

import json  
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import os
import re
import csv

from cyac import Trie

################################################################################################

# stop_words = ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"]

# umls_nonsense = ['syndrome', 'syndromes', 'disease', 'diseases', 'disorder', 'disorders', 'infection', 'infections']

# trie = Trie()

# UMLS_DIR = '/global_data/pretrain/zsc_backup/UMLS/'
# MRCONSO_PATH = os.path.join(UMLS_DIR, 'MRCONSO.RRF')
# MRSTY_PATH = os.path.join(UMLS_DIR, 'MRSTY.RRF')
# MRREL_PATH = os.path.join(UMLS_DIR, 'MRREL.RRF')

# seps = set([ord(c) for c in ' |[]【】(){},.\"\'\\?:;%#`'])
# seps.add(ord('\n'))

# RELEVANT_SEMANTIC_TYPES = {
#     # 'T004', # Fungus
#     # 'T005', # Virus
#     # 'T007', # Bacterium
#     # 'T019', # Congenital Abnormality
#     # 'T020', # Acquired Abnormality
#     # 'T037', # Injury or Poisoning
#     # 'T046', # Pathologic Function
#     'T047', # Disease or Syndrome
#     # 'T048', # Mental or Behavioral Dysfunction
#     # 'T049', # Cell or Molecular Dysfunction
#     # 'T190', # Anatomical Abnormality
#     # 'T191', # Neoplastic Process
#     # 'T194', # Archaeon
#     }

# def load_mrconso(trie, cui_to_semantic_types):
#     """Load MRCONSO.RRF to map terms to CUIs."""
#     term_to_cuis = {}
#     cui_to_terms = {}
#     with open(MRCONSO_PATH, 'r', encoding='utf-8') as file:
#         reader = csv.reader(file, delimiter='|')
#         for row in reader:
#             cui = row[0]
#             term = row[14].lower()
#             # if term not in term_to_cuis:
#             #     term_to_cuis[term] = set()
#             # term_to_cuis[term].add(cui)
#             # if cui not in cui_to_terms:
#             #     cui_to_terms[cui] = set()
#             # cui_to_terms[cui].add(term)

#             # trie.insert(term)

#             if cui in cui_to_semantic_types and cui_to_semantic_types[cui] & RELEVANT_SEMANTIC_TYPES:
#                 if term not in term_to_cuis:
#                     term_to_cuis[term] = set()
#                 term_to_cuis[term].add(cui)
#                 if cui not in cui_to_terms:
#                     cui_to_terms[cui] = set()
#                 cui_to_terms[cui].add(term)
                
#                 trie.insert(term)
            
#     return term_to_cuis, cui_to_terms, trie

# def load_mrsty():
#     """Load MRSTY.RRF to map CUIs to their semantic types."""
#     cui_to_semantic_types = {}
#     with open(MRSTY_PATH, 'r', encoding='utf-8') as file:
#         reader = csv.reader(file, delimiter='|')
#         for row in reader:
#             cui = row[0]
#             tui = row[1]
#             if cui not in cui_to_semantic_types:
#                 cui_to_semantic_types[cui] = set()
#             cui_to_semantic_types[cui].add(tui)
#     return cui_to_semantic_types

# def load_mrrel():
#     """Load MRREL.RRF to map CUIs to their parent CUIs (hypernyms)."""
#     cui_to_parents = {}
#     with open(MRREL_PATH, 'r', encoding='utf-8') as file:
#         reader = csv.reader(file, delimiter='|')
#         for row in reader:
#             cui1 = row[0]
#             cui2 = row[4]
#             rel = row[3]
#             if rel == 'PAR':  # 'PAR' indicates a parent (hypernym) relationship
#                 if (cui1 in cui_to_semantic_types and cui_to_semantic_types[cui1] & RELEVANT_SEMANTIC_TYPES) and \
#                    (cui2 in cui_to_semantic_types and cui_to_semantic_types[cui2] & RELEVANT_SEMANTIC_TYPES):
#                     if cui1 not in cui_to_parents:
#                         cui_to_parents[cui1] = set()
#                     cui_to_parents[cui1].add(cui2)
#     return cui_to_parents

# def reward_function(llm_response, correct_diagnosis, term_to_cuis, cui_to_terms, cui_to_parents, trie):
#     llm_response = llm_response.lower()
#     correct_diagnosis = correct_diagnosis.lower()
    
#     # llm_cuis = term_to_cuis.get(llm_response, set())
#     correct_cuis = term_to_cuis.get(correct_diagnosis, set())

#     # if not llm_cuis or not correct_cuis:
#     #     return 0  # If either term is not in UMLS, assign a score of 0

#     pre_sets = trie.match_longest(llm_response.lower(), seps)
#     trie_matched = []
#     for id_, start, end in pre_sets:
#         word = trie[id_] 
#         # if word == head_ent.lower() or word in head_ent.lower() or word in filtered_words:
#         #     continue
#         trie_matched.append([word, start, end])
    
#     mentioned_terms = [word[0] for word in trie_matched]
    
#     # Check for synonymy (intersection of CUIs)
#     # if llm_cuis & correct_cuis:
#     #     return 10
#     for correct_cui in correct_cuis:
#         for correct_term in cui_to_terms[correct_cui]:
#             # if correct_term in llm_response:
#             if correct_term in mentioned_terms:
#                 return 10

#     # print(mentioned_terms)
#     # print(correct_diagnosis)
#     # # Check for hypernymy
#     for correct_cui in correct_cuis:
#         if correct_cui in cui_to_parents:
#             for correct_parent in cui_to_parents[correct_cui]:
#                 for correct_term in cui_to_terms[correct_parent]:
#                     # if correct_term in llm_response:
#                     if correct_term not in umls_nonsense and correct_term not in stop_words:
#                         if correct_term in mentioned_terms:
#                             # print(correct_term)
#                             return 5
    
#     return 0  # Neither synonym nor hypernym


# cui_to_semantic_types = load_mrsty()
# term_to_cuis, cui_to_terms, trie = load_mrconso(trie, cui_to_semantic_types)
# cui_to_parents = load_mrrel()

################################################################################################

def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)


    elif data_source in ['generated_v2']:

        client = OpenAI(
            api_key='sk-CdSOwDI0CDxSle5HeWnSn6lpgNdoIgbFq1Q1qf7oBpDFHXp9', 
            base_url='http://oneapi-bcloud.bc-inner.com/v1'
        )

        # prompt = """
        # **Task:** Evaluate the model's diagnostic solution by comparing it to the ground truth diagnosis.

        # **Instructions:**

        # 1. **Exact Match or Synonym:** If the model's solution includes a diagnosis that is identical to or a recognized synonym of the ground truth diagnosis, assign a reward score of **10**.

        # 2. **Hypernym Match:** If the model's solution includes a diagnosis that is a hypernym (a more general term) of the ground truth diagnosis, assign a reward score of **5**.

        # 3. **No Match:** If the model's solution does not include the ground truth diagnosis or an appropriate synonym or hypernym, assign a reward score of **0**.

        # **Notes:**

        # - The model's solution may list multiple diagnoses. Evaluate each diagnosis independently based on the criteria above.

        # - Use medical knowledge to determine synonyms and hypernyms accurately.

        # **Input:**

        # - **Model's Solution:** {solution}

        # - **Ground Truth Diagnosis:** {answer}

        # **Output:**

        # - **Reward Score:** [0, 5, or 10]

        # You should only output an integer value for the reward score.

        # """
        prompt = """
        **Task:** Evaluate the model's diagnostic solution by comparing it to the ground truth diagnosis.

        **Instructions:**

        1. **Exact Match or Synonym:** If the model's solution includes a diagnosis that is identical to or a recognized synonym of the ground truth diagnosis, assign a reward score of **2**.

        2. **Hypernym Match:** If the model's solution includes a diagnosis that is a hypernym (a more general term) of the ground truth diagnosis, assign a reward score of **1**.

        3. **No Match:** If the model's solution does not include the ground truth diagnosis or an appropriate synonym or hypernym, assign a reward score of **0**.

        **Notes:**

        - The model's solution may list multiple diagnoses. Evaluate each diagnosis independently based on the criteria above.

        - Use medical knowledge to determine synonyms and hypernyms accurately.

        **Input:**

        - **Model's Solution:** {solution}

        - **Ground Truth Diagnosis:** {answer}

        **Output:**

        - **Reward Score:** [0, 1, or 2]

        You should only output an integer value for the reward score.

        """

        def call_api(solution, answer):

            # retry = 0
            # while retry <= 3:
            while True:
                try:
                    prompt_completed = prompt.format(solution=solution, answer=answer)
                    response_R = client.chat.completions.create(
                        model="deepseek-v3",
                        # model="qwq-32b",
                        messages=[{"role": "user", "content": prompt_completed},],
                        max_tokens=1
                    )
                    final_answer = response_R.choices[0].message.content
                    print(final_answer)
                    try:
                        final_reward = int(final_answer)
                        break
                    except:
                        continue
                    # break

                except Exception as e:
                    print(f"API 调用失败: {e}")
                    # return 1
                    # retry += 1
                    continue
            
            # if retry > 3:
            #     return 1

            # try:
            #     final_reward = int(final_answer)
            # except:
            #     print(final_answer)
            #     final_reward = 1

            return final_reward

        def compute_rd_score(solution_str, ground_truth) -> float:

            try:
                disease_answer = solution_str.split("</think>\n\n")[1]
            except:
                reward = 0.
                return reward

            # if ground_truth.lower() in disease_answer.lower():
            #     reward = 10.
            # else:
            #     reward = 1.
            
            reward = call_api(disease_answer, ground_truth)
            
            # reward = reward_function(disease_answer, ground_truth, term_to_cuis, cui_to_terms, cui_to_parents, trie)
            # print(reward)
            
            return reward
        ###
        res = compute_rd_score(solution_str, ground_truth)

    else:
        return NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
