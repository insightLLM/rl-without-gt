
import re

# Source:
# https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
# def zipngram(text: str, ngram_size: int):
#     words = text.lower().split()
#     return zip(*[words[i:] for i in range(ngram_size)])
def zipngram(text: str, ngram_size: int):
    words = text.lower().split()
    for i in range(len(words) - ngram_size + 1):
        yield tuple(words[i:i + ngram_size])



def build_suffix_array(s):
    n = len(s)
    suffixes = [(s[i:], i) for i in range(n)]
    suffixes.sort()
    suffix_array = [suffix[1] for suffix in suffixes]
    return suffix_array


def build_lcp_array(s, suffix_array):
    n = len(s)
    rank = [0] * n
    lcp = [0] * (n - 1)

    # Create the rank array based on suffix array
    for i, suffix_idx in enumerate(suffix_array):
        rank[suffix_idx] = i

    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = suffix_array[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i] - 1] = h
            if h > 0:
                h -= 1
    return lcp



def find_longest_repeated_substring(s, rep_length_threshold=100):
    """
    找出最长的重复子串

    """

    n = len(s)
    suffix_array = build_suffix_array(s)
    lcp_array = build_lcp_array(s, suffix_array)

    max_len = 0
    max_substr = ""
    max_substr_cnt = 0

    # 寻找LCP数组中的最大值
    max_len = max(lcp_array)

    # 返回最大LCP对应的子串
    i = lcp_array.index(max_len)

    max_substr = s[suffix_array[i]:suffix_array[i] + lcp_array[i]]

    max_substr_cnt = s.count(max_substr) # 出现的次数，统计的时候不会记录重叠

    if max_len >= rep_length_threshold:

        return max_substr, max_substr_cnt

    else:
        return '', 0




