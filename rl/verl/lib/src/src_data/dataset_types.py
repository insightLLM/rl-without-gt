

import enum
from typing import Union


class TrainDataset(enum.Enum):
    """Enum for training datasets.
    
    Contains identifiers for various math problem datasets used during training.
    """
    AIME = 'AIME'  # American Invitational Mathematics Examination
    AMC = 'AMC'    # American Mathematics Competition
    OMNI_MATH = 'OMNI_MATH'  # Omni Math
    NUMINA_OLYMPIAD = 'OLYMPIAD'  # Unique Olympiad problems from NUMINA
    MATH = 'MATH'  # Dan Hendrycks Math Problems
    MATH_8K = 'MATH_8K' # 难度在3-5分的 math 数据集

    STILL = 'STILL'  # STILL dataset
    DEEPSCALER = 'DEEPSCALER'  # DeepScaler (AIME, AMC, OMNI_MATH, MATH, STILL)


class TestDataset(enum.Enum):
    """Enum for testing/evaluation datasets.
    
    Contains identifiers for datasets used to evaluate model performance.
    """
    AIME = 'AIME'  # American Invitational Mathematics Examination
    AMC = 'AMC'    # American Mathematics Competition  
    MATH = 'MATH'  # Math 500 problems
    MINERVA = 'MINERVA'  # Minerva dataset
    OLYMPIAD_BENCH = 'OLYMPIAD_BENCH'  # Olympiad benchmark problems

"""Type alias for either training or testing dataset types."""
Dataset = Union[TrainDataset, TestDataset]
