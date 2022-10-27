#!
from abc import ABCMeta, abstractmethod
import re
import happybase
import pandas as pd
import traceback
import pdb
import time
from functools import wraps
from datetime import datetime
import json
import numpy as np

p = pdb.set_trace

'''
json.dumps(arr, cls=NpEncoder)
'''

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def count_time(func):
    """
    打印函数运行起始时间、终止时间、运行时间的装饰器
    """
    @wraps(func)
    def printFuncRunTime(*args, **kwargs):
        """
        打印函数运行时间
        """
        time1 = datetime.now()
        #print(time1, f'{func.__name__} start running.')
        res = func(*args, **kwargs)
        time2 = datetime.now()
        #print(time2, f'{func.__name__} done.')
        print(f'{func.__name__} total use time:', time2 - time1)
        return res
    return printFuncRunTime
