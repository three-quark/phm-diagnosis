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

p = pdb.set_trace

'''
formular the code with code, but not with doc
interface of db client, puporse for make the code formuler, but it's hard to define the function to match all time and status
'''

class DBInterface(metaclass=ABCMeta):

    '''
    @abstractmethod
    def connect(self, **keargs):
        raise NotImplementedError

    @abstractmethod
    def exec(self, **keargs):
        raise NotImplementedError

    @abstractmethod
    def disconnect(self, **keargs):
        raise NotImplementedError
    '''

    @abstractmethod
    def read(self, **keargs):
        raise NotImplementedError


