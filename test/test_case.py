'''
测试
'''
import sys
import os
sys.path.append(os.curdir)
from phm_diagnosis import why_builder
from phm_diagnosis import get_cause_graph_from_why
from phm_diagnosis import causal_model_build
from phm_diagnosis import plot_causal_graph
from phm_diagnosis import causal_discovery
from ylearn.exp_dataset.gen import gen
from ylearn import Why
import pdb

import pandas as pd

import numpy as np
import unittest

p = pdb.set_trace


# 计时工具
def timer(func):
    def func_wrapper(*args,**kwargs):
        from time import time
        time_start = time()
        result = func(*args,**kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('\n{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result
    return func_wrapper


class TestSetUp(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_why_builder(self):
        df = pd.read_csv('data/BankChurners.csv.zip')
        assert isinstance(df, pd.DataFrame)
        cols = ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 'Income_Category', 
                'Months_on_book', 'Card_Category', 'Credit_Limit',  
                'Total_Trans_Amt' 
             ]
        data = df[cols]
        outcome = 'Total_Trans_Amt'
        why = why_builder(data,outcome,treatment='Income_Category')
        assert isinstance(why,Why)

        graph = get_cause_graph_from_why(why)
        print('\n nodes {0} s\n'.format(str(graph.causation)))
        print(graph.dag.nodes)
        print(graph.dag.edges)

        causal_model_build(graph=graph, treatment='Income_Category', outcome=outcome)
        plot_causal_graph(graph)
        df = pd.read_csv('data/BankChurners.csv.zip')
        cols = ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 'Income_Category', 
                'Months_on_book', 'Card_Category', 'Credit_Limit',  
                'Total_Trans_Amt' 
             ]
        cols = ['Customer_Age', 'Total_Trans_Amt', 'Credit_Limit','Dependent_count','Months_on_book']
        
        data = df[cols]
        
        data = data.astype(np.float16)
        data.dropna()
        def mean_norm(df_input):
            return df_input.apply(lambda x: ((x.max()-x) / (x.max()-x.min())), axis=0)
        X = mean_norm(data.iloc[:,:])

        plot_causal_graph(causal_discovery(X, 2, 0.01))

if __name__ == "__main__":
    unittest.main()
else:
    unittest.main()

