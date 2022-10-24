'''
使用ylearn api接口，封装的故障诊断工具包
用于故障树生成，因果识别，因果估计，决策验证，工具量发现，因果解释等功能
'''

import os
import sys
import pandas as pd
from ylearn import Why
import logging
import pdb
import networkx as nx
import matplotlib.pyplot as plt

from ylearn.causal_model.model import CausalModel
from ylearn.causal_model.graph import CausalGraph
from ylearn.causal_discovery import CausalDiscovery

#__version__ = '0.0.5'
__license__ = 'Apache License 2.0'
__name__ = 'phm_diagnosis.api'

def _why_builder(train_data, outcome, *, treatment=None, **kwargs):
    '''
    使用配置参数与训练数据，拟合出故障诊断实体'why',why实例是各功能的API集成
    '''
    assert isinstance(train_data,pd.DataFrame)
    why=Why(**kwargs)
    why.fit(train_data,outcome,treatment=treatment)
    return why

def _get_cause_graph_from_why(why, **kwargs):
    '''
    why=>causal_graph
    从why中读取因果结构causal_graph
    '''
    assert isinstance(why, Why)
    return why.causal_graph()

def _causal_model_build(graph, treatment, outcome, **kwargs):
    '''
    causal_graph==>causalmodel
    使用因果结构，构建分析模型
    '''
    assert isinstance(graph, CausalGraph, **kwargs)
    model = CausalModel(graph, treatment='Card_Category', outcome='Total_Trans_Amt').outcome
    return model

def _get_causal_effect(why, control, return_detail, **kwargs):
    '''
    从why读取评估结果
    '''
    effect=why.causal_effect(control,return_detail, **kwargs)
    return effect

def _plot_causal_graph(graph, savefile='Graph.png', **kwargs):
    '''
    打印causal_graph因果结构图,并保存在本地
    '''
    assert savefile[-3:].lower()=='png'
    ng = nx.DiGraph(graph.causation).reverse()
    options = dict(with_labels=True, node_size=1000, **kwargs)

    nx.draw(ng, **options)
    plt.savefig(savefile, format="PNG")
    plt.show(block=False)

    plt.close()

def _causal_discovery(X, hidden_layer_dim, threshold):
    '''
    从张量数据中发现因果结构
    '''
    cd = CausalDiscovery(hidden_layer_dim=[hidden_layer_dim])
    est = cd(X, threshold=threshold, return_dict=True)
    return CausalGraph(CausalModel(est).causal_graph)

