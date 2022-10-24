import pandas as pd
import os

def load_ssd():

    failure_df = pd.read_csv('../data/大型SSD故障预测数据集/ssd_failure_label.csv')
    pths = os.walk('../data/大型SSD故障预测数据集/')
    datas_csv = next(pths)[2]
    failure_times = list(set(failure_df.failure_time.apply(lambda x:''.join(x.split(' ')[0].split('-'))).values))

    dfs = []
    
    for t in failure_times:
        df = pd.read_csv('../data/大型SSD故障预测数据集/%s.csv'%t)
        dfs.append(df)
        if len(dfs)>0:
            return dfs

dfs = load_ssd()
