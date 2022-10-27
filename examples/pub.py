#!
import numpy as np
import pdb
import json
from phm_diagnosis.database import HBaseClient
from phm_diagnosis.database import RedisClient
from phm_diagnosis.database.utils import NpEncoder
import time
from config import read_hbase_table
from config import read_columnFamilyName
from config import time_columnName
from config import paramNameList
from config import startTime
from config import endTime
from config import hb_client_paras

from config import rds_ip
from config import rds_port

import datetime

from phm_diagnosis.dummydata.gen_wave import noise 
from phm_diagnosis.dummydata.gen_wave import sin_wave
from phm_diagnosis.dummydata.gen_wave import cos_wave

import pandas as pd

@noise('normal', 0.0, 1.0)
def my_sin_wave(a=1.0, f=40, p=0.0, l=10, s=120):
    return sin_wave(a,f,p,l,s)

def get_time_before_current(sec):
    _now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _bef = (
        datetime.datetime.now() -
        datetime.timedelta(
            days=0,
            seconds=sec)).strftime("%Y-%m-%d %H:%M:%S")
    return _bef, _now

def fetch_data(hb_conn, rd_conn, startTime, endTime, label):
    print(startTime, endTime)
    dt = hb_conn.read(
        read_hbase_table,
        read_columnFamilyName,
        time_columnName,
        paramNameList,
        startTime,
        endTime
    )
    # pdb.set_trace()
    status = rd_conn.publish(label, dt.to_json())
    #status = rd_conn.publish(label, json.dumps(dt.to_json(), cls=NpEncoder))
    return status

def generate_data(rd_conn, label):
    print(label)
    columns = ["data:ch%d"%i for i in range(16)]
    datas = [my_sin_wave(a=1.0, f=40, p=0.0, l=10, s=1200)[1] for i in range(16)]
    ''' make the data contains causal to test my causaldiscovery '''
    datas = pd.DataFrame(np.array(datas).T, columns=columns)
    datas.iloc[:,-1] = datas.iloc[:,0] + datas.iloc[:,1]
    datas.iloc[:,-2] = datas.iloc[:,2] * datas.iloc[:,3]
    datas.iloc[:,-3] = datas.iloc[:,4] - datas.iloc[:,5]
    datas.iloc[:,-4] = datas.iloc[:,-1] + datas.iloc[:,-2] + datas.iloc[:,-3]
    status = rd_conn.publish(label, datas.to_json())
    #status = rd_conn.publish(label, json.dumps(dt.to_json(), cls=NpEncoder))
    return status

def run():
    ''' fetch hbase data every 10 seconds '''
    hb_conn = HBaseClient(**hb_client_paras)
    rd_conn = RedisClient(ip=rds_ip, port=rds_port)
    while (True):
        hour_ago, cur_1 = get_time_before_current(60 * 60)
        half_day_ago, cur_2 = get_time_before_current(60 * 60 * 12)
        day_ago, cur_3 = get_time_before_current(60 * 60 * 24)
        week_ago, cur_4 = get_time_before_current(60 * 60 * 24 * 7)
        month_ago, cur_5 = get_time_before_current(60 * 60 * 24 * 7 * 4)
        half_year_ago, cur_6 = get_time_before_current(
            60 * 60 * 24 * 7 * 4 * 26)

        fetch_data(hb_conn, rd_conn, hour_ago, cur_1, 'hour')
        fetch_data(hb_conn, rd_conn, half_day_ago, cur_2, 'halfday')
        fetch_data(hb_conn, rd_conn, day_ago, cur_3, 'day')
        fetch_data(hb_conn, rd_conn, week_ago, cur_4, 'week')
        fetch_data(hb_conn, rd_conn, month_ago, cur_5, 'month')
        fetch_data(hb_conn, rd_conn, half_year_ago, cur_6, 'halfyear')

        time.sleep(10)

def run_offline():
    ''' fetch hbase data every 10 seconds '''
    rd_conn = RedisClient(ip=rds_ip, port=rds_port)
    while (True):
        generate_data(rd_conn, 'hour')
        generate_data(rd_conn, 'halfday')
        generate_data(rd_conn, 'day')
        generate_data(rd_conn, 'week')
        generate_data(rd_conn, 'month')
        generate_data(rd_conn, 'halfyear')

        time.sleep(1)

if __name__ == "__main__":
    #run()
    run_offline()
