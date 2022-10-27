#!
import pdb
import pandas as pd
import json
from phm_diagnosis.database import HBaseClient
from phm_diagnosis.database import RedisClient
from phm_diagnosis.database.utils import NpEncoder
from phm_diagnosis.causal import causal_discovery
from phm_diagnosis.causal import plot_causal_graph
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

now_str = lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    status = rd_conn.publish(label, json.dumps(dt.to_json(), cls=NpEncoder))
    return status


def listen(cli, name):
    try:
        msg = cli.subscribe(name).parse_response(block=False, timeout=3)
        return msg
    except redis.exceptions.ConnectionError as e:
        traceback.print_exc()


def handle(data, tag):
    assert isinstance(data, pd.DataFrame)
    t = now_str().split(" ")[0]
    tt = now_str().split(" ")[1]
    plot_causal_graph(causal_discovery(data, 2, 0.01), savefile='graph_{t}_{tag}_{tt}.png'.format(t=t, tag=tag, tt=tt))


def run():
    ''' fetch hbase data every 10 seconds '''
    rd_conn = RedisClient(ip=rds_ip, port=rds_port)
    labels = 'hour,halfday,day,week,month,halfyear'.split(',')

    while (True):
        for lb in labels:
            msg = listen(rd_conn, lb)
            if msg is None:
                print(lb, msg)
            elif msg[2] == b"{}":
                print(lb, msg)
            else:
                assert msg[1].decode() == lb
                dt = pd.read_json(
                    msg[2]).iloc[:, :-1].applymap(lambda x: 0.00 if x == "DUMMY" or x == "" else float(x))
                handle(dt, msg[1].decode())


if __name__ == "__main__":
    run()
