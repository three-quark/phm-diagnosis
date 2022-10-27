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
from .utils import count_time
from .base import DBInterface

p = pdb.set_trace

'''
the hbase client 
'''

def timeFilter(startTime, endTime, time_columnName):
    return f"SingleColumnValueFilter('data','{time_columnName}',>=,'binary:" + startTime + \
        f"') AND SingleColumnValueFilter('data','{time_columnName}',<=,'binary:" + endTime + "')"

class HBaseClient(DBInterface):

    def __init__(self, _id, name, size, host, port=9000, **kwargs):
        self._id = _id
        self.name = name
        self.connect_pool(size, host, port, **kwargs)

    def connect_pool(self, numThread, host, port=9000, **kwargs):
        self.pool = happybase.ConnectionPool(
            size=numThread + 1, host=host, port=9090)

    #def connect(self):
    #    self.pool.connection()

    #def disconnect(self):
    #    self.pool.close()

    #def exec(self, **kwargs):
    #    pass

    def build_data_struct(self, read_columnFamilyName, time_columnName, paramNameList):
        # data struct [cols_cluster_name: cols_name, cols_cluster_name: cols_name]
        paramNameList = ["row_key", time_columnName] + paramNameList
        hbaseReadParamList = paramNameList[1:]
        hbaseReadParamList.sort()
        choose_columns = []
        for param in hbaseReadParamList:
            choose_columns.append(read_columnFamilyName + ":" + param)
        return choose_columns

    @count_time
    def read(
        self,
        read_hbase_table,
        read_columnFamilyName,
        time_columnName,
        paramNameList,
        startTime,
        endTime,
    ):
        with self.pool.connection() as hbase_connection:
            try:
                choose_columns = self.build_data_struct(read_columnFamilyName, time_columnName, paramNameList)
                # conn to table
                read_table = hbase_connection.table(read_hbase_table)
 
                # specify the target columns and filter data by time
                m_time_filter = timeFilter(startTime, endTime, time_columnName)
                #print(choose_columns)
                #print(m_time_filter)
                table_data = read_table.scan(
                    columns=choose_columns, filter=m_time_filter)
 
                paramData = []
                for dataKey, dataDict in table_data:
                    # 每一行的时间
                    datetime_row = dataDict[(
                        read_columnFamilyName + ":" + time_columnName).encode("utf-8")].decode("utf-8")
                    m = len(
                        re.findall(
                            r"\A[1-9]\d/\d+/\d+\s+\d+:\d+:\d+",
                            datetime_row))
 
                    dictKeyList = dataDict.keys()
                    if len(dictKeyList) != len(choose_columns):
                        for content in choose_columns:
                            if content.encode("utf-8") not in dictKeyList:
                                dataDict[content.encode("utf-8")] = "DUMMY".encode("utf-8")
                    dataDict = {k: v for k, v in sorted(dataDict.items())}
                    #print(dataDict)
                    paramList = list(dataDict.values())
                    #print(paramList)
                    paramData.append([x.decode('utf-8') for x in paramList])
                #print(paramData)
                if len(paramData) == 0:
                    df = self.arr2df(paramData, [])
                    hbase_connection.close()
                    return df

                df = self.arr2df(paramData, choose_columns)
                hbase_connection.close()
                return df
            except AssertionError as e:
                traceback.print_exc()

    def arr2df(self, paramData, dfParamList):
        df = pd.DataFrame(data=paramData, columns=dfParamList)
        #df.fillna(0)          # 将缺失值以0填充
        return df
