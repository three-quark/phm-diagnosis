# -*- coding: UTF-8 -*-
import re
import happybase
import pandas as pd
from .base import DBInterface


def time_filter(startTime, endTime, time_column):
    build_timeFilter = f"SingleColumnValueFilter('data','{time_column}',>=,'binary:" + startTime + \
        f"') AND SingleColumnValueFilter('data','{time_column}',<=,'binary:" + \
        endTime + "')"
    return build_timeFilter


class HbaseClient(DBInterface):

    def __init__(self, size, host, port):
        self.pool = happybase.ConnectionPool(size=size, host=host, port=port)

    def read(self, table, columns, startTime, endTime, column_family, time_column):
        '''read data from hbase, with time columns to filter it
        '''
        paramNameList = ["row_key", time_column] + columns
        #print(paramNameList)
        hbaseReadParamList = paramNameList[1:]  # remove key list
        #print(hbaseReadParamList)
        choose_columns = []
        hbaseReadParamList.sort()
        for param in hbaseReadParamList:
            choose_columns.append(column_family + ":" + param)


        paramData = []
        with self.pool.connection() as hbase_connection:
            read_table = hbase_connection.table(table)
            timeFilter = time_filter(startTime, endTime, time_column)
            table_data = read_table.scan(
                columns=choose_columns, filter=timeFilter)
            # table_data = read_table.scan(columns=choose_columns)

            for dataKey, dataDict in table_data:
                datetime_row = dataDict[(
                    column_family + ":" + time_column).encode("utf-8")].decode("utf-8")
                # time format 21/7/29 9:42:44
                m = len(re.findall(
                    r"\A[1-9]\d/\d+/\d+\s+\d+:\d+:\d+", datetime_row))
                dictKeyList = dataDict.keys()
                ''' fill the null cell with 'DUMMY' instead
                '''
                if len(dictKeyList) != len(choose_columns):
                    for col in choose_columns:
                        if col.encode("utf-8") not in dictKeyList:
                            dataDict[col.encode(
                                "utf-8")] = "DUMMY".encode("utf-8")
                dataDict = {k: v for k, v in sorted(dataDict.items())}
                paramList = list(dataDict.values())
                paramData.append([x.decode('utf-8') for x in paramList])

            hbase_connection.close()
        dfParamList = hbaseReadParamList.copy()
        df = pd.DataFrame(data=paramData, columns=dfParamList)
        df.fillna(0.0)
        return df
