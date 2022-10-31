from phm_diagnosis.database.hbase_client import HbaseClient

m_hbase_client = HbaseClient(size=1, host='192.168.0.4', port=9090)
print(m_hbase_client)

table = "data_fan"
columns = ['ch0', 'ch1']
startTime = "2022-06-10 17:12:58"
endTime = "2022-09-06 17:18:58"
column_family = 'data'
time_column= 'time'
res = m_hbase_client.read(table, columns, startTime, endTime, column_family, time_column)
print(res)


