
from phm_diagnosis.database.rabbitmq_client import RabbitmqClient
import yaml
import os
import numpy as np

global config
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), 'r', encoding='UTF-8') as stream:
    config = yaml.safe_load(stream)

resource = config["resource"]
log_path = resource["log_path"]

# %% hdfs信息
hdfsAddress = resource["hdfs"]["hdfsAddress"]

# %% hbase信息
hbase_master_host = resource["hbase"]["hbase_master_host"]
read_columnFamilyName = resource["hbase"]["read_columnFamilyName"]
time_columnName = resource["hbase"]["time_columnName"]

numThread = resource["thread"]["definedThreadNum"]  # 线程数
# partitionNum = rp["partition_number"]
# ruleFileSavePath = resource["ruleFile_savePath"]

# %% rabbit相关配置
rabbit = resource["rabbit"]
rabbitmqHost = rabbit["rabbitmqHost"]
rabbitmqPort = rabbit["rabbitmqPort"]
rabbitmqUserName = rabbit["rabbitmqUserName"]
rabbitmqPassword = rabbit["rabbitmqPassword"]
exchangeType = rabbit["exchangeType"]  # 交换机名称
queueName_verify_main = rabbit["queueName_verify_main"]  # 故障树验证主线程队列名称
exchangeName_verify_main = rabbit["exchangeName_verify_main"]  # 故障树验证主线程交换机名称
queueName_verify_thread = rabbit["queueName_verify_thread"]  # 故障树验证子线程队列名称
exchangeName_verify_thread = rabbit["exchangeName_verify_thread"]  # 故障树验证子线程交换机名称
queueName_release = rabbit["queueName_release"]  # 故障树发布队列名称
exchangeName_release = rabbit["exchangeName_release"]  # 故障树发布交换机名称

virtual_host = '/'
heartbeat = 0

m_rabbitmq_client = RabbitmqClient(rabbitmqHost, rabbitmqPort, rabbitmqUserName, rabbitmqPassword, virtual_host, heartbeat)

exchange_name = exchangeName_verify_main
queue_name = queueName_verify_main

print(exchange_name, exchangeType, queue_name)
m_rabbitmq_client.read(exchange_name, exchangeType, queue_name)
