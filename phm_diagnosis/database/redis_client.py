#!
from abc import ABCMeta, abstractmethod
import re
import happybase
import pandas as pd
import traceback
import pdb
import time
import redis
from datetime import datetime
from .base import DBInterface
from .utils import count_time

'''
the redis client
'''
 
class RedisClient(DBInterface):

    def __init__(self, ip='127.0.0.1', port=6379):
        self.connection_pool = redis.ConnectionPool(host=ip, port=port)
        self._conn = redis.Redis(connection_pool=self.connection_pool)
 
    def publish(self, pub, msg):
        return self._conn.publish(pub, msg)
 
    def subscribe(self, sub):
        pub = self._conn.pubsub()
        pub.subscribe(sub)
        pub.parse_response()
        return pub

    @count_time
    def read(self, sub, cnt):
        response = self.subscribe(sub)
        return response

