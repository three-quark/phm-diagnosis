import unittest
import time
from phm_dataio import HBaseClient
from phm_dataio import RedisClient
from phm_dataio.utils import NpEncoder
import json
import numpy as np
import pdb

class TestSetUp(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_redis_client(self):
        rdcli = RedisClient(ip='192.168.0.4', port=6379)
        cnt = 0
        while(True>0):
            dt = np.arange(1000000).reshape(-1,10,10).tolist()
            rdcli.publish('time', json.dumps(dt, cls=NpEncoder))
            rdcli.publish('date', json.dumps(dt, cls=NpEncoder))
            rdcli.publish('month', json.dumps(dt, cls=NpEncoder))
            rdcli.publish('year', json.dumps(dt, cls=NpEncoder))
            cnt+=1
            print(cnt)

if __name__ == "__main__":
    unittest.main()
else:
    unittest.main()

