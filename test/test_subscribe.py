import unittest
import time
from phm_dataio import HBaseClient
from phm_dataio import RedisClient
import pdb
import traceback
from phm_dataio.utils import count_time

class TestSetUp(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @count_time
    def test_sub(self, cli, name): 
        try:
            msg = cli.subscribe(name).parse_response(block=False, timeout=60)
        except redis.exceptions.ConnectionError as e:
            traceback.print_exc()

    def test_redis_client(self):
        rdcli = RedisClient(ip='192.168.0.4', port=6379)
        cnt=0
        while(True):
            msg = self.test_sub(rdcli, 'date')
            msg = self.test_sub(rdcli, 'month')
            msg = self.test_sub(rdcli, 'year')
            msg = self.test_sub(rdcli, 'time')
            cnt+=1
            print(cnt)

if __name__ == "__main__":
    unittest.main()
else:
    unittest.main()

