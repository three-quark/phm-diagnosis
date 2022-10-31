import unittest
import time
from phm_dataio import HBaseClient

class TestSetUp(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_hbase_client(self):
        read_hbase_table = "data_fan"
        paramNameList = ['ch%d'%i for i in range(17)]
        startTime = "2022-06-10 17:12:58"
        endTime = "2022-09-06 17:18:58"
        read_columnFamilyName="data"
        time_columnName="time"
        hb_conn = HBaseClient(1, 'hb_conn', 1, '192.168.0.4', 9000)
        a=3
        while(a>0):
            a=a-1
            result = hb_conn.read(
                read_hbase_table,
                read_columnFamilyName,
                time_columnName,
                paramNameList,
                startTime,
                endTime,
            )
            print(result)
            time.sleep(1)
            break

if __name__ == "__main__":
    unittest.main()
else:
    unittest.main()

