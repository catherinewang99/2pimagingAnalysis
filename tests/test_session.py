# Test that session.py works on your machine. 
from alm_2p import session

test_data_path = "/Volumes/TOSHIBA EXT STO/alm_learning_data/CW32/2023_10_05/"

class test_Session():
    def test_init(self):
        session.Session(test_data_path)

