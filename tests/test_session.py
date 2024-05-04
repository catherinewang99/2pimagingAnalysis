# Test that session.py works on your machine. 
from alm_2p import session
import numpy as np

test_data_path = "/Volumes/Seagate_Storage/alm_learning_data/CW32/2023_10_05/"

class Test_Session():
    """Test the session object used to aggregate data over layers. 

    """
    def test_init_all_layers(self):
        """Just checking that everything runs as expected without errors. 

        """
        # With default parameters. 
        session.Session(test_data_path)
        session.Session(test_data_path,use_reg=True)
        # With one layer, using registration 
        session.Session(test_data_path,use_reg=True,triple=True)
        # With one layer, using registration 
        session.Session(test_data_path,use_reg=True,filter_reg = False)

    def test_init_single_layers(self):
        """Just checking that everything runs as expected without errors. 

        """
        # With one layer only. 
        [session.Session(test_data_path,layer_num = i) for i in range(1,6)]
        # With one layer, using registration 
        [session.Session(test_data_path,layer_num = i,use_reg=True) for i in range(1,6)]
        # With one layer, using registration 
        [session.Session(test_data_path,layer_num = i,use_reg=True,triple=True) for i in range(1,6)]
        # With one layer, using registration 
        [session.Session(test_data_path,layer_num = i,use_reg=True,filter_reg = False) for i in range(1,6)]

    def test_normalize_all_by_baseline(self):
        """Check that your new preprocessing works as expected.

        """
        sess_default = session.Session(test_data_path)
        sess_dff = session.Session(test_data_path,baseline_normalization="dff_avg")
        assert all([np.all(a==b) for a,b in zip(sess_dff.dff[0],sess_default.dff[0])]), "default behavior gives what we expected previously"
        sess_med = session.Session(test_data_path,baseline_normalization="median_zscore")
        assert type(sess_med) == type(sess_dff)
        for i in range(sess_med.dff.shape[-1]):
            assert type(sess_med.dff[0,i]) == type(sess_dff.dff[0,i])
            assert np.shape(sess_med.dff[0,i]) == np.shape(sess_dff.dff[0,i])
        assert any([np.all(a!=b) for a,b in zip(sess_dff.dff[0],sess_med.dff[0])]), "different baseline"


