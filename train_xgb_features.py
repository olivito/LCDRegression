import os,sys
import h5py as h5
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

if len(sys.argv) < 2:
    print 'usage: python train_xgb_features.py <output_label>'

label = sys.argv[1]

#input_filename = '/home/olivito/datasci/lcd/data/EleEscan/merged_featuresonly/merged_minfeatures.h5'
input_filename = '/home/olivito/datasci/lcd/data/GammaEscan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = '/home/olivito/datasci/lcd/data/Pi0Escan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = '/home/olivito/datasci/lcd/data/ChPiEscan/merged_featuresonly/merged_minfeatures.h5'

def load_hdf5(filename):
    with h5.File(filename, 'r') as f:
        ECAL_E = f['ECAL_E'][:].reshape(-1,1)
        HCAL_E = f['HCAL_E'][:].reshape(-1,1)
        ECALmomentX2 = f['ECALmomentX2'][:].reshape(-1,1)
        ECALmomentY2 = f['ECALmomentY2'][:].reshape(-1,1)
        ECALmomentZ1 = f['ECALmomentZ1'][:].reshape(-1,1)
        ECALmomentXY2 = np.sqrt(np.square(ECALmomentX2) + np.square(ECALmomentY2))
        HCALmomentX2 = f['HCALmomentX2'][:].reshape(-1,1)
        HCALmomentY2 = f['HCALmomentY2'][:].reshape(-1,1)
        HCALmomentXY2 = np.sqrt(np.square(HCALmomentX2) + np.square(HCALmomentY2))
        HCALmomentZ1 = f['HCALmomentZ1'][:].reshape(-1,1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2, ECALmomentZ1, HCALmomentXY2, HCALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2, HCALmomentXY2], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentZ1, HCALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, HCALmomentXY2, HCALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2, ECALmomentY2, ECALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentXY2, ECALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2, ECALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2], axis=1)
        features = np.concatenate([ECAL_E, HCAL_E, ECALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E], axis=1)
        energy = f['energy'][:]
    return features.astype(np.float32), energy.astype(np.float32)

X, y = load_hdf5(input_filename)
print X.shape, y.shape

# for ChPi: remove events with reco energy < 0.3 * true energy
if 'ChPi' in input_filename:
    Xy = np.concatenate([X,y.reshape(-1,1)],axis=1)
    Xy_good = Xy[np.where((Xy[:,0] + Xy[:,1]) > 0.3 * Xy[:,-1])]
    X, y = Xy_good[:,:-1], Xy_good[:,-1]
    print X.shape, y.shape

X_train, X_test, y_train, y_test = \
    train_test_split(X,y,test_size=0.3)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {'max_depth': 3, 'objective': 'reg:linear','eval_metric':['rmse']}
#param = {'max_depth': 3, 'objective': 'reg:linear','eval_metric':['rmse'],'min_child_weight': 1,'eta':0.9}
evallist = [(dtrain, 'train'), (dtest, 'test')]

num_round = 1000
progress = {}

bst = xgb.train(param,dtrain,num_round,evallist,evals_result=progress,early_stopping_rounds=10)

y_pred = bst.predict(dtest)

output_dir = './Output/%s/'%(label)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = output_dir + 'results.h5'
outfile = h5.File(output_filename,'w')

outfile.create_dataset('regressor_loss_history_train',data=np.array(progress['train']['rmse']))
outfile.create_dataset('regressor_loss_history_test',data=np.array(progress['test']['rmse']))
outfile.create_dataset('regressor_pred',data=np.array(y_pred))
outfile.create_dataset('regressor_true',data=np.array(y_test))

outfile.close()
