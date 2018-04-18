import os,sys
import h5py as h5
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if len(sys.argv) < 2:
    print 'usage: python train_xgb_features.py <output_label>'

label = sys.argv[1]

input_filename = '/home/olivito/datasci/lcd/data/EleEscan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = '/home/olivito/datasci/lcd/data/GammaEscan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = '/home/olivito/datasci/lcd/data/Pi0Escan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = '/home/olivito/datasci/lcd/data/ChPiEscan/merged_featuresonly/merged_minfeatures.h5'

def load_hdf5(filename):
    with h5.File(filename, 'r') as f:
        ECAL_E = f['ECAL_E'][:].reshape(-1,1)
        HCAL_E = f['HCAL_E'][:].reshape(-1,1)
        features = np.concatenate([ECAL_E, HCAL_E], axis=1)
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

linreg = LinearRegression()
linreg.fit(X_train,y_train)

print linreg.coef_

y_pred = linreg.predict(X_test)

output_dir = './Output/%s/'%(label)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = output_dir + 'results.h5'
outfile = h5.File(output_filename,'w')

outfile.create_dataset('regressor_pred',data=np.array(y_pred))
outfile.create_dataset('regressor_true',data=np.array(y_test))

outfile.close()
