import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

input_files = [

               # ('Output/Reg_EleFixed_Features_SumOnly_25epNoStop/results.h5', 'ECAL/HCAL Sums Only, NN Relu'),
               # ('Output/Reg_EleFixed_xgb_SumsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums Only, XGBoost'),
               # ('Output/Reg_EleFixed_Features_ECALMomsOnly_25epNoStop/results.h5', 'ECAL/HCAL Sums, ECAL Moments, NN Relu'),
               # ('Output/Reg_EleFixed_xgb_ECALMomsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECAL Moments, XGBoost'),

               #('Output/Reg_EleFixed_LinReg_SumsOnly/results.h5', 'ECAL/HCAL Sums Only, Linear Regression'),
               #('Output/Reg_EleFixed_xgb_SumsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums Only, XGBoost'),
               #('Output/Reg_EleFixed_xgb_ECALMomsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECAL Moments, XGBoost'),

              # ('Output/Reg_EleFixed_LinReg_ECALEOnly/results.h5', 'ECAL Sum Only, Linear Regression'),
              # ('Output/Reg_EleFixed_LinReg_SumsOnly/results.h5', 'ECAL/HCAL Sums Only, Linear Regression'),
              # ('Output/Reg_EleFixed_xgb_ECALEOnly_depth3_1000rounds/results.h5', 'ECAL Sum Only, XGBoost'),
              # ('Output/Reg_EleFixed_xgb_SumsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums Only, XGBoost'),

              # ('Output/Reg_EleFixed_xgb_SumsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums Only, XGBoost'),
              # ('Output/Reg_EleFixed_xgb_ECALX2Only_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECALmomentX2, XGBoost'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECALmomentZ1, XGBoost'),
              # ('Output/Reg_EleFixed_xgb_ECALMomsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECAL Moments, XGBoost'),
              # ('Output/Reg_EleFixed_xgb_AllMoms_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECAL/HCAL Moments, XGBoost'),

              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth2_1000rounds/results.h5', 'maxdepth 2'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'maxdepth 3'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_eta0p9_1000rounds/results.h5', 'maxdepth 3, eta 0.9'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_minchildwt5_1000rounds/results.h5', 'maxdepth 3, minchildweight 5'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth4_1000rounds/results.h5', 'maxdepth 4'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth4_minchildwt5_1000rounds/results.h5', 'maxdepth 4, minchildweight 5'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth5_minchildwt5_1000rounds/results.h5', 'maxdepth 5, minchildweight 5'),

               ('Output/Reg_GammaFixed_LinReg_SumsOnly/results.h5', 'ECAL/HCAL Sums Only, Linear Regression'),
               ('Output/Reg_GammaFixed_xgb_SumsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums Only, XGBoost'),
               ('Output/Reg_GammaFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECALMomentZ1, XGBoost'),
               ('Output/Reg_GammaFixed_xgb_ECALMomsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECAL Moments, XGBoost'),

               # ('Output/Reg_Pi0Fixed_LinReg_SumsOnly/results.h5', 'ECAL/HCAL Sums Only, Linear Regression'),
               # ('Output/Reg_Pi0Fixed_xgb_SumsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums Only, XGBoost'),
               # ('Output/Reg_Pi0Fixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECALMoment Z1, XGBoost'),
               # ('Output/Reg_Pi0Fixed_xgb_ECALX2Y2Z1_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECALMoments X2, Y2, Z1, XGBoost'),
               # ('Output/Reg_Pi0Fixed_xgb_ECALXY2Z1_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECALMoments XY2, Z1, XGBoost'),
               # ('Output/Reg_Pi0Fixed_xgb_ECALMomsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECALMoments X2, Z1, XGBoost'),

               # ('Output/Reg_ChPiFixed_LinReg_SumsOnly/results.h5', 'ECAL/HCAL Sums Only, Linear Regression'),
               # ('Output/Reg_ChPiFixed_xgb_SumsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums Only, XGBoost'),
               # ('Output/Reg_ChPiFixed_xgb_ECALMomsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECAL Moments, XGBoost'),
               # ('Output/Reg_ChPiFixed_xgb_HCALMomsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, HCAL Moments, XGBoost'),
               # ('Output/Reg_ChPiFixed_xgb_AllMoms_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECAL/HCAL Moments, XGBoost'),

               # ('Output/Reg_ChPiFixed_Cut30_LinReg_SumsOnly/results.h5', 'ECAL/HCAL Sums Only, Linear Regression'),
               # ('Output/Reg_ChPiFixed_Cut30_xgb_SumsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums Only, XGBoost'),
               # ('Output/Reg_ChPiFixed_Cut30_xgb_ECALMomsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECAL Moments, XGBoost'),
               # ('Output/Reg_ChPiFixed_Cut30_xgb_HCALMomsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, HCAL Moments, XGBoost'),
               # ('Output/Reg_ChPiFixed_Cut30_xgb_XYsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, X2/XY Moments, XGBoost'),
               # ('Output/Reg_ChPiFixed_Cut30_xgb_Z1sOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, Z1 Moments, XGBoost'),
               # ('Output/Reg_ChPiFixed_Cut30_xgb_AllMoms_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECAL/HCAL Moments, XGBoost'),

               # ('Output/Reg_ChPiFixed_Cut30_xgb_AllMoms_depth3_1000rounds/results.h5', 'maxdepth 3'),
               # ('Output/Reg_ChPiFixed_Cut30_xgb_AllMoms_depth4_1000rounds/results.h5', 'maxdepth 4'),
               # ('Output/Reg_ChPiFixed_Cut30_xgb_AllMoms_depth5_1000rounds/results.h5', 'maxdepth 5'),

               #('Output/Reg_xgb_AllMoms_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECAL/HCAL Moments, XGBoost'),
               #('Output/Reg_EleFixed_Features_HCALxy_25epNoStop/results.h5', 'ECAL/HCAL Sums, ECAL/HCAL Moments, NN Relu'),
#               ('Output/Reg_EleFixed_Features_SumOnly_2hid_sigmoid_25epNoStop/results.h5', 'ECAL/HCAL Sums Only, NN Sigmoid'),
#               ('Output/Reg_EleFixed_Features_ECALMomsOnly_2hid_sigmoid_25epNoStop/results.h5', 'ECAL/HCAL Sums, ECAL Moments, NN Sigmoid'),
#               ('Output/Reg_EleFixed_Features_AllMoms_2hid_sigmoid_25epNoStop/results.h5', 'ECAL/HCAL Sums, ECAL/HCAL Moments, NN Sigmoid'),

               # ('Output/Reg_EleFixed_LinReg_SumsOnly/results.h5', 'Electrons, Linear Regression'),
               # ('Output/Reg_GammaFixed_LinReg_SumsOnly/results.h5', 'Photons, Linear Regression'),
               # ('Output/Reg_Pi0Fixed_LinReg_SumsOnly/results.h5', 'Pi0s, Linear Regression'),
               # ('Output/Reg_ChPiFixed_Cut30_LinReg_SumsOnly/results.h5', 'Charged Pions, Linear Regression'),
               # ('Output/Reg_EleFixed_xgb_ECALMomsOnly_depth3_1000rounds/results.h5', 'Electrons, XGBoost'),
               # ('Output/Reg_GammaFixed_xgb_ECALMomsOnly_depth3_1000rounds/results.h5', 'Photons, XGBoost'),
               # ('Output/Reg_Pi0Fixed_xgb_ECALMomsOnly_depth3_1000rounds/results.h5', 'Pi0s, XGBoost'),
               # ('Output/Reg_ChPiFixed_Cut30_xgb_AllMoms_depth3_1000rounds/results.h5', 'Charged Pions, XGBoost'),


              ]

outdir = 'Output/'

#particle_name = 'Electron'
particle_name = 'Photon'
#particle_name = 'Pi0'
#particle_name = 'Charged Pion'
#particle_name = 'All Particles,'

#outlabel = 'EleFixed_xgb_ecalhcal'
#outlabel = 'EleFixed_xgb_calomoms'
#outlabel = 'EleFixed_xgb_z1_hyperparams'
#outlabel = 'EleFixed_xgb_vsnn'
#outlabel = 'EleFixed_xgb_vs_linreg'

outlabel = 'GammaFixed_xgb_ecalmoms'
#outlabel = 'Pi0Fixed_xgb_ecalmoms'
#outlabel = 'ChPiFixed_Cut30_xgb_calomoms'
#outlabel = 'ChPiFixed_Cut30_xgb_hyperparams'

#outlabel = 'allparts_xgb_calomoms'

results_dict = OrderedDict()

coarse_bins = np.arange(0,501,25)
coarse_bin_centers = np.arange(12.5,501,25)

for (infile,label) in input_files:
    results = []
    with h5.File(infile,'r') as f:
        reg_pred = f['regressor_pred'][:]
        reg_true = f['regressor_true'][:]
    reldiff = (reg_true - reg_pred) / reg_true * 100.

    reldiff_means = []
    reldiff_sigmas = []
    for i in range(len(coarse_bins)-1):
        bin_lower = coarse_bins[i]
        bin_upper = coarse_bins[i+1]
        indices = (reg_true > bin_lower) & (reg_true < bin_upper)
        reldiff_bin = reldiff[indices]
        reldiff_means.append(np.mean(reldiff_bin))
        reldiff_sigmas.append(np.std(reldiff_bin))
    results_dict[label] = [reldiff_means,reldiff_sigmas]
    print label,reldiff_sigmas[0],reldiff_sigmas[-1]

# plot reldiff means
fig = plt.Figure()
ax = plt.gca()
ax.set_xlim(0,500)
#ax.set_ylim(-20,30)
ax.set_ylim(-30,20)
plt.xlabel('True Energy [GeV]')
plt.ylabel('Relative Bias [%]')
plt.title('%s Energy Regression'%(particle_name))
for label, results in results_dict.items():
    mark = 'v' if 'Linear Reg' in label else 'o'
    plt.plot(coarse_bin_centers,results[0],marker=mark,label=label)
plt.legend(loc='best')
plt.grid(True)
plt.savefig('%s/comp_mean_reldiff_vs_E_%s.eps'%(outdir,outlabel))
plt.clf()

# plot reldiff means, zoomed
fig = plt.Figure()
ax = plt.gca()
ax.set_xlim(0,500)
ax.set_ylim(-2,2)
#ax.set_ylim(-12,6)
plt.xlabel('True Energy [GeV]')
plt.ylabel('Relative Bias [%]')
plt.title('%s Energy Regression'%(particle_name))
for label, results in results_dict.items():
    mark = 'v' if 'Linear Reg' in label else 'o'
    plt.plot(coarse_bin_centers,results[0],marker=mark,label=label)
plt.legend(loc='best')
plt.grid(True)
plt.savefig('%s/comp_mean_reldiff_vs_E_%s_zoom.eps'%(outdir,outlabel))
plt.clf()


# plot reldiff sigmas
fig = plt.Figure()
ax = plt.gca()
ax.set_xlim(0,500)
ax.set_ylim(0,30)
plt.xlabel('True Energy [GeV]')
plt.ylabel('Relative Resolution [%]')
plt.title('%s Energy Regression'%(particle_name))
for label, results in results_dict.items():
    mark = 'v' if 'Linear Reg' in label else 'o'
    plt.plot(coarse_bin_centers,results[1],marker=mark,label=label)
plt.legend(loc='best')
plt.grid(True)
plt.savefig('%s/comp_sigma_reldiff_vs_E_%s.eps'%(outdir,outlabel))
plt.clf()

# plot reldiff sigmas, log scale
fig = plt.Figure()
ax = plt.gca()
ax.set_xlim(0,500)
ax.set_ylim(0.5,50)
#ax.set_ylim(0.5,200)
plt.xlabel('True Energy [GeV]')
plt.ylabel('Relative Resolution [%]')
plt.title('%s Energy Regression'%(particle_name))
for label, results in results_dict.items():
    mark = 'v' if 'Linear Reg' in label else 'o'
    plt.plot(coarse_bin_centers,results[1],marker=mark,label=label)
plt.legend(loc='best')
plt.yscale('log')
plt.grid(True,which='both')
plt.savefig('%s/comp_sigma_reldiff_vs_E_%s_log.eps'%(outdir,outlabel))
plt.clf()

