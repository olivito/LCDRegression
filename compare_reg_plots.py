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

#               ### hyperparams/tuning of feature-based NN
#               ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'XGBoost'),
#               ('Output/Reg_EleFixed_Features_ECALZ1Only_2hid_relu_learnrate0p01_25epNoStop/results.h5', 'NN Relu, Alpha 0.01 L2reg 0.01 25 eps'),
#               #('Output/Reg_EleFixed_Features_ECALZ1Only_2hid_relu_learnrate0p001_25epNoStop/results.h5', 'NN Relu, Alpha 0.001 L2reg 0.01 25 eps'),
#               #('Output/Reg_EleFixed_Features_ECALZ1Only_2hid_relu_learnrate0p001_50epNoStop/results.h5', 'NN Relu, Alpha 0.001 L2reg 0.01 50 eps'),
#               ### lowest bias, maybe by luck? 
#               ('Output/Reg_EleFixed_Features_ECALZ1Only_2hid_relu_learnrate0p001_50epNoStop_try2/results.h5', 'NN Relu, Alpha 0.001 L2reg 0.01 50 eps Try2'),
# #              ('Output/Reg_EleFixed_Features_ECALZ1Only_2hid_relu_learnrate0p001_L2reg0p1_50epNoStop/results.h5', 'NN Relu, Alpha 0.001 L2reg 0.1 50 eps'),
#               ('Output/Reg_EleFixed_Features_ECALZ1Only_Skip_2hid_relu_learnrate0p001_50epNoStop/results.h5', 'NN Skip Relu, Alpha 0.001 L2reg 0.01 50 eps'),
#               ### fastest convergence, ok bias, tied for best resolution 
#               ('Output/Reg_EleFixed_Features_ECALZ1Only_SkipInit_2hid_relu_learnrate0p001_50epNoStop/results.h5', 'NN Skip Init Relu, Alpha 0.001 L2reg 0.01 50 eps'),
# #              ('Output/Reg_EleFixed_Features_ECALZ1Only_Skip_2hid_relu_learnrate0p001_L2reg0p1_50epNoStop/results.h5', 'NN Skip Relu, Alpha 0.001 L2reg 0.1 50 eps'),

              # ### hyperparams/tuning of cell-based DNN
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'XGBoost Features'),
              # ('Output/Reg_EleFixed_Features_ECALZ1Only_SkipInit_2hid_relu_learnrate0p001_50epNoStop/results.h5', 'NN Features Skip Init Relu, 50 eps'),
              # #('Output/Reg_EleFixed_DNNSmallLayersSumSkipInit_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_10epNoStop/results.h5', 'NN Layers Skip Init Relu, 10 eps'),
              # ('Output/Reg_EleFixed_DNNSmallLayersSumSkip2Init_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_10epNoStop/results.h5', 'NN Layers Skip2 Init Relu, 10 eps'),
              # ('Output/Reg_EleFixed_DNNSmall_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_10epNoStop/results.h5', 'NN Cells Relu, 10 eps'),
              # #('Output/Reg_EleFixed_DNNSmallSumSkip_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_10epNoStop/results.h5', 'NN Cells Skip Relu, 10 eps'),
              # #('Output/Reg_EleFixed_DNNSmallSumSkip_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_25epNoStop/results.h5', 'NN Cells Skip Relu, 25 eps'),
              # #('Output/Reg_EleFixed_DNNSmallSumSkipInit_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_1ep/results.h5', 'NN Cells Skip Init Relu, 1 ep'),
              # #('Output/Reg_EleFixed_DNNSmallSumSkipInit_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_5epNoStop/results.h5', 'NN Cells Skip Init Relu, 5 eps'),
              # #('Output/Reg_EleFixed_DNNSmallSumSkipInit_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_10epNoStop/results.h5', 'NN Cells Skip Init Relu, 10 eps'),
              # ('Output/Reg_EleFixed_DNNSmallSumSkip2Init_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_10epNoStop/results.h5', 'NN Cells Skip2 Init Relu, 10 eps'),
              # #('Output/Reg_EleFixed_DNNSmallSumSkipInit_2hid_512nodes_sigmoid_learnrate0p001_L2reg0p01_dropout0p2_5epNoStop/results.h5', 'NN Cells Skip Init Sigmoid, 5 eps'),
              # #('Output/Reg_EleFixed_DNNSmallSumSkipInit_4hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_10epNoStop/results.h5', 'NN Cells Skip Init Relu, 4 hidden layers, 10 eps'),
              # #('Output/Reg_EleFixed_DNNSmallSumSkipInit_2hid_1024nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_10epNoStop/results.h5', 'NN Cells Skip Init Relu, 1024 nodes, 2 hidden layers, 10 eps'),


              # ### comparison features vs layers vs cells
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'XGBoost Features'),
              # ('Output/Reg_EleFixed_Features_ECALZ1Only_SkipInit_2hid_relu_learnrate0p001_50epNoStop/results.h5', 'DNN w/ Features'),
              # ('Output/Reg_EleFixed_DNNSmallLayersSumSkip2Init_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_10epNoStop/results.h5', 'DNN w/ Layers'),
              # ('Output/Reg_EleFixed_DNNSmallSumSkip2Init_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_10epNoStop/results.h5', 'DNN w/ Cells'),

#               ### comparison on size of window used
#               ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'XGBoost Features'),
#               ('Output/Reg_EleFixed_DNNCellsSumSkip2Init_ecal13_hcal3_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_5epNoStop/results.h5', 'DNN w/ Cells E/HCAL 13/3'),
# #              ('Output/Reg_EleFixed_DNNCellsSumSkip2Init_ecal17_hcal3_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_10epNoStop/results.h5', 'DNN w/Cells E/HCAL 17/3'),
#               ('Output/Reg_EleFixed_DNNSmallSumSkip2Init_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_10epNoStop/results.h5', 'DNN w/ Cells E/HCAL 25/5'),
# #              ('Output/Reg_EleFixed_DNNCellsSumSkip2Init_ecal35_hcal7_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_5epNoStop/results.h5', 'DNN w/ Cells E/HCAL 35/7'),
#               ('Output/Reg_EleFixed_DNNCellsSumSkip2Init_ecal51_hcal11_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_5epNoStop/results.h5', 'DNN w/ Cells E/HCAL 51/11'),

              # ### CNN: reg vs skip cons
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'XGBoost Features'),
              # ('Output/Reg_EleFixed_DNNCellsSumSkip2Init_ecal51_hcal11_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_5epNoStop/results.h5', 'DNN w/ Cells, E/HCAL 51/11 cells'),
              # ('Output/Reg_EleFixed_CNNCells_ecal25_hcal5_ecalconv3_hcalconv10_learnrate0p001_L2reg0p01_dropout0p2_5epNoStop/results.h5', 'CNN, E/HCAL 25/5 cells, 3/10 convs [NIPS]'),
              # ('Output/Reg_EleFixed_CNNCellsSumSkip_ecal25_hcal5_ecalconv3_hcalconv3_learnrate0p001_L2reg0p01_dropout0p2_5epNoStop/results.h5', 'CNN Skip, E/HCAL 25/5 cells, 3/3 convs'),

              ### CNN window size and num conv filters
              ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'XGBoost Features'),
              #('Output/Reg_EleFixed_Features_ECALZ1Only_SkipInit_2hid_relu_learnrate0p001_50epNoStop/results.h5', 'DNN w/ Features'),
              #('Output/Reg_EleFixed_DNNSmallSumSkip2Init_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_10epNoStop/results.h5', 'DNN w/ Cells, E/HCAL 25/5'),
              ('Output/Reg_EleFixed_DNNCellsSumSkip2Init_ecal51_hcal11_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_5epNoStop/results.h5', 'DNN w/ Cells, E/HCAL 51/11 cells'),
              ('Output/Reg_EleFixed_CNNCellsSumSkip_ecal25_hcal5_ecalconv3_hcalconv3_learnrate0p001_L2reg0p01_dropout0p2_5epNoStop/results.h5', 'CNN Skip, E/HCAL 25/5 cells, 3/3 convs'),
              ('Output/Reg_EleFixed_CNNCellsSumSkip_ecal25_hcal5_ecalconv10_hcalconv3_learnrate0p001_L2reg0p01_dropout0p2_5epNoStop/results.h5', 'CNN Skip, E/HCAL 25/5 cells, 10/3 convs'),
              ('Output/Reg_EleFixed_CNNCellsSumSkip_ecal51_hcal11_ecalconv10_hcalconv3_learnrate0p001_L2reg0p01_dropout0p2_5epNoStop/results.h5', 'CNN Skip, E/HCAL 51/11 cells, 10/3 convs'),



              # ### summary with xgb, feature NN, cell NN
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'XGBoost w/Features'),
              # ('Output/Reg_EleFixed_Features_ECALZ1Only_SkipInit_2hid_relu_learnrate0p001_50epNoStop/results.h5', 'NN w/Features'),
              # ('Output/Reg_EleFixed_DNNSmallSumSkipInit_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_10epNoStop/results.h5', 'NN w/Cells'),

              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'Train & Test Full Energy Range'),
              # ('Output/Reg_TrainEleFixed_Elt400_TestEleFixed_allE_xgb_ECALZ1Only_depth3_1000rounds_nostop/results.h5', 'Train E < 400 GeV, Test Full Range'),

              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'Train & Test Electrons'),
              # ('Output/Reg_GammaFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'Train & Test Photons'),
              # ('Output/Reg_TrainEleFixed_TestGammaFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'Train Electrons, Test Photons'),

              # ('Output/Reg_TrainGammaFixed_EvalGammaFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'Train & Test Photons'),
              # ('Output/Reg_TrainGammaFixed_EvalEleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'Train Photons, Test Electrons'),
              # ('Output/Reg_TrainGammaFixed_EvalPi0Fixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'Train Photons, Test Pi0s'),
              # ('Output/Reg_TrainGammaFixed_EvalChPiFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'Train Photons, Test Charged Pions'),

              # ('Output/Reg_EleFixed_LinReg_SumsOnly/results.h5', 'Linear Reg, Test & Train on Fixed Angle'),
              # ('Output/Reg_EleVariable_LinReg_SumsOnly/results.h5', 'Linear Reg, Test & Train on Variable Angle'),
              # ('Output/Reg_TrainEleFixed_TestEleVariable_LinReg_SumsOnly/results.h5', 'Linear Reg, Train Fixed, Test Variable Angle'),
              # ('Output/Reg_TrainEleVariable_TestEleFixed_LinReg_SumsOnly/results.h5', 'Linear Reg, Train Variable, Test Fixed Angle'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'XGBoost, Train & Test on Fixed Angle'),
              # ('Output/Reg_EleVariable_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'XGBoost, Train & Test on Variable Angle'),
              # ('Output/Reg_TrainEleFixed_TestEleVariable_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'XGBoost, Train Fixed, Test Variable Angle'),
              # ('Output/Reg_TrainEleVariable_TestEleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'XGBoost, Train Variable, Test Fixed Angle'),

              # ('Output/Reg_EleVariable_xgb_ECALZ1OnlyEta_depth3_1000rounds/results.h5', 'Using E'),
              # ('Output/Reg_EleVariable_xgb_ECALZ1OnlyEta_Et_depth3_1000rounds/results.h5', 'Using Et'),

              # ('Output/Reg_EleVariable_xgb_ECALZ1OnlyEta_Et_depth3_1000rounds/results.h5', 'maxdepth 3'),
              # ('Output/Reg_EleVariable_xgb_ECALZ1OnlyEta_Et_depth5_minchildwt5_1000rounds/results.h5', 'maxdepth 5, minchildweight 5'),
              # ('Output/Reg_EleVariable_xgb_ECALZ1OnlyEta_Et_depth10_1000rounds/results.h5', 'maxdepth 10'),
              # ('Output/Reg_EleVariable_xgb_ECALZ1OnlyEta_Et_depth10_minchildwt5_1000rounds/results.h5', 'maxdepth 10, minchildweight 5'),

 #             ('Output/Reg_EleVariable_xgb_ECALZ1OnlyEta_Et_depth3_1000rounds/results.h5', 'Train & Test Variable Angle'),
#              ('Output/Reg_EleVariable_xgb_ECALZ1OnlyEta_Et_etalt0p2_depth3_1000rounds/results.h5', 'Train & Test Variable |eta| < 0.2 '),
#              ('Output/Reg_TrainEleFixed_TestEleVariable_etalt0p05_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'Train Fixed Angle, Test Variable |eta| < 0.05 '),
#              ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'Train & Test Fixed Angle'),

              # ('Output/Reg_EleVariable_xgb_ECALZ1OnlyEta_Et_depth3_1000rounds/results.h5', 'Train & Test Variable Angle'),
              # ('Output/Reg_TrainEleFixed_TestEleVariable_etalt0p05_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'Train Fixed Angle, Test Variable |$\eta$| < 0.05 '),
              # ('Output/Reg_EleVariable_etalt0p05_xgb_ECALZ1EtaOnly_depth3_1000rounds/results.h5', 'Train & Test Variable |$\eta$| < 0.05 '),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'Train & Test Fixed Angle'),
              # ('Output/Reg_EleFixed_newsetup_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'Train & Test Fixed Angle, New Setup'),

#               ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'Normal XGBoost'),
#               ('Output/Reg_EleFixed_xgb_residuals_add_ECALZ1Only_depth3_1000rounds/results.h5', 'Additive Residual on Lin Reg'),
# #              ('Output/Reg_EleFixed_xgb_residuals_mult_ECALZ1Only_depth3_1000rounds/results.h5', 'Multiplicative Residual on Lin Reg'),
#               ('Output/Reg_TrainEleFixed_Elt400_TestEleFixed_allE_xgb_residuals_ECALZ1Only_depth3_1000rounds/results.h5', 'Additive Residual, Train < 400 GeV Early Stopping'),
#               ('Output/Reg_TrainEleFixed_Elt400_TestEleFixed_allE_xgb_residuals_ECALZ1Only_depth3_1000rounds_nostop/results.h5', 'Additive Residual, Train < 400 GeV 1000 rounds'),

              # ('Output/Reg_EleVariable_LinReg_SumsOnly/results.h5', 'ECAL/HCAL Sums Only, Linear Regression'),
              # ('Output/Reg_EleVariable_xgb_SumsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums Only, XGBoost'),
              # ('Output/Reg_EleVariable_xgb_SumsOnlyEta_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, Eta, XGBoost'),
              # ('Output/Reg_EleVariable_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECALmomentZ1, XGBoost'),
              # ('Output/Reg_EleVariable_xgb_ECALZ1OnlyEta_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECALmomentZ1, Eta, XGBoost'),

              # ('Output/Reg_EleVariable_xgb_SumsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums Only'),
              # ('Output/Reg_EleVariable_xgb_SumsOnlyEta_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, Eta'),
              # ('Output/Reg_EleVariable_xgb_SumsOnlyEta_depth10_minchildwt5_1000rounds/results.h5', 'ECAL/HCAL Sums, Eta, maxdepth 10, minchildweight 5'),
              # ('Output/Reg_EleVariable_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECALmomentZ1'),
              # ('Output/Reg_EleVariable_xgb_ECALZ1OnlyEta_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECALmomentZ1, Eta'),

              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth2_1000rounds/results.h5', 'maxdepth 2'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'maxdepth 3'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_eta0p9_1000rounds/results.h5', 'maxdepth 3, eta 0.9'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_minchildwt5_1000rounds/results.h5', 'maxdepth 3, minchildweight 5'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth4_1000rounds/results.h5', 'maxdepth 4'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth4_minchildwt5_1000rounds/results.h5', 'maxdepth 4, minchildweight 5'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth5_minchildwt5_1000rounds/results.h5', 'maxdepth 5, minchildweight 5'),

               # ('Output/Reg_GammaFixed_LinReg_SumsOnly/results.h5', 'ECAL/HCAL Sums Only, Linear Regression'),
               # ('Output/Reg_GammaFixed_xgb_SumsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums Only, XGBoost'),
               # ('Output/Reg_GammaFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECALMomentZ1, XGBoost'),
               # ('Output/Reg_GammaFixed_xgb_ECALMomsOnly_depth3_1000rounds/results.h5', 'ECAL/HCAL Sums, ECAL Moments, XGBoost'),

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

particle_name = 'Electron'
#particle_name = 'Photon'
#particle_name = 'Pi0'
#particle_name = 'Charged Pion'
#particle_name = 'All Particles,'
#particle_name = 'Electron / Photon'

#outlabel = 'Ele_xgb_angles'

#outlabel = 'EleFixed_xgb_residual_extrap'

#outlabel = 'EleFixed_xgb_ecalhcal'
#outlabel = 'EleFixed_xgb_ecut'
#outlabel = 'EleFixed_xgb_calomoms_finebins'
#outlabel = 'EleFixed_xgb_z1_hyperparams'
#outlabel = 'EleFixed_xgb_vsnn_cells'
#outlabel = 'EleFixed_nn_inputs'
#outlabel = 'EleFixed_nn_numcells'
#outlabel = 'EleFixed_cnn_skip'
outlabel = 'EleFixed_cnn_numcells'
#outlabel = 'EleFixed_nn_vs_cnn'
#outlabel = 'EleFixed_xgb_vsnn_summary'
#outlabel = 'EleFixed_xgb_vs_linreg'
#outlabel = 'EleVariable_xgb_hyperparams'
#outlabel = 'EleVariable_xgb_etacut'
#outlabel = 'EleVariable_xgb_etavars'

#outlabel = 'EleVariable_xgb_newsetup'

#outlabel = 'GammaFixed_xgb_ecalmoms'
#outlabel = 'Pi0Fixed_xgb_ecalmoms_finebins'
#outlabel = 'ChPiFixed_Cut30_xgb_calomoms'
#outlabel = 'ChPiFixed_Cut30_xgb_hyperparams'

#outlabel = 'GammaFixed_xgb_cross'

#outlabel = 'allparts_xgb_calomoms'

results_dict = OrderedDict()

coarse_bins = np.arange(0,501,25)
coarse_bin_centers = np.arange(12.5,501,25)

#coarse_bins = np.arange(10,501,10)
#coarse_bin_centers = np.arange(15,501,10)

#coarse_bins = np.arange(10,501,5)
#coarse_bin_centers = np.arange(12.5,501,5)

for (infile,label) in input_files:
    results = []
    with h5.File(infile,'r') as f:
        reg_pred = f['regressor_pred'][:].reshape(-1)
        reg_true = f['regressor_true'][:].reshape(-1)
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

