# -*- coding: utf-8 -*-
"""

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author: Erik A. Kneller
"""
import os
import math
import fileIO
import map_tools
import fluid_props
import numpy as np
import numba
from numba import jit


class Srckinetics:
    
    
    def __init__(self, model):
#                 input_path, script_path, imass_gen,
#                 iuse_high_lom_gas, src_top_names,
#                 inert_frac, adsorption_perc, 
#                 TRmin, TRmax, 
#                 gas_frac_adj_TRmin, 
#                 gas_frac_adj_TRmax
        self.imass_gen = model.imass_gen
        self.file_path_local = os.path.join(
            model.input_path, "kinetics_library.csv")
        self.file_path_master = os.path.join(
            model.script_path,"ipa_data", "kinetics_library.csv")
        self.iuse_high_lom_gas = model.iuse_high_lom_gas
        self.src_top_names = model.src_top_names
        self.inert_frac = model.inert_frac
        self.adsorption_perc = model.adsorption_perc
        self.TRmin = model.TRmin
        self.TRmax = model.TRmax
        self.gas_frac_adj_TRmin = model.gas_frac_adj_TRmin
        self.gas_frac_adj_TRmax = model.gas_frac_adj_TRmax
        
        
    def load_kinetics(self):
        ifind_kinetics = 0
        if self.imass_gen == 1 or self.imass_gen == 2:
            if os.path.isfile(self.file_path_local) == True:
                print("Found kinetics library file kinetics_library.csv in"
                      " local input directory.")
                ifind_kinetics = 1
                input_file_path = self.file_path_local
            else:
                print("Did not find kinetics library file kinetics_library.csv"
                      " in local input directory. Checking master directory..")
                if os.path.isfile(self.file_path_master) == True:
                    print("Found kinetics library file in master location : ")
                    print(self.file_path_master)
                    ifind_kinetics = 1
                    input_file_path = self.file_path_master
            if ifind_kinetics == 1:
                ifind_kinetics = 1
                (
                    self.nEa,
                    self.Ea,
                    self.A_Early_all,
                    self.f_Early_all,
                    self.A_Normal_all,
                    self.f_Normal_all,
                    self.A_Late_all,
                    self.f_Late_all,
                    self.A_OilCrack,
                    self.f_OilCrack_all,
                    self.A_HighLOM_TII,
                    self.f_HighLOM_TII_all,
                    self.A_HighLOM_TIII,
                    self.f_HighLOM_TIII_all
                )   = fileIO.read_kinetics_library_csv(input_file_path)
            if ifind_kinetics == 0:
                self.imass_gen = 0
        else:
            print("Mass generation is turned off. Source kinetics library was "
                  "not loaded. imass_gen = ", self.imass_gen)
            

def get_kinetics_library():
    nEa = 40
    # 1/Myr
    A_vk_norm_all = np.array([2.4774E+27, 1.2764E+27, 2.5704E+29, 3.3300E+25])
    f_vk_norm_all = np.array([
        [40.0, 0, 0, 0, 0.00545],
        [41.0, 0, 0.00001, 0, 0.01422],
        [42.0, 0, 0.00005, 0, 0.03166],
        [43.0, 0, 0.00024, 0.00001, 0.06004],
        [44.0, 0, 0.00095, 0.00003, 0.09702],
        [45.0, 0, 0.00317, 0.00009, 0.13361],
        [46.0, 0, 0.00896, 0.00022, 0.1568],
        [47.0, 0, 0.02160, 0.00051, 0.1568],
        [48.0, 0, 0.04437, 0.00111, 0.13361],
        [49.0, 0, 0.07767, 0.00227, 0.09702],
        [50.0, 0.0062, 0.11588, 0.00438, 0.06004],
        [51.0, 0.0606, 0.14731, 0.00793, 0.03166],
        [52.0, 0.2417, 0.15958, 0.0135, 0.01422],
        [53.0, 0.3829, 0.14731, 0.02157, 0.00545],
        [54.0, 0.2417, 0.11588, 0.03238, 0.00178],
        [55.0, 0.0606, 0.07767, 0.04566, 0.00049],
        [56.0, 0.006, 0.04437, 0.06049, 0.00012],
        [57.0, 0.0003, 0.0216, 0.07528, 0.00002],
        [58.0, 0, 0.00896, 0.08802, 0],
        [59.0, 0, 0.00317, 0.09667, 0],
        [60.0, 0, 0.00095, 0.09974, 0],
        [61.0, 0, 0.00024, 0.09667, 0],
        [62.0, 0, 0.00005, 0.08802, 0],
        [63.0, 0, 0.00001, 0.07528, 0],
        [64.0, 0, 0, 0.06049, 0],
        [65.0, 0, 0, 0.04566, 0],
        [66.0, 0, 0, 0.03238, 0],
        [67.0, 0, 0, 0.02157, 0],
        [68.0, 0, 0, 0.0135, 0],
        [69.0, 0, 0, 0.00793, 0],
        [70.0, 0, 0, 0.00438, 0],
        [71.0, 0, 0, 0.00227, 0],
        [72.0, 0, 0, 0.00111, 0],
        [73.0, 0, 0, 0.00051, 0],
        [74.0, 0, 0, 0.00022, 0],
        [75.0, 0, 0, 0.00009, 0],
        [76.0, 0, 0, 0.00003, 0],
        [77.0, 0, 0, 0.00001, 0],
        [78.0, 0, 0, 0, 0],
        [79.0, 0, 0, 0, 0],
        [80.0, 0, 0, 0, 0],
        ])
    
    # 1/Myr
    A_vk_early_all = np.array([1.2764E+27,3.3300E+25,9.1620E+26,3.3300E+25])
    f_vk_early_all = np.array([
    [40,0,0,0,0.005446317],
    [41,0,0,0,0.014224127],
    [42,0,0,0.0015,0.031656378],
    [43,0,0.0014,0.0025,0.060035715],
    [44,0,0.0214,0.006,0.097022183],
    [45,0,0.1359,0.0129,0.133611942],
    [46,0,0.3413,0.025,0.156795066],
    [47,0,0.3413,0.0434,0.156795066],
    [48,0,0.1359,0.0674,0.133611942],
    [49,0.0062,0.0214,0.0938,0.097022183],
    [50,0.0606,0.0014,0.1169,0.060035715],
    [51,0.2417,0,0.1306,0.031656378],
    [52,0.3829,0,0.1306,0.014224127],
    [53,0.2417,0,0.1169,0.005446317],
    [54,0.0606,0,0.0938,0.001777023],
    [55,0.006,0,0.0674,0.000494079],
    [56,0.0003,0,0.0434,0.000117061],
    [57,0,0,0.025,2.36342E-05],
    [58,0,0,0.0129,4.06614E-06],
    [59,0,0,0.006,5.96125E-07],
    [60,0,0,0.0025,7.4474E-08],
    [61,0,0,0.0015,7.92839E-09],
    [62,0,0,0,7.19247E-10],
    [63,0,0,0,5.56012E-11],
    [64,0,0,0,3.66272E-12],
    [65,0,0,0,2.05606E-13],
    [66,0,0,0,9.83513E-15],
    [67,0,0,0,4.00902E-16],
    [68,0,0,0,1.39254E-17],
    [69,0,0,0,4.12185E-19],
    [70,0,0,0,1.03965E-20],
    [71,0,0,0,2.23459E-22],
    [72,0,0,0,4.09279E-24],
    [73,0,0,0,6.38784E-26],
    [74,0,0,0,8.49576E-28],
    [75,0,0,0,9.62859E-30],
    [76,0,0,0,9.299E-32],
    [77,0,0,0,7.65285E-34],
    [78,0,0,0,5.36689E-36],
    [79,0,0,0,3.20726E-38],
    [80,0,0,0,1.63328E-40],
    ])
    # 1/Myr 
    A_vk_late_all = np.array([1.3243E+29,1.3243E+29,1.3740E+31,3.3300E+25])
    f_vk_late_all = np.array([
    [40,0,0,0,0.005446317],
    [41,0,0,0,0.014224127],
    [42,0,0,0,0.031656378],
    [43,0,0,0,0.060035715],
    [44,0,0,0,0.097022183],
    [45,0,0,0,0.133611942],
    [46,0,0,0,0.156795066],
    [47,0,0,0,0.156795066],
    [48,0,0,0,0.133611942],
    [49,0,0,0,0.097022183],
    [50,0,0,0,0.060035715],
    [51,0,0,0,0.031656378],
    [52,0,0,0,0.014224127],
    [53,0,0,0,0.005446317],
    [54,0,0.0013,0,0.001777023],
    [55,0,0.0085,0,0.000494079],
    [56,0.0062,0.038,0,0.000117061],
    [57,0.0606,0.1109,0.0023,2.36E-05],
    [58,0.2417,0.2108,0.0039,4.07E-06],
    [59,0.3829,0.2611,0.0089,5.96E-07],
    [60,0.2417,0.2108,0.0182,7.45E-08],
    [61,0.0606,0.1109,0.0334,7.93E-09],
    [62,0.006,0.038,0.0549,7.19E-10],
    [63,0.0003,0.0085,0.0807,5.56E-11],
    [64,0,0.0012,0.1062,3.66E-12],
    [65,0,0,0.1253,2.06E-13],
    [66,0,0,0.1324,9.84E-15],
    [67,0,0,0.1253,4.01E-16],
    [68,0,0,0.1062,1.39E-17],
    [69,0,0,0.0807,4.12E-19],
    [70,0,0,0.0549,1.04E-20],
    [71,0,0,0.0334,2.23E-22],
    [72,0,0,0.0182,4.09E-24],
    [73,0,0,0.0089,6.39E-26],
    [74,0,0,0.0039,8.50E-28],
    [75,0,0,0.0015,9.63E-30],
    [76,0,0,0.0008,9.30E-32],
    [77,0,0,0,7.65E-34],
    [78,0,0,0,5.37E-36],
    [79,0,0,0,3.21E-38],
    [80,0,0,0,1.63E-40],
    ])
    # 1/Myr
    A_OilCrack = 2.018559626E+29 
    f_OilCrack = np.array([
    [40,0],
    [41,0],
    [42,0],
    [43,0],
    [44,0],
    [45,0],
    [46,0],
    [47,0],
    [48,0],
    [49,0],
    [50,0],
    [51,0],
    [52,0],
    [53,0],
    [54,0],
    [55,7.20307E-06],
    [56,8.85138E-05],
    [57,0.000769536],
    [58,0.004733391],
    [59,0.020598742],
    [60,0.063421156],
    [61,0.138150574],
    [62,0.212910131],
    [63,0.232148057],
    [64,0.179084861],
    [65,0.097741227],
    [66,0.037741729],
    [67,0.010310765],
    [68,0.001992898],
    [69,0.000272524],
    [70,2.63663E-05],
    [71,0],
    [72,0],
    [73,0],
    [74,0],
    [75,0],
    [76,0],
    [77,0],
    [78,0],
    [79,0],
    [80,0],
    ])
    A_HighLOM = 6.31e26
    f_HighLOM_type3 = np.array([
    [40,0],
    [41,0],
    [42,0],
    [43,0],
    [44,0],
    [45,0],
    [46,0],
    [47,0],
    [48,0],
    [49,0],
    [50,0],
    [51,0],
    [52,0],
    [53,0],
    [54,0],
    [55,0],
    [56,0],
    [57,0],
    [58,0],
    [59,0],
    [60,0.315],
    [61,0],
    [62,0.275],
    [63,0],
    [64,0.165],
    [65,0],
    [66,0.125],
    [67,0],
    [68,0.085],
    [69,0],
    [70,0.035],
    [71,0],
    [72,0],
    [73,0],
    [74,0],
    [75,0],
    [76,0],
    [77,0],
    [78,0],
    [79,0],
    [80,0],
    ])    
    f_HighLOM = np.array([
    [40,0],
    [41,0],
    [42,0],
    [43,0],
    [44,0],
    [45,0],
    [46,0],
    [47,0],
    [48,0],
    [49,0],
    [50,0],
    [51,0],
    [52,0],
    [53,0],
    [54,0],
    [55,0],
    [56,0],
    [57,0],
    [58,0],
    [59,0],
    [60,0.459330144],
    [61,0],
    [62,0.272727273],
    [63,0],
    [64,0.162679426],
    [65,0],
    [66,0.057416268],
    [67,0],
    [68,0.04784689],
    [69,0],
    [70,0],
    [71,0],
    [72,0],
    [73,0],
    [74,0],
    [75,0],
    [76,0],
    [77,0],
    [78,0],
    [79,0],
    [80,0],
    ])
    return (nEa,
            A_vk_early_all, f_vk_early_all, 
            A_vk_norm_all, f_vk_norm_all, 
            A_vk_late_all, f_vk_late_all,
            A_OilCrack, f_OilCrack,
            A_HighLOM, f_HighLOM, 
            f_HighLOM_type3
            )


@jit(nopython=True, cache=True)
def define_kinetics_np(
                        iomt, isktype, 
                        A_vk_early_all, f_vk_early_all, 
                        A_vk_norm_all, f_vk_norm_all, 
                        A_vk_late_all, f_vk_late_all,
                        A_OilCrack, f_OilCrack,
                        A_HighLOM, f_HighLOM, 
                        f_HighLOM_type3, nEa
):
    """ Define kinetic model
    
    Input Description
    -----------------
    iomt = 1 = "Type_I"
    iomt = 2 = "Type_II"
    iomt = 3 = "Type_III"
    iomt = 4 = "Type_IIS"
    
    isktype = -1 = "early"
    isktype = 0 = "normal"
    isktype = 1 = "late"
    isktype = 2 = "OilCrack"
    isktype = 3 = "HighLOM"
    
    """  
    f_sel = np.zeros((nEa))
    Ea_sel = np.zeros((nEa))    
    for i in range(nEa):
        if isktype == 0:
            # Normal
            f_sel[i] = f_vk_norm_all[i][iomt]
            Ea_sel[i] = f_vk_norm_all[i][0]
            A = A_vk_norm_all[iomt-1]            
        elif isktype == -1:
            # Early
            f_sel[i] = f_vk_early_all[i][iomt]
            Ea_sel[i] = f_vk_early_all[i][0]
            A = A_vk_early_all[iomt-1]            
        elif isktype == 1:
            # Late
            f_sel[i] = f_vk_late_all[i][iomt]
            Ea_sel[i] = f_vk_late_all[i][0]
            A = A_vk_late_all[iomt-1]
        elif isktype == 2:
            # Oil Cracking
            f_sel[i] = f_OilCrack[i][1]
            Ea_sel[i] = f_OilCrack[i][0]
            A = A_OilCrack
        elif isktype == 3:
            # High LOM Gas
            if iomt == 3:
                f_sel[i] = f_HighLOM_type3[i][1]
                Ea_sel[i] = f_HighLOM_type3[i][0]
            else:
                f_sel[i] = f_HighLOM[i][1]
                Ea_sel[i] = f_HighLOM[i][0]
            A = A_HighLOM   
    return A, f_sel, Ea_sel


@jit(nopython=True, cache=True)
def oil_gas_split_params(
                            HI, HI_lup, fkp_lup, fp_lup, ro_lup, rp_lup,
                            igas_frac_type, ipolar_frac_type
):
    
    # HI	 fkp	   fp	       ro	       rp
    # 0	 0.45000	0.07000	1.53000 	1.26000
    # 200	 0.45000	0.07000	1.53000	1.26000
    # 600	 0.50000	0.10000	1.74000	1.38000
    # 900	 0.30000	0.07000	1.78000	1.72000
    # 1200	 0.30000	0.07000	1.78000	1.72000
    HI_lup[0] = 0
    HI_lup[1] = 200
    HI_lup[2] = 600
    HI_lup[3] = 900
    HI_lup[4] = 1200
    #
    fp_lup[0] = 0.07
    fp_lup[1] = 0.07
    fp_lup[2] = 0.1
    fp_lup[3] = 0.07
    fp_lup[4] = 0.07
    #
    ro_lup[0] = 1.53
    ro_lup[1] = 1.53
    ro_lup[2] = 1.74
    ro_lup[3] = 1.78
    ro_lup[4] = 1.78
    #
    rp_lup[0] = 1.26
    rp_lup[1] = 1.36
    rp_lup[2] = 1.38
    rp_lup[3] = 1.72
    rp_lup[4] = 1.72
    if ipolar_frac_type == -1: # minimum
        fkp_lup[0] = 0.3
        fkp_lup[1] = 0.3
        fkp_lup[2] = 0.25
        fkp_lup[3] = 0.2
        fkp_lup[4] = 0.2
    elif ipolar_frac_type == 0: # minimum
        fkp_lup[0] = 0.45
        fkp_lup[1] = 0.45
        fkp_lup[2] = 0.5
        fkp_lup[3] = 0.3
        fkp_lup[4] = 0.3
    elif ipolar_frac_type == 1: # minimum
        fkp_lup[0] = 0.6
        fkp_lup[1] = 0.6
        fkp_lup[2] = 0.75
        fkp_lup[3] = 0.4
        fkp_lup[4] = 0.4
    #	Inputs for primary gas fraction calculation
    # minimum	
    if igas_frac_type == -1:
        fkg0 = 0.32
        HI1 = 280.00
        fkg1 = 0.05
        fkg1200 = 0.05
    # base
    elif igas_frac_type == 0:
        fkg0 = 0.41000
        HI1 = 390.00
        fkg1 = 0.07500
        fkg1200 = 0.07500
    # maximum
    elif igas_frac_type == 1:
        fkg0 = 0.5
        HI1 = 500
        fkg1 = 0.1
        fkg1200 = 0.1
    #	Inputs for coke coefficients for gas and oil	
    rc = 0.4500 # H/C atomic ratio of coke
    rg = 2.9000 # H/C atomic ratio of gas
    # Primary gas fraction
    if HI < HI1:
        fkg = fkg0-(fkg0-fkg1)*(HI/HI1)
    else:
        fkg = fkg1200
    #Primary polar fraction    
    if HI > HI_lup[3]: 
        fkp = fkp_lup[3]
    elif HI > HI_lup[2]:
        fkp = (
                  fkp_lup[2]
                + (fkp_lup[3]-fkp_lup[2])*
                                        ((HI-HI_lup[2])/(HI_lup[3]-HI_lup[2]))
            )
    elif HI > HI_lup[1]:
        fkp = (
                  fkp_lup[1]
                + (fkp_lup[2]-fkp_lup[1])*
                                        ((HI-HI_lup[1])/(HI_lup[2]-HI_lup[1]))
            )
    else:
        fkp = fkp_lup[0]
    #Primary oil fraction
    fko = 1-fkg-fkp
    #Ratio of primary oil over primary gas
    R = fko/fkg
    # Polar fraction of total oil 
    if HI > HI_lup[3]:
        fp = fp_lup[3]
    elif HI > HI_lup[2]:
        fp = (
                  fp_lup[2]
                + (fp_lup[3]-fp_lup[2])*((HI-HI_lup[2])/(HI_lup[3]-HI_lup[2]))
            )
    elif HI > HI_lup[1]:
        fp = (
                  fp_lup[1]
                + (fp_lup[2]-fp_lup[1])*((HI-HI_lup[1])/(HI_lup[2]-HI_lup[1]))
            )
    else:
        fp = fp_lup[0]
    # H/C atomic ratio of oil
    if HI > HI_lup[3]:
        ro = ro_lup[3]
    elif HI > HI_lup[2]:
        ro = (
                  ro_lup[2]
                + (ro_lup[3]-ro_lup[2])*((HI-HI_lup[2])/(HI_lup[3]-HI_lup[2]))
            )
    elif HI > HI_lup[1]:
        ro = (
                  ro_lup[1]
                + (ro_lup[2]-ro_lup[1])*((HI-HI_lup[1])/(HI_lup[2]-HI_lup[1]))
            )
    else:
        ro = ro_lup[0]
    # H/C atomic ratio of polars
    if HI > HI_lup[3]:
        rp = rp_lup[3]
    elif HI > HI_lup[2]:
        rp = (
                  rp_lup[2]
                + (rp_lup[3]-rp_lup[2])*((HI-HI_lup[2])/(HI_lup[3]-HI_lup[2]))
            )
    elif HI > HI_lup[1]:
        rp = (
                  rp_lup[1]
                + (rp_lup[2]-rp_lup[1])*((HI-HI_lup[1])/(HI_lup[2]-HI_lup[1]))
            )
    else:
        rp = rp_lup[0]
    # coke coefficient for gas
    drg = ((12.0+rc)/(12.0+rg))*((rg-rp)/(rp-rc))
    # coke coefficient for oil
    dro = ((12.0+rc)/(12.0+ro))*((ro-rp)/(rp-rc))
    # gas fraction from polar cracking
    fpg = (fkp-fko*(fp/(1.0-fp)))/((1.0+drg)+(1.0/(1.0-fp)+dro)*R)
    # oil fraction from polar cracking
    fpo = R*fpg
    # fraction of stable polar compounds
    fpp = (fp/(1-fp))*(fko+fpo)
    # fraction of total gas
    ftg = fkg+fpg
    # fraction of total oil
    fto = fko+fpo+fpp
    # fraction of coke
    ftc = 1.0-(ftg+fto)
    return fkg, fkp, fko, R, fp, ro, rp, drg, dro, fpg, fpo, fpp, ftg, fto, ftc


@jit(nopython=True, cache=True)
def calc_gas_zfactor(
                        TF, Ppsi, gas_grav, oil_api, 
                        mol_frac_CO2, mol_frac_H2S
):    
    Nmain = 5
    N = 10
    z_mino = 0.001
    z_maxo = 4
    dzo = (z_maxo - z_mino)/float(N)    
    # Psuedocritical Temperature R (free gas) (eq 25)
    TR_cr = 169.2+gas_grav*(349.5-74*gas_grav)    
    #Psuedocritical Pressure psia free gas (eq 24)
    Ppsi_cr = 756.8-gas_grav*(131+3.6*gas_grav)
    # McCain epsilon ( CO2,H2Sadjustment factor)( eq 30)
    epsilon = (
                120*(
                          (mol_frac_CO2+mol_frac_H2S)**0.9
                        + (mol_frac_CO2+mol_frac_H2S)**1.6)
                        + (mol_frac_H2S**0.5-mol_frac_H2S**4
                    )
            )
    # Adjusted psuedocritical Temperature R (eq 28)
    TR_cr_adj = TR_cr - epsilon
    #Adjusted psuedocritcal Pressure (eq 29)
    Ppsi_cr_adj = (
                    Ppsi_cr*TR_cr_adj/
                                (TR_cr+mol_frac_H2S*(1-mol_frac_H2S)*epsilon)
                )
    # 1/Psuedoreduced Temperature  1/eq 23a)
    Tred_inv = TR_cr_adj/(TF+459.6)
    # Psuedoreduced Pressure (eq 23b)
    Pred = Ppsi/Ppsi_cr_adj
    zroot = 1e39
    for mm in range(Nmain):
        if mm == 0:
            z_min = z_mino
            z_max = z_maxo
            dz = dzo
        else:
            z_min = zroot - dz
            z_max = zroot + dz
            dz = (z_max - z_min)/float(N)                        
        z_new_min = 1e39
        diff_min = 1e39
        for i in range(N):
            if i == 0:
                z_old = z_min
            else:
                z_old = z_old + dz
            # rhopr (eq 22)
            rhopr = 0.27*Pred*Tred_inv/z_old
            S41 = rhopr
            Q41 = Tred_inv    
            z_new = (
                     (    1 
                        + S41*(0.3265 + Q41*(- 1.07 + Q41**2*
                                     (- 0.5339 + Q41*(0.01569 - 0.05165*Q41))))
                        + S41**2*(0.5475 + Q41*(-0.7361 + 0.1844*Q41))
                        - 0.1056*S41**5*Q41*(-0.7361 + Q41*0.1844)
                        + 0.6134*Q41**3*S41**2*(1 + 0.721*S41**2)*
                                                    math.exp(-0.721*S41**2)
                     )
                    )
            diff = z_new - z_old
            if abs(diff) < diff_min:
                diff_min = diff
                z_new_min = z_new
        zroot = z_new_min
    return zroot


@jit(nopython=True, cache=True)
def mass_generation_calc(
                            imass_calc_type, HI, TOC, rho_grain, inert_frac,
                            adsorption_perc, por_frac, fpress_psi, tSm, aSkm2, 
                            Oil_API, sgrav_gas, TR, TF,
                            HI_lup, fkp_lup, fp_lup, ro_lup, rp_lup, 
                            por_frac_last, mGg_ini, mOg_ini, mCg_ini,
                            por_threshold, igas_frac_type, ipolar_frac_type,
                            TRmin, TRmax, gas_frac_adj_TRmin, 
                            gas_frac_adj_TRmax, idebug_out
):
    # 0 = oil is not dissolved in free gas; 1 = oil is dissolved in free gas
    # if option 0 is used then GOR_g_g gas is set to 5 (~oil window value).
    # Option 1 needs to be worked more. At high pressure single phase "gas" 
    # takes up all oil and adsorption does not work.
    # To resolve the issue ensure all hydrocarbons are liquid at high pressure 
    # and reformulate density for single phase
    iuse_dissolved_oil = 0 # Do not set to 1 until above issue is resolved
    iuse_variable_gas_fac = 1
    crack_frac = 1.0 - inert_frac
    cm3_scf = 28316.85
    cm3_bbl = 158987.30
    # Areas of source for this cell    
    aSm2 = aSkm2*1e6
    # Volume of source for this cell
    vSm3 = aSm2*tSm
    # Pore capacity for expulsion
    por_cap_m3 = vSm3*por_frac*por_threshold
    # Calculate the density of oil and gas end-members at surface conditions
    # g/cm3
    rho_air_surface = 0.001292
    rho_gas_surface = rho_air_surface*sgrav_gas
    sgrav_oil = 141.5/(131.5 + Oil_API)
    rho_oil_surface = sgrav_oil # g/cm3
    # Calculate zfactor for gas
    zfactor = calc_gas_zfactor(TF, fpress_psi, sgrav_gas, Oil_API, 0, 0)
    # Calculate gas, polar and coke fractions based on HI
    if imass_calc_type == 0: # Primary generation
        (
            fkg, fkp, fko, R, 
            fp, ro, rp, drg, 
            dro, fpg, fpo, fpp, 
            ftg, fto, ftc
         ) = oil_gas_split_params(
                                     HI, HI_lup, fkp_lup, fp_lup, 
                                     ro_lup, rp_lup, igas_frac_type, 
                                     ipolar_frac_type
                                     )
        # variable
        if iuse_variable_gas_fac == 1:
            ftg_initial = ftg
            if TR <= TRmin:
                gas_frac_adj = gas_frac_adj_TRmin
            elif TR >= TRmax:
                gas_frac_adj = gas_frac_adj_TRmax
            else:    
                gas_frac_adj = (
                                    gas_frac_adj_TRmin 
                                  + (TR-TRmin)*
                                      (gas_frac_adj_TRmax - gas_frac_adj_TRmin)
                                                                 /(TRmax-TRmin)
                               )
            ftg_new = ftg*gas_frac_adj
            if ftg_new > fto + ftg:
                ftg_new = fto + ftg
            fto = fto + ftg_initial - ftg_new
            ftg = ftg_new
    # For secondary cracking we are not producing oil
    elif imass_calc_type == 1:
        ftg = 0.62
        fto = 0.0
        ftc = 0.38
    # High LOM Gas
    elif imass_calc_type == 2:
        ftg = 1.0
        fto = 0.0
        ftc = 0.0
    # Calculate pressure correction for moving from McCain 1991 to McCain 2011
    iuse_sGOR_cor = 0
    sGOR_at_charp = 380.0
    charp = 4750.0
    exp_fac = 2.4
    if iuse_sGOR_cor == 1:
        sGOR_cor = sGOR_at_charp*((fpress_psi/charp)**exp_fac)
    else:
        sGOR_cor = 0.0
    sGOR_scf_bbl = (
                    sgrav_gas*(
                                 (fpress_psi/18.2 + 1.4)*
                                             10**(0.0125*Oil_API - 0.00091*TF))
                                                                   **(1.0/0.83) 
                               + sGOR_cor
                  )
    sGOR_cm3_cm3 = sGOR_scf_bbl*cm3_scf/cm3_bbl    
    sGOR_g_g = sGOR_cm3_cm3*(rho_gas_surface/rho_oil_surface)
    # Gas saturation GOR parameters
    a_gas = 27500
    b_gas = 0.1920
    # calculate saturation GOR for gas
    if fpress_psi <= 0:
        fpress_psi_tmp = 0.01
    else:
        fpress_psi_tmp = fpress_psi
    sGORgas_scf_bbl = (a_gas/fpress_psi_tmp)**(1.0/b_gas)
    if sGORgas_scf_bbl < sGOR_scf_bbl:
        sGORgas_scf_bbl = sGOR_scf_bbl
    sGORgas_cm3_cm3 = sGORgas_scf_bbl*cm3_scf/cm3_bbl
    sGORgas_g_g = sGORgas_cm3_cm3*(rho_gas_surface/rho_oil_surface)
    # Mass of source matrix in hecto-grams (i.e. 100 g units)
    mSMAThg = (1.0 - por_frac_last)*rho_grain*vSm3*10
    # Mass of initial organic carbon in grams 
    # (note TOC is in g carbon per 100 g of dry rock)
    mOCIg = TOC*mSMAThg
    # Petroleum potential (g HC / g organic carbon)
    pet_potential_g_g = HI*TR*0.001
    # Mass of Generated Hydrocarbons (g)
    mHCg = pet_potential_g_g*mOCIg
    mHCmg = mHCg*1000.0
    # Mass of Generated Hydrocarbons (mg / g OC)
    mHCmg_gOC = mHCmg/mOCIg
    # Mass of unreacted kerogen (g) 
    # The following formulation is the Stellar approach:
    #   mKURg = (
    #                 mOCIg*crack_frac*HI/1000*(1.0 - TR) 
    #               + mOCIg - mOCIg*crack_frac*HI/1000
    #           )
    # The following is a simplified but equivalent formulation.
    #   mKURg = mOCIg - mOCIg*HI*TR*0.001*crack_frac
    mKURg = mOCIg - mOCIg*HI*TR*0.001*crack_frac
    if imass_calc_type == 0 or imass_calc_type == 2: 
        # Primary generation
        # Mass of gas (g)
        mGg = mHCg*ftg
    elif imass_calc_type == 1: 
        # Secondary cracking or high LOM gas
        # Mass of gas (g)
        mGg = mHCg*ftg + mGg_ini
    # Primary generation or high LOM gas
    if imass_calc_type == 0 or imass_calc_type == 2:
        mOg = mHCg*fto # Mass of oil (g)
    elif imass_calc_type == 1: # Secondary cracking
        mOg = max(mOg_ini - mHCg*ftg,0)
    if imass_calc_type == 0 or imass_calc_type == 2:
        # Primary generation or high LOM gas
        mCOKEg = mHCg*ftc
    elif imass_calc_type == 1: 
        # Secondary cracking
        # Mass of coke (g)
        mCOKEg = mHCg*ftc + mCg_ini
    if iuse_dissolved_oil > 0:
        if sGORgas_g_g > sGOR_g_g:        
            (
                mDOg_calc, mDGg_calc, 
                mL_g_calc, mFG_g_calc
            ) = fluid_props.calc_liquid_and_free_gas_mass(
                                                        2, mOg, mGg, 
                                                        sGOR_g_g, sGORgas_g_g
                                                        )
        else:
            mDOg_calc = 0
            mDGg_calc = mGg
    if iuse_dissolved_oil == 0:
        # Maximum mass of dissolved gas (g)
        mDGg = mOg*sGOR_g_g
        mDOg = 0
    else:
        mDGg = mDGg_calc
        mDOg = 0
    # Mass of free gas generated (g)
    if iuse_dissolved_oil == 0:
        
        if mGg < mDGg:
            mFGg = 0
            mDOg = 0
        else:
            mFGg = mGg-mDGg
            mDOg = 0
    else:      
        mFGg = mGg-mDGg_calc + mDOg_calc
        mDOg = mDOg_calc
    # Mass of oil + dissolved gas generated (g)
    if mGg <= mDGg:
        mODGg = mGg + mOg
    else:
        if iuse_dissolved_oil == 0:
            # This is the mass of the liquid
            mODGg = mDGg + mOg
        else:
            # Add the mass of free oil
            mODGg = mDGg + (mOg - mDOg)
    # Mass of generated dissolved gas
    if iuse_dissolved_oil == 0:
        mDGg = mODGg - mOg
    else:
        # Substract the mass of free oil
        mDGg = mODGg - (mOg - mDOg)
    # GOR of generated liquid g/g
    if mOg > 0:
        if iuse_dissolved_oil == 0:
            GOR_liquid = mDGg/mOg
        else:
            if mOg-mDOg > 0:
                GOR_liquid = mDGg/(mOg - mDOg)
            else:
                GOR_liquid = 0
    else:
        GOR_liquid = 0
    # GOR of generated gas g/g
    # assume something close to saturation within the oil window
    GOR_freegas = 5.0
    if iuse_dissolved_oil == 0:
        if mDOg > 0:
            GOR_freegas = mFGg/mDOg
        else:
            GOR_freegas = 5.0
    else:
        if mDOg > 0:
            GOR_freegas = (mFGg - mDOg)/mDOg
        else:
            GOR_freegas = 5.0
    # Mass of unreacted kerogen + mass of coke (g)
    mKURCOKEg = mKURg + mCOKEg
    # Adsorptive capacity (g)
    ACg = mKURCOKEg*adsorption_perc/100
    # Mass of free gas expelled into pore space (g)
    if iuse_dissolved_oil == 0:
        mFGEXPSg = mFGg
    else:
        mFGEXPSg = mFGg + mDOg
    # Mass of free gas expelled into pore space (mg / g OC)
    if mOCIg > 0:
        mFGEXPSmg_gOC = mFGEXPSg*1000/mOCIg
    else:
        mFGEXPSmg_gOC = 0
    # Mass of liquid expelled into pore space (g)
    if mODGg >= ACg:
        mODGEXPSg = mODGg - ACg
    else:
        mODGEXPSg = 0.0
    # Mass of fluid expelled into pore space (mg / g OC)
    if mOCIg > 0:
        mODGEXPSmg_gOC = mODGEXPSg*1000/mOCIg
    else:
        mODGEXPSmg_gOC = 0
    # Mass of liquid adsorped onto organic material
    mODGg_sorped = mODGg - mODGEXPSg
    # Mass of dissolved gas adsorped onto organic material
    mDGg_sorped = mODGg_sorped*GOR_liquid
    # Mass of oil adsorped onto organic material
    mOg_sorped = mODGg_sorped - mDGg_sorped
    # Gas FVF (bg) rbbl/scf
    if fpress_psi == 0.0:
        fpress_psi = 0.001
    GFVF = 0.00502*zfactor*(TF + 460.0)/fpress_psi
    # Reservoir gas density g/cc
    # This could be improved by accounting for dissolved heavies.
    # One approach to consider is as follows:
    #   rho = (rho_rgas_kg_m3 + rho_rgas_kg_m3*OGR_gas)
    rho_rgas_g_cc = 0.21870617*(0.001/GFVF)*sgrav_gas
    # Liquid FVF (bo) rbbl/stb
    LFVF = (
              0.9759 
            + 0.00012*(
                        sGOR_scf_bbl*
                                math.sqrt(sgrav_gas/sgrav_oil) + 1.25*TF)**1.2
           )
    # Reservoir fluid density g/cc
    rho_rfluid_g_cc = (sgrav_oil + 0.0002179*sGOR_scf_bbl*sgrav_gas)/LFVF
    # Volume of free gas expelled into pore space (m3)
    vFGEXPSm3 = (mFGEXPSg/rho_rgas_g_cc)/1000000
    # Volume of fluid expelled into pore space (m3)
    vODGEXPSm3 = (mODGEXPSg/rho_rfluid_g_cc)/1000000
    # Volume of gas expelled from source rock (m3)
    if vFGEXPSm3 + vODGEXPSm3 <= por_cap_m3:
        vGEXSRm3 = 0.0
    else:
        vGEXSRm3 = (
                      vFGEXPSm3 
                    - (vFGEXPSm3/(vFGEXPSm3 + vODGEXPSm3))*por_cap_m3
                   )
    mGEXSRg = vGEXSRm3*rho_rgas_g_cc*1000*1000
    mRESIDUAL_FG_PORESg = mFGEXPSg - mGEXSRg
    # Volume of fluid expelled from source rock (m3)
    if vFGEXPSm3 + vODGEXPSm3 <= por_cap_m3:
        vODGEXSRm3 = 0
    else:
        vODGEXSRm3 = (
                          vODGEXPSm3 
                        - (vODGEXPSm3/(vFGEXPSm3 + vODGEXPSm3))*por_cap_m3
                    )
    mODGEXSRg = vODGEXSRm3*rho_rfluid_g_cc*1000*1000
    mRESIDUAL_LIQUID_PORESg = mODGEXPSg - mODGEXSRg
    mRESIDUAL_DGAS_PORESg = mRESIDUAL_LIQUID_PORESg*GOR_liquid
    mRESIDUAL_OIL_PORESg = mRESIDUAL_LIQUID_PORESg - mRESIDUAL_DGAS_PORESg
    mRESIDUAL_OILg = mOg_sorped + mRESIDUAL_OIL_PORESg
    mRESDUAL_GASg = mDGg_sorped + mRESIDUAL_DGAS_PORESg + mRESIDUAL_FG_PORESg
    mRESDUAL_COKEg = mCOKEg
    mRESIDUAL_OIL_mg_gTOC = mRESIDUAL_OILg*1000.0/mOCIg
    mRESIDUAL_LIQUIDg = mODGEXPSg - mODGEXSRg + mODGg_sorped
    mRESIDUAL_LIQUID_mg_gTOC = mRESIDUAL_LIQUIDg*1000.0/mOCIg    
    return (
            mOCIg, mHCg, mHCmg_gOC, mGg, mOg, mCOKEg, 
            mFGEXPSg, mFGEXPSmg_gOC, mODGEXPSg, mODGEXPSmg_gOC, 
            vGEXSRm3, vODGEXSRm3, mRESDUAL_GASg, mRESIDUAL_OILg, 
            mRESDUAL_COKEg, mRESIDUAL_OIL_mg_gTOC, vFGEXPSm3,
            vODGEXPSm3, rho_rgas_g_cc, rho_rfluid_g_cc, GFVF, LFVF,
            mRESIDUAL_LIQUID_mg_gTOC, GOR_liquid, GOR_freegas, zfactor
            )


@jit(nopython=True, cache=True)
def calc_TR(
                imass_calc_type, ntimes, To, dt_Myr, 
                hr_Myr, Kat0C, Rgas, A, nEa, Ea_array,
                xi_array, xi_initial, TR_array, xi_final
):
    for i in range(ntimes):
        tMyr = float(i)*dt_Myr
        TC = To + tMyr * hr_Myr
        TK = TC+Kat0C
        if i > 0:
            tMyr_prev = float(i-1)*dt_Myr
            TC_prev = To + tMyr_prev * hr_Myr
            TK_prev = TC_prev+Kat0C
        else:
            TC_prev = TC
            TK_prev = TK
        for j in range(nEa):
            Ea = Ea_array[j]
            if i > 0:
                xi_array[i][j] = (
                                  xi_array[i-1][j]*
                                    (
                                     2 - dt_Myr*A*math.exp(-Ea/Rgas/TK_prev)
                                    )/(
                                        2 + dt_Myr*A*math.exp(-Ea/Rgas/TK_prev)
                                      )
                                 )
            else:
                xi_array[i][j] = xi_initial[j]
        sumit = 0.0
        for j in range(nEa):
            sumit = sumit + xi_array[i][j]
        TR = 1.0 - sumit
        TR_array[i] = TR
    for j in range(nEa):
        xi_final[j] = xi_array[ntimes - 1][j]
    TR_final = TR_array[ntimes - 1]
    tMyr_final = float(ntimes - 1)*dt_Myr
    TC_final = To + tMyr * hr_Myr
    TF_final = TC_final*9.0/5.0 + 32.0
    return TF_final, TR_final, tMyr_final
    

@jit(nopython=True, cache=True)    
def calc_TR_and_mass_gen_event(
                                imass_calc_type, ntimes, xi_array, TR_array,
                                xi_final, Ea_array, A, xi_initial, HI, TOC, 
                                rho_grain, inert_frac, adsorption_perc,
                                por_frac, fpress_psi, To, Tf, event_duration, 
                                tSm, aSkm2, Oil_API, HI_lup,fkp_lup, fp_lup, 
                                ro_lup, rp_lup, por_frac_last,
                                mGg_ini, mOg_ini, mCg_ini, sgrav_gas, 
                                por_threshold, igas_frac_type,
                                ipolar_frac_type, TRmin, TRmax, 
                                gas_frac_adj_TRmin, gas_frac_adj_TRmax, 
                                idebug_out
):
    nEa = 40
    Rgas = 0.00198720425864083 # kcal/K/mol
    Kat0C = 273.15
    if event_duration > 0:
        # linear heating rate C/Myr
        hr_Myr = (Tf-To)/event_duration
    else:
        hr_Myr = 0
    hr_Myr = round(hr_Myr,4)
    if abs(hr_Myr) > 0.0:
        dt_Myr = 2.0/hr_Myr*1.5
    else:
        dt_Myr = 0.0
    tMyr_end_ini = (ntimes-1)*dt_Myr
    if (tMyr_end_ini < event_duration) or (tMyr_end_ini > event_duration):
        if abs(tMyr_end_ini) > 0:
            dt_Myr = event_duration/tMyr_end_ini*dt_Myr
        else:
            dt_Myr = 0.0
        tMyr_end_ini = float(ntimes-1)*dt_Myr
    (
     TF_final, 
     TR_final, 
     tMyr_final
    ) = calc_TR(
                imass_calc_type, ntimes, To, dt_Myr, hr_Myr,
                Kat0C, Rgas, A, nEa, Ea_array, xi_array, 
                xi_initial, TR_array, xi_final
                )
    (
         mOCIg, mHCg, mHCmg_gOC, 
         mGg, mOg, mCOKEg, mFGEXPSg, 
         mFGEXPSmg_gOC, mODGEXPSg,
         mODGEXPSmg_gOC, vGEXSRm3, 
         vODGEXSRm3, mRESDUAL_GASg, 
         mRESIDUAL_OILg, mRESDUAL_COKEg,
         mRESIDUAL_OIL_mg_gTOC, 
         vFGEXPSm3, vODGEXPSm3, 
         rho_rgas_g_cc, rho_rfluid_g_cc, 
         GFVF, LFVF, mRESIDUAL_LIQUID_mg_gTOC, 
         GOR_liquid, GOR_freegas, zfactor
    ) = mass_generation_calc(
                                imass_calc_type, HI, TOC, rho_grain, 
                                inert_frac, adsorption_perc, por_frac, 
                                fpress_psi, tSm, aSkm2, Oil_API, sgrav_gas, 
                                TR_final, TF_final, HI_lup,fkp_lup, fp_lup, 
                                ro_lup, rp_lup, por_frac_last, mGg_ini, 
                                mOg_ini, mCg_ini, por_threshold, 
                                igas_frac_type, ipolar_frac_type, 
                                TRmin, TRmax, gas_frac_adj_TRmin,
                                gas_frac_adj_TRmax, idebug_out
                        )
    return (
                tMyr_final, TF_final, TR_final, 
                mOCIg, mHCg, mHCmg_gOC, mGg, mOg,
                mCOKEg, mFGEXPSg, mFGEXPSmg_gOC, 
                mODGEXPSg, mODGEXPSmg_gOC, vGEXSRm3,
                vODGEXSRm3, mRESDUAL_GASg, mRESIDUAL_OILg, 
                mRESDUAL_COKEg, mRESIDUAL_OIL_mg_gTOC,
                vFGEXPSm3, vODGEXPSm3, rho_rgas_g_cc, 
                rho_rfluid_g_cc, GFVF, LFVF,
                mRESIDUAL_LIQUID_mg_gTOC, GOR_liquid, 
                GOR_freegas, zfactor
        )


@jit(nopython=True, cache=True)
def calculate_TR_and_mass_history_loop(
                    event_ID, icount_steps, event_duration, 
                    nEa, nx, ny, inode, jnode,
                    itype3D, imass_calc_type, AOI_np, cell_area_km2,
                    adsorption_perc, inert_frac, TRmin, TRmax, 
                    TR_limit, TR_cutoff_crack, TR_cutoff_highLOM,
                    gas_frac_adj_TRmin, gas_frac_adj_TRmax,
                    ireach_limit_xy, por_frac_prev_xy, 
                    fpress_psi_prev_xy, 
                    Tf_prev_xy, xi_final_xy_events,
                    To_xy, Tf_xy, HI_xy,
                    z_top_subsea_xy, phi_o_xy, decay_depth_xy,
                    maxfb_xy, maxfb_last_xy, rho_grain_xy,
                    TR_xy, mHCmg_gOC_xy, mODGEXPSmg_gOC_xy, 
                    mFGEXPSmg_gOC_xy, mRESDUAL_GASg_xy, 
                    mRESIDUAL_OILg_xy, mRESDUAL_COKEg_xy,
                    mRESIDUAL_LIQUID_mg_gTOC_xy, vGEXSRtcf_xy,
                    vODGEXSRgob_xy, rho_rgas_g_cc_xy, 
                    rho_rfluid_g_cc_xy, GFVFrcf_scf_xy, LFVF_xy, 
                    GOR_liquid_xy, mGg_xy, mOg_xy, mCOKEg_xy, mOCIg_xy, 
                    mFGEXPSmg_gOC_sc_xy, vGEXSRtcf_sc_xy, 
                    rho_rgas_g_cc_sc_xy, GFVF_rcf_scf_sc_xy,
                    mHCmg_gOC_sc_xy, mGg_sc_xy, mOCIg_sc_xy,
                    iomt_xy, igas_scenario_xy, ipolar_scenario_xy,
                    ikinetics_scenario_xy, hi_primary_xy, toc_xy,
                    src_thick_xy, oil_api_xy, gas_grav_xy,
                    por_thresh_xy,
                    A_vk_early_all, f_vk_early_all, 
                    A_vk_norm_all, f_vk_norm_all, 
                    A_vk_late_all, f_vk_late_all,
                    A_OilCrack, f_OilCrack,
                    A_HighLOM, f_HighLOM, 
                    f_HighLOM_type3
):
    oilapi_avg = 0.0
    gasgrav_avg = 0.0
    icount_vals = 0
    for i in numba.prange(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if itype3D == 0:
                if i == inode and j == jnode:
                    AOI_flag = AOI_flag
                else:
                    AOI_flag = 0
            if AOI_flag == 1:
                if icount_steps > 0:
                    To = To_xy[i][j]
                else:
                    To = 0.0                                    
                Tf = Tf_xy[i][j]
                # source specific parameters
                #"Type_II" # Type_I, Type_II, 
                # Type_III or Type_IIS
                iomt = iomt_xy[i][j]
                TOC = toc_xy[i][j]
                if imass_calc_type == 0: 
                    # Primary generation
                    HI = hi_primary_xy[i][j]
                    mGg_ini = 0
                    mOg_ini = 0
                    mCg_ini = 0
                elif imass_calc_type == 1: 
                    # Secondary cracking
                    mGg_ini = 0
                    mOg_ini = 0
                    mCg_ini = 0                                    
                    HI = HI_xy[i][j]
                    inert_frac = 0
                elif imass_calc_type == 2: 
                    # High LOM Gas
                    mGg_ini = 0
                    mOg_ini = 0
                    mCg_ini = 0
                    HI_main = hi_primary_xy[i][j]                                  
                    if iomt == 3:
                        HI = 15
                    else:
                        HI = 24
                    TOC = TOC*(1200-HI_main)*0.83/1000
                    inert_frac = 0
                # We should only perform this calculation 
                # where TOC > 0
                if TOC > 0:
                    # "normal" # early, normal, late
                    isktype = ikinetics_scenario_xy[i][j]
                    tSm = src_thick_xy[i][j]
                    aSkm2 = cell_area_km2
                    Oil_API = oil_api_xy[i][j]
                    gas_grav = gas_grav_xy[i][j]
                    por_threshold = por_thresh_xy[i][j]
                    igas_frac_type = igas_scenario_xy[i][j]
                    ipolar_frac_type = ipolar_scenario_xy[i][j]
                    oilapi_avg = oilapi_avg + Oil_API
                    gasgrav_avg = gasgrav_avg + gas_grav
                    icount_vals = icount_vals + 1
                    if imass_calc_type == 0: 
                        # Primary generation
                        (
                            A, 
                            xi_initial_tmp, 
                            Ea_array
                        ) = define_kinetics_np(
                                                iomt, isktype, 
                                                A_vk_early_all, f_vk_early_all, 
                                                A_vk_norm_all, f_vk_norm_all, 
                                                A_vk_late_all, f_vk_late_all,
                                                A_OilCrack, f_OilCrack,
                                                A_HighLOM, f_HighLOM, 
                                                f_HighLOM_type3, nEa
                                            )
                    elif imass_calc_type == 1: 
                        # Secondary cracking
                        (
                            A, 
                            xi_initial_tmp, 
                            Ea_array
                        ) = define_kinetics_np(
                                                iomt, 2, 
                                                A_vk_early_all, f_vk_early_all, 
                                                A_vk_norm_all, f_vk_norm_all, 
                                                A_vk_late_all, f_vk_late_all,
                                                A_OilCrack, f_OilCrack,
                                                A_HighLOM, f_HighLOM, 
                                                f_HighLOM_type3, nEa
                                            )
                    elif imass_calc_type == 2: 
                        # High LOM Gas
                        (
                            A, 
                            xi_initial_tmp, 
                            Ea_array
                        ) = define_kinetics_np(
                                                iomt, 3, 
                                                A_vk_early_all, f_vk_early_all, 
                                                A_vk_norm_all, f_vk_norm_all, 
                                                A_vk_late_all, f_vk_late_all,
                                                A_OilCrack, f_OilCrack,
                                                A_HighLOM, f_HighLOM, 
                                                f_HighLOM_type3, nEa
                                            )
                    # number of calculation time steps per event
                    ntimes = 101
                    # subsea depth in meters
                    z_top_subsea = z_top_subsea_xy[i][j]
                    phi_o = phi_o_xy[i][j]/100.0
                    # Porosity Decay Depth (km) 
                    # converted to 1/m
                    c = 1.0/(decay_depth_xy[i][j]*1000)
                    # maximum forward burial in meters     
                    maxfb = maxfb_xy[i][j]                               
                    maxfb_last = maxfb_last_xy[i][j]
                    # W/m/K    
                    rho_grain = rho_grain_xy[i][j]                               
                    # Event specific parameters
                    por_frac = phi_o*math.exp(-maxfb*c)                                    
                    (
                        por_frac_last
                    ) = phi_o*math.exp(-maxfb_last*c)                                    
                    fpress_psi = z_top_subsea/0.3048*0.43                                
                    if ireach_limit_xy[i][j] == 1:                                        
                        por_frac = por_frac_prev_xy[i][j]
                        fpress_psi = fpress_psi_prev_xy[i][j]
                        To = Tf_prev_xy[i][j]
                        Tf = Tf_prev_xy[i][j]
                    HI_lup = np.zeros((5))
                    fkp_lup = np.zeros((5))
                    fp_lup = np.zeros((5))
                    ro_lup = np.zeros((5))
                    rp_lup = np.zeros((5))
                    xi_array = np.zeros((ntimes, nEa))
                    TR_array = np.zeros((ntimes))
                    if icount_steps == 0:
                        xi_initial = np.copy(xi_initial_tmp)
                        xi_final = np.copy(xi_initial_tmp)
                    else:
                        (
                            xi_initial
                        ) = np.copy(xi_final_xy_events[event_ID-1][i][j])
                        xi_final = np.zeros((nEa))
                    
                    (
                        tMyr_final, TF_final, TR_final, mOCIg, 
                        mHCg, mHCmg_gOC, mGg, mOg, mCOKEg, 
                        mFGEXPSg, mFGEXPSmg_gOC, 
                        mODGEXPSg, mODGEXPSmg_gOC, 
                        vGEXSRm3, vODGEXSRm3, mRESDUAL_GASg, 
                        mRESIDUAL_OILg, mRESDUAL_COKEg, 
                        mRESIDUAL_OIL_mg_gTOC,
                        vFGEXPSm3, vODGEXPSm3, rho_rgas_g_cc, 
                        rho_rfluid_g_cc, GFVF, LFVF,
                        mRESIDUAL_LIQUID_mg_gTOC, GOR_liquid, 
                        GOR_freegas, zfactor
                    ) = calc_TR_and_mass_gen_event(
                            imass_calc_type, ntimes, xi_array, 
                            TR_array, xi_final, Ea_array, A, 
                            xi_initial, HI, TOC, rho_grain,
                            inert_frac, adsorption_perc, 
                            por_frac, fpress_psi, To, Tf, 
                            event_duration, tSm, aSkm2, 
                            Oil_API, HI_lup, fkp_lup, fp_lup, 
                            ro_lup, rp_lup, por_frac_last,
                            mGg_ini, mOg_ini, mCg_ini, 
                            gas_grav, por_threshold, 
                            igas_frac_type, ipolar_frac_type, 
                            TRmin, TRmax, gas_frac_adj_TRmin,
                            gas_frac_adj_TRmax, 0
                    )
                    xi_final_xy_events[event_ID][i][j] = np.copy(xi_final)
                    if imass_calc_type == 0: 
                        # Primary generation
                        TR_xy[i][j] = TR_final
                        mHCmg_gOC_xy[i][j] = mHCmg_gOC
                        mODGEXPSmg_gOC_xy[i][j] = mODGEXPSmg_gOC
                        mFGEXPSmg_gOC_xy[i][j] = mFGEXPSmg_gOC
                        mRESDUAL_GASg_xy[i][j] = mRESDUAL_GASg
                        mRESIDUAL_OILg_xy[i][j] = mRESIDUAL_OILg
                        mRESDUAL_COKEg_xy[i][j] = mRESDUAL_COKEg
                        mRESIDUAL_LIQUID_mg_gTOC_xy[i][j] = (
                                                    mRESIDUAL_LIQUID_mg_gTOC)
                        vGEXSRtcf_xy[i][j] = (
                                vGEXSRm3/(GFVF*5.6145833333)*35.3147/1e12
                                )
                        vODGEXSRgob_xy[i][j] = (vODGEXSRm3/LFVF*6.28981/1e9)
                        rho_rgas_g_cc_xy[i][j] = rho_rgas_g_cc
                        rho_rfluid_g_cc_xy[i][j] = rho_rfluid_g_cc
                        # Conversion to rcf/scf
                        GFVFrcf_scf_xy[i][j] = (GFVF*5.6145833333)
                        LFVF_xy[i][j] = LFVF
                        GOR_liquid_xy[i][j] = GOR_liquid
                        #if itype3D == 0:
                        mGg_xy[i][j] = mGg
                        mOg_xy[i][j] = mOg
                        mCOKEg_xy[i][j] = mCOKEg
                        mOCIg_xy[i][j] = mOCIg
                    elif imass_calc_type == 1: 
                        # Secondary cracking
                        if TR_final > TR_cutoff_crack:
                            mFGEXPSmg_gOC_sc_xy[i][j] = mFGEXPSmg_gOC
                            vGEXSRtcf_sc_xy[i][j] = (
                                    vGEXSRm3/(GFVF*5.6145833333)*35.3147/1e12
                                    )
                        else:
                            mFGEXPSmg_gOC_sc_xy[i][j] = 0.0
                            vGEXSRtcf_sc_xy[i][j] = 0.0
                        rho_rgas_g_cc_sc_xy[i][j] = rho_rgas_g_cc
                        GFVF_rcf_scf_sc_xy[i][j] = GFVF*5.6145833333
                        #if itype3D == 0:
                        if TR_final > TR_cutoff_crack:
                            mHCmg_gOC_sc_xy[i][j] = mHCmg_gOC
                            mGg_sc_xy[i][j] = (mGg - mGg_ini)
                        else:
                            mHCmg_gOC_sc_xy[i][j] = 0
                            mGg_sc_xy[i][j] = 0
                        mOCIg_sc_xy[i][j] = mOCIg
                    elif imass_calc_type == 2: 
                        # High LOM Gas
                        if TR_final > TR_cutoff_highLOM:
                            vGEXSRtcf_sc_xy[i][j] = (
                                                        vGEXSRtcf_sc_xy[i][j] 
                                                        + mFGEXPSmg_gOC
                                                    )
                            vGEXSRtcf_sc_xy[i][j] = (
                                      vGEXSRtcf_sc_xy[i][j] 
                                    + vGEXSRm3/(GFVF*5.6145833333)*35.3147/1e12
                                    )
                    if TR_final >= TR_limit:
                        ireach_limit_xy[i][j] = 1
                        fpress_psi_prev_xy[i][j] = fpress_psi
                        por_frac_prev_xy[i][j] = por_frac
                        Tf_prev_xy[i][j] = Tf                       
    oilapi_avg = oilapi_avg/float(icount_vals)
    gasgrav_avg = gasgrav_avg/float(icount_vals)
    return oilapi_avg, gasgrav_avg


def unpack_src_param_maps(src_params_xy, nx, ny):
    iomt_xy = np.zeros((nx, ny), dtype=int)
    igas_scenario_xy = np.zeros((nx, ny), dtype=int)
    ipolar_scenario_xy = np.zeros((nx, ny), dtype=int)
    ikinetics_scenario_xy = np.zeros((nx, ny), dtype=int)
    hi_primary_xy = np.zeros((nx, ny))
    toc_xy = np.zeros((nx, ny))
    src_thick_xy = np.zeros((nx, ny))
    oil_api_xy = np.zeros((nx, ny))
    gas_grav_xy = np.zeros((nx, ny))
    por_thresh_xy = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            src_params = src_params_xy[i][j]
            hi_primary_xy[i][j] = src_params[1]
            toc_xy[i][j] = src_params[2]
            src_thick_xy[i][j] = src_params[4]
            oil_api_xy[i][j] = src_params[6]
            gas_grav_xy[i][j] = src_params[7]
            por_thresh_xy[i][j] = src_params[8]
            omt_str = src_params[0]
            kinetics_scenario_str = src_params[3]
            gas_scenario_str = src_params[9]
            polar_scenario_str = src_params[10]
            if omt_str == "Type_I":
                iomt_xy[i][j] = 1
            elif omt_str == "Type_II":
                iomt_xy[i][j] = 2
            elif omt_str == "Type_III":
                iomt_xy[i][j] = 3
            elif omt_str == "Type_IIS":
                iomt_xy[i][j] = 4
            if gas_scenario_str == "minimum":
                igas_scenario_xy[i][j] = -1
            elif gas_scenario_str == "maximum":
                igas_scenario_xy[i][j] = 1
            if polar_scenario_str == "minimum":
                ipolar_scenario_xy[i][j] = -1
            elif polar_scenario_str == "maximum":
                ipolar_scenario_xy[i][j] = 1
            if kinetics_scenario_str == "early":
                ikinetics_scenario_xy[i][j] = -1
            elif kinetics_scenario_str == "late":
                ikinetics_scenario_xy[i][j] = 1
    return (
            iomt_xy, igas_scenario_xy, ipolar_scenario_xy,
            ikinetics_scenario_xy, hi_primary_xy, toc_xy,
            src_thick_xy, oil_api_xy, gas_grav_xy,
            por_thresh_xy
            )
                

def calculate_TR_and_mass_history(srckinetics, model, imass_calc_type, ioutput
):
    nx = model.nx
    ny = model.ny
    AOI_np = model.AOI_np
    cell_area_km2 = model.cell_area_km2
    itype3D = model.itype3D
    inode = model.inode
    jnode = model.jnode
    
    keys = list(model.event_dict_bs.keys())
    OILAPI_list = []
    GASGRAV_list = []
    TR_limit = 1.0 - 1e-2
    TR_cutoff_crack = 0.05
    TR_cutoff_highLOM = 0.05
    ntops = len(model.tops_list_bs)
    nevents = ntops
    
    # Unpack source kinetics and expulsion parameters
    nEa = srckinetics.nEa
    A_vk_early_all = srckinetics.A_Early_all
    f_vk_early_all = srckinetics.f_Early_all 
    A_vk_norm_all = srckinetics.A_Normal_all
    f_vk_norm_all = srckinetics.f_Normal_all
    A_vk_late_all = srckinetics.A_Late_all
    f_vk_late_all = srckinetics.f_Late_all
    A_OilCrack = srckinetics.A_OilCrack
    f_OilCrack = srckinetics.f_OilCrack_all
    A_HighLOM = srckinetics.A_HighLOM_TII
    f_HighLOM = srckinetics.f_HighLOM_TII_all
    f_HighLOM_type3 = srckinetics.f_HighLOM_TIII_all
    inert_frac = srckinetics.inert_frac
    adsorption_perc = srckinetics.adsorption_perc
    src_top_names = srckinetics.src_top_names
    TRmin = srckinetics.TRmin
    TRmax = srckinetics.TRmax
    gas_frac_adj_TRmin = srckinetics.gas_frac_adj_TRmin
    gas_frac_adj_TRmax = srckinetics.gas_frac_adj_TRmax
    
    for mm in range(ntops):
        jj = ntops - 1 - mm
        name = model.tops_list_bs[jj][6]
        # Only perform this calculation for sources
        if name in src_top_names:
            xi_final_xy_events = np.zeros((nevents, nx, ny, nEa))
            fpress_psi_prev_xy = np.zeros((nx, ny))
            por_frac_prev_xy = np.ones((nx, ny))*0.5
            Tf_prev_xy = np.zeros((nx, ny))
            ireach_limit_xy = np.zeros((nx, ny), dtype=int)         
            event_ID_last = keys[len(keys)-1]
            icount_steps = 0
            # Loop over events from old to young
            for event_ID, key in enumerate(keys):
                # Skip the oldest node since this is "basement rock"
                if event_ID > 0:
                    itop_event = model.event_dict_bs[event_ID][2]
                    age_event = model.event_dict_bs[event_ID][0]
                    event_ID_prev = event_ID - 1
                    age_event_prev = model.event_dict_bs[event_ID_prev][0]
                    if jj <= itop_event:
                        # We only consider tops 
                        # that have been deposited
                        event_duration = age_event_prev - age_event
                        event_index = model.tops_list_bs[jj][14][event_ID]
                        event_index_last = model.tops_list_bs[jj][14]\
                                                                [event_ID_last]
                        if icount_steps > 0:
                            (
                                event_index_prev
                            ) = model.tops_list_bs[jj][14][event_ID_prev]
                        if icount_steps > 0:
                            (
                                To_xy
                            ) = np.copy(
                                  model.tops_list_bs[jj][38][event_index_prev])
                        else:
                            To_xy = np.zeros((nx, ny))
                        Tf_xy = np.copy(model.tops_list_bs[jj][38][event_index])
                        src_params_xy = np.copy(model.tops_list_bs[jj][47])
                        # Need to unpack the src_params_xy list and make
                        # numpy arrays
                        (
                            iomt_xy, igas_scenario_xy, ipolar_scenario_xy,
                            ikinetics_scenario_xy, hi_primary_xy, toc_xy,
                            src_thick_xy, oil_api_xy, gas_grav_xy,
                            por_thresh_xy
                        ) = unpack_src_param_maps(src_params_xy, nx, ny)
                        HI_xy = np.copy(
                                  model.tops_list_bs[jj][51][event_index_last])
                        phi_o_xy = np.copy(model.tops_list_bs[jj][33])
                        decay_depth_xy = np.copy(model.tops_list_bs[jj][34])
                        maxfb_xy = np.copy(
                                       model.tops_list_bs[jj][46][event_index])
                        rho_grain_xy = np.copy(model.tops_list_bs[jj][32])
                        (
                            maxfb_last_xy
                        ) = np.copy(
                                  model.tops_list_bs[jj][46][event_index_last])
                        (
                            z_top_subsea_xy
                        ) = np.copy(model.tops_list_bs[jj][1][event_index])
                        # Define numpy arrays that will be updated
                        if imass_calc_type == 0:
                            # Primary generation
                            TR_xy = np.copy(
                                       model.tops_list_bs[jj][41][event_index])
                            (
                                mHCmg_gOC_xy
                            ) = np.copy(
                                       model.tops_list_bs[jj][42][event_index])
                            (
                                mODGEXPSmg_gOC_xy
                            ) = np.copy(
                                       model.tops_list_bs[jj][43][event_index])
                            (
                                mFGEXPSmg_gOC_xy
                            ) = np.copy(
                                       model.tops_list_bs[jj][44][event_index])
                            (
                                mRESDUAL_GASg_xy
                             ) = np.copy(
                                       model.tops_list_bs[jj][48][event_index])
                            (
                                mRESIDUAL_OILg_xy
                             ) = np.copy(
                                     model.tops_list_bs[jj][49][event_index])
                            (
                                mRESDUAL_COKEg_xy
                             ) = np.copy(
                                     model.tops_list_bs[jj][50][event_index])
                            (
                                mRESIDUAL_LIQUID_mg_gTOC_xy
                            ) = np.copy(
                                    model.tops_list_bs[jj][51][event_index])
                            (
                                vGEXSRtcf_xy
                            ) = np.copy(
                                    model.tops_list_bs[jj][52][event_index])
                            (
                                vODGEXSRgob_xy
                            ) = np.copy(
                                    model.tops_list_bs[jj][53][event_index])
                            (
                                rho_rgas_g_cc_xy
                            ) = np.copy(
                                    model.tops_list_bs[jj][54][event_index])
                            (
                                rho_rfluid_g_cc_xy
                            ) = np.copy(
                                    model.tops_list_bs[jj][55][event_index])
                            # Conversion to rcf/scf
                            (
                                GFVFrcf_scf_xy
                            ) = np.copy(
                                    model.tops_list_bs[jj][56][event_index])
                            (
                                LFVF_xy
                            ) = np.copy(
                                    model.tops_list_bs[jj][57][event_index])
                            (
                                GOR_liquid_xy
                            ) = np.copy(
                                    model.tops_list_bs[jj][70][event_index])
                            #if itype3D == 0:
                            (
                                mGg_xy
                            ) = np.copy(
                                   model.tops_list_bs[jj][63][event_index])
                            (
                                mOg_xy
                            ) = np.copy(
                                   model.tops_list_bs[jj][64][event_index]) 
                            (
                                mCOKEg_xy
                            ) = np.copy(
                                   model.tops_list_bs[jj][65][event_index])
                            (
                                mOCIg_xy
                            ) = np.copy(
                                   model.tops_list_bs[jj][68][event_index])
                            #else:
                            #    mGg_xy = np.zeros((nx, ny))
                            #    mOg_xy = np.zeros((nx, ny))
                            #    mCOKEg_xy = np.zeros((nx, ny))
                            #    mOCIg_xy = np.zeros((nx, ny))        
                        else:
                            # Primary generation
                            TR_xy = np.zeros((nx, ny))
                            mHCmg_gOC_xy = np.zeros((nx, ny))
                            mODGEXPSmg_gOC_xy = np.zeros((nx, ny))
                            mFGEXPSmg_gOC_xy = np.zeros((nx, ny))
                            mRESDUAL_GASg_xy = np.zeros((nx, ny))
                            mRESIDUAL_OILg_xy = np.zeros((nx, ny))
                            mRESDUAL_COKEg_xy = np.zeros((nx, ny))
                            mRESIDUAL_LIQUID_mg_gTOC_xy = np.zeros((nx, ny))
                            vGEXSRtcf_xy = np.zeros((nx, ny))
                            vODGEXSRgob_xy = np.zeros((nx, ny))
                            rho_rgas_g_cc_xy = np.zeros((nx, ny))
                            rho_rfluid_g_cc_xy = np.zeros((nx, ny))
                            # Conversion to rcf/scf
                            GFVFrcf_scf_xy = np.zeros((nx, ny))
                            LFVF_xy = np.zeros((nx, ny))
                            GOR_liquid_xy = np.zeros((nx, ny))
                            mGg_xy = np.zeros((nx, ny))
                            mOg_xy = np.zeros((nx, ny))
                            mCOKEg_xy = np.zeros((nx, ny))
                            mOCIg_xy = np.zeros((nx, ny))
                        if imass_calc_type == 1: 
                            # Secondary cracking
                            (
                                mFGEXPSmg_gOC_sc_xy
                            ) = np.copy(
                                    model.tops_list_bs[jj][58][event_index])
                            (
                                vGEXSRtcf_sc_xy
                            ) = np.copy(
                                    model.tops_list_bs[jj][59][event_index])
                            (
                                rho_rgas_g_cc_sc_xy
                            ) = np.copy(
                                    model.tops_list_bs[jj][60][event_index])
                            (
                                GFVF_rcf_scf_sc_xy
                            ) = np.copy(
                                    model.tops_list_bs[jj][61][event_index])
                            #if itype3D == 0:
                            (
                                mHCmg_gOC_sc_xy
                            ) = np.copy(
                                   model.tops_list_bs[jj][66][event_index])
                            (
                                mGg_sc_xy
                            ) = np.copy(
                                   model.tops_list_bs[jj][67][event_index])
                            (
                                mOCIg_sc_xy
                            ) = np.copy(
                                   model.tops_list_bs[jj][69][event_index])
                            #else:
                            #    mHCmg_gOC_sc_xy = np.zeros((nx, ny))
                            #    mGg_sc_xy = np.zeros((nx, ny))
                            #    mOCIg_sc_xy = np.zeros((nx, ny))
                        else:
                            # Secondary cracking
                            mFGEXPSmg_gOC_sc_xy = np.zeros((nx, ny))
                            vGEXSRtcf_sc_xy = np.zeros((nx, ny))
                            rho_rgas_g_cc_sc_xy = np.zeros((nx, ny))
                            GFVF_rcf_scf_sc_xy = np.zeros((nx, ny))
                            mHCmg_gOC_sc_xy = np.zeros((nx, ny))
                            mGg_sc_xy = np.zeros((nx, ny))
                            mOCIg_sc_xy = np.zeros((nx, ny))
                        (
                            oilapi_avg, 
                            gasgrav_avg
                        ) = calculate_TR_and_mass_history_loop(
                            event_ID, icount_steps, event_duration, 
                            nEa, nx, ny, inode, jnode,
                            itype3D, imass_calc_type, AOI_np, cell_area_km2,
                            adsorption_perc, inert_frac, TRmin, TRmax, 
                            TR_limit, TR_cutoff_crack, TR_cutoff_highLOM,
                            gas_frac_adj_TRmin, gas_frac_adj_TRmax,
                            ireach_limit_xy, por_frac_prev_xy, 
                            fpress_psi_prev_xy, 
                            Tf_prev_xy, xi_final_xy_events,
                            To_xy, Tf_xy, HI_xy,
                            z_top_subsea_xy, phi_o_xy, decay_depth_xy,
                            maxfb_xy, maxfb_last_xy, rho_grain_xy,
                            TR_xy, mHCmg_gOC_xy, mODGEXPSmg_gOC_xy, 
                            mFGEXPSmg_gOC_xy, mRESDUAL_GASg_xy, 
                            mRESIDUAL_OILg_xy, mRESDUAL_COKEg_xy,
                            mRESIDUAL_LIQUID_mg_gTOC_xy, vGEXSRtcf_xy,
                            vODGEXSRgob_xy, rho_rgas_g_cc_xy, 
                            rho_rfluid_g_cc_xy, GFVFrcf_scf_xy, LFVF_xy, 
                            GOR_liquid_xy, mGg_xy, mOg_xy, mCOKEg_xy, mOCIg_xy, 
                            mFGEXPSmg_gOC_sc_xy, vGEXSRtcf_sc_xy, 
                            rho_rgas_g_cc_sc_xy, GFVF_rcf_scf_sc_xy,
                            mHCmg_gOC_sc_xy, mGg_sc_xy, mOCIg_sc_xy,
                            iomt_xy, igas_scenario_xy, ipolar_scenario_xy,
                            ikinetics_scenario_xy, hi_primary_xy, toc_xy,
                            src_thick_xy, oil_api_xy, gas_grav_xy,
                            por_thresh_xy, 
                            A_vk_early_all, f_vk_early_all, 
                            A_vk_norm_all, f_vk_norm_all, 
                            A_vk_late_all, f_vk_late_all,
                            A_OilCrack, f_OilCrack,
                            A_HighLOM, f_HighLOM, 
                            f_HighLOM_type3
                        )
                        if imass_calc_type == 0:
                            # Primary generation
                            model.tops_list_bs[jj][41][event_index] = np.copy(
                                                                         TR_xy)
                            (
                                model.tops_list_bs[jj][42][event_index]
                            ) = np.copy(mHCmg_gOC_xy)
                            (
                                model.tops_list_bs[jj][43][event_index]
                            ) = np.copy(mODGEXPSmg_gOC_xy)
                            (
                                model.tops_list_bs[jj][44][event_index]
                            ) = np.copy(mFGEXPSmg_gOC_xy)
                            (
                                model.tops_list_bs[jj][48][event_index]
                             ) = np.copy(mRESDUAL_GASg_xy)
                            (
                                model.tops_list_bs[jj][49][event_index]
                             ) = np.copy(mRESIDUAL_OILg_xy)
                            (
                                model.tops_list_bs[jj][50][event_index]
                             ) = np.copy(mRESDUAL_COKEg_xy)
                            (
                                model.tops_list_bs[jj][51][event_index]
                            ) = np.copy(mRESIDUAL_LIQUID_mg_gTOC_xy)
                            (
                                model.tops_list_bs[jj][52][event_index]
                            ) = np.copy(vGEXSRtcf_xy)
                            (
                                model.tops_list_bs[jj][53][event_index]
                            ) = np.copy(vODGEXSRgob_xy)
                            (
                                model.tops_list_bs[jj][54][event_index]
                            ) = np.copy(rho_rgas_g_cc_xy)
                            (
                                model.tops_list_bs[jj][55][event_index]
                            ) = np.copy(rho_rfluid_g_cc_xy)
                            # Conversion to rcf/scf
                            (
                                model.tops_list_bs[jj][56][event_index]
                            ) = np.copy(GFVFrcf_scf_xy)
                            (
                                model.tops_list_bs[jj][57][event_index]
                            ) = np.copy(LFVF_xy)
                            (
                                model.tops_list_bs[jj][70][event_index]
                            ) = np.copy(GOR_liquid_xy)
                            #if itype3D == 0:
                            (
                                model.tops_list_bs[jj][63][event_index]
                            ) = np.copy(mGg_xy)
                            (
                                model.tops_list_bs[jj][64][event_index]
                            ) = np.copy(mOg_xy) 
                            (
                                model.tops_list_bs[jj][65][event_index]
                            ) = np.copy(mCOKEg_xy)
                            (
                                model.tops_list_bs[jj][68][event_index]
                            ) = np.copy(mOCIg_xy)
                        if imass_calc_type == 1: 
                            # Secondary cracking
                            (
                                model.tops_list_bs[jj][58][event_index]
                            ) = np.copy(mFGEXPSmg_gOC_sc_xy)
                            (
                                model.tops_list_bs[jj][59][event_index]
                            ) = np.copy(vGEXSRtcf_sc_xy)
                            (
                                model.tops_list_bs[jj][60][event_index]
                            ) = np.copy(rho_rgas_g_cc_sc_xy)
                            (
                                model.tops_list_bs[jj][61][event_index]
                            ) = np.copy(GFVF_rcf_scf_sc_xy)
                            #if itype3D == 0:
                            (
                                model.tops_list_bs[jj][66][event_index]
                            ) = np.copy(mHCmg_gOC_sc_xy)
                            (
                                model.tops_list_bs[jj][67][event_index]
                            ) = np.copy(mGg_sc_xy)
                            (
                                model.tops_list_bs[jj][69][event_index]
                            ) = np.copy(mOCIg_sc_xy)
                        if oilapi_avg not in OILAPI_list:
                            OILAPI_list.append(oilapi_avg)
                        if gasgrav_avg not in GASGRAV_list:
                            GASGRAV_list.append(gasgrav_avg)                        
                        icount_steps = icount_steps + 1
    return OILAPI_list, GASGRAV_list


@jit(nopython=True, cache=True)
def calculate_expulsion_history_loop(
                                        nx, ny, AOI_np, itype3D, inode, jnode,
                                        event_duration,
                                        icount_steps, imass_calc_type,
                                        mODGEXPSmg_gOC_xy,
                                        mFGEXPSmg_gOC_xy,
                                        mODGEXPSmg_gOC_prev_xy,
                                        mFGEXPSmg_gOC_prev_xy,
                                        ex_rate_xy
):
    for i in range(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if itype3D == 0:
                if i == inode and j == jnode:
                    AOI_flag = AOI_flag
                else:
                    AOI_flag = 0
            if AOI_flag == 1:                                    
                if icount_steps > 0:
                    if imass_calc_type == 0:
                        mODGEXPSmg_gOC_prev = mODGEXPSmg_gOC_prev_xy[i][j]
                        mFGEXPSmg_gOC_prev = mFGEXPSmg_gOC_prev_xy[i][j]
                    else:
                        mODGEXPSmg_gOC_prev = 0.0
                        mFGEXPSmg_gOC_prev = mFGEXPSmg_gOC_prev_xy[i][j]
                else:
                    mODGEXPSmg_gOC_prev = 0.0
                    mFGEXPSmg_gOC_prev = 0.0
                if imass_calc_type == 0:
                    mODGEXPSmg_gOC = mODGEXPSmg_gOC_xy[i][j]
                    mFGEXPSmg_gOC = mFGEXPSmg_gOC_xy[i][j]
                    sum_o = mODGEXPSmg_gOC_prev + mFGEXPSmg_gOC_prev
                    sum_f = mODGEXPSmg_gOC + mFGEXPSmg_gOC
                else:
                    mFGEXPSmg_gOC = mFGEXPSmg_gOC_xy[i][j]
                    sum_o = mFGEXPSmg_gOC_prev
                    sum_f = mFGEXPSmg_gOC
                ex_rate = (sum_f - sum_o)/event_duration
                if imass_calc_type == 0:
                    ex_rate_xy[i][j] = ex_rate
                else:
                    ex_rate_xy[i][j] = ex_rate
                    
                    
def calculate_expulsion_history(imass_calc_type, model):
    # Unpack model object
    #tops_list_bs = model.tops_list_bs
    #event_dict_bs = model.event_dict_bs
    nx = model.nx
    ny = model.ny
    AOI_np = model.AOI_np
    itype3D = model.itype3D
    inode = model.inode
    jnode = model.jnode

    keys = list(model.event_dict_bs.keys())
    ntops = len(model.tops_list_bs)
    # Loop over tops at this location
    for mm in range(ntops):
        jj = ntops - 1 - mm
        icount_steps = 0
        for event_ID, key in enumerate(keys): 
            # Loop over events from old to young
            if event_ID > 0: 
                # Skip the oldest node 
                # since this is "basement rock"
                itop_event = model.event_dict_bs[event_ID][2]
                age_event = model.event_dict_bs[event_ID][0]
                event_ID_prev = event_ID - 1
                age_event_prev = model.event_dict_bs[event_ID_prev][0]
                event_duration = age_event_prev - age_event
                if jj <= itop_event:
                    # We only consider tops 
                    # that have been deposited
                    if icount_steps > 0:
                        event_index_prev = model.tops_list_bs[jj][14]\
                                                                [event_ID_prev]
                    event_index = model.tops_list_bs[jj][14][event_ID]
                    if icount_steps > 0:
                        if imass_calc_type == 0:
                            (
                            mODGEXPSmg_gOC_prev_xy
                            ) = np.copy(
                                  model.tops_list_bs[jj][43][event_index_prev])
                            (
                            mFGEXPSmg_gOC_prev_xy
                            ) = np.copy(
                                  model.tops_list_bs[jj][44][event_index_prev])
                        else:
                            mODGEXPSmg_gOC_prev_xy = np.zeros((nx, ny))
                            (
                            mFGEXPSmg_gOC_prev_xy
                            ) = np.copy(
                                  model.tops_list_bs[jj][58][event_index_prev])
                    else:
                        mODGEXPSmg_gOC_prev_xy = np.zeros((nx, ny))
                        mFGEXPSmg_gOC_prev_xy = np.zeros((nx, ny))
                    if imass_calc_type == 0:
                        ex_rate_xy = np.copy(
                                       model.tops_list_bs[jj][45][event_index])
                        mODGEXPSmg_gOC_xy = np.copy(
                                       model.tops_list_bs[jj][43][event_index])
                        mFGEXPSmg_gOC_xy = np.copy(
                                       model.tops_list_bs[jj][44][event_index])
                    else:
                        mODGEXPSmg_gOC_xy = np.zeros((nx, ny))
                        mFGEXPSmg_gOC_xy = np.copy(
                                       model.tops_list_bs[jj][58][event_index])               
                        ex_rate_xy = np.copy(
                                       model.tops_list_bs[jj][62][event_index])
                    calculate_expulsion_history_loop(
                                                nx, ny, AOI_np, 
                                                itype3D, inode, jnode,
                                                event_duration,
                                                icount_steps, imass_calc_type,
                                                mODGEXPSmg_gOC_xy,
                                                mFGEXPSmg_gOC_xy,
                                                mODGEXPSmg_gOC_prev_xy,
                                                mFGEXPSmg_gOC_prev_xy,
                                                ex_rate_xy
                                            )
                    icount_steps = icount_steps + 1                            


def source_expulsion_history(
                                output_path, tops_list_bs, event_dict_bs,
                                Lx, Ly, nx, ny, dx, dy, 
                                xmin, xmax, ymin, ymax, AOI_np, 
                                itype3D, inode, jnode, src_top_names
):    
    keys = list(event_dict_bs.keys())
    ntops = len(tops_list_bs)
    file_name1 = os.path.join(output_path, 
                              "Source_Expulsion_History_Liquid_GOB_CUM.csv")
    fout1 = open(file_name1, 'w')
    data1 = ["Age_Ma"]
    file_name2 = os.path.join(output_path, 
                              "Source_Expulsion_History_Gas_Tcf_CUM.csv")
    fout2 = open(file_name2, 'w')
    data2 = ["Age_Ma"]
    file_name3 = os.path.join(output_path, 
                              "Source_Expulsion_History_Liquid_GOB_INCR.csv")
    fout3 = open(file_name3, 'w')
    data3 = ["Age_Ma"]
    file_name4 = os.path.join(output_path, 
                              "Source_Expulsion_History_Gas_Tcf_INCR.csv")
    fout4 = open(file_name4, 'w')
    data4 = ["Age_Ma"]
    for event_ID, key in enumerate(keys):  
        # Loop over events from old to young
        itop_event = event_dict_bs[event_ID][2]
        age_event = event_dict_bs[event_ID][0]
        data1.append(age_event)
        data2.append(age_event)
        data3.append(age_event)
        data4.append(age_event)
    str_out = ','.join(map(str, data1)) + "\n"
    fout1.write(str_out)
    str_out = ','.join(map(str, data2)) + "\n"
    fout2.write(str_out)
    str_out = ','.join(map(str, data3)) + "\n"
    fout3.write(str_out)    
    str_out = ','.join(map(str, data4)) + "\n"
    fout4.write(str_out)        
    for mm in range(ntops): 
        # Loop over tops from young to old        
        jj = ntops-1-mm        
        name_top = tops_list_bs[jj][6]
        data1 = [name_top]
        data2 = [name_top]
        data3 = [name_top]
        data4 = [name_top]
        icount = 0         
        for event_ID, key in enumerate(keys):  
            # Loop over events from old to young
            itop_event = event_dict_bs[event_ID][2]
            age_event = event_dict_bs[event_ID][0]
            sum_splexp = 0.0
            sum_spgexp = 0.0
            sum_ssgexp = 0.0            
            # Calculate total cummulative source 
            # expulsion for a given layer and age
            if jj <= itop_event:
                for i in range(nx):                     
                    for j in range(ny): # Rows
                        AOI_flag = AOI_np[i][j]
                        if itype3D == 0:
                            if i == inode and j == jnode:
                                AOI_flag = AOI_flag
                            else:
                                AOI_flag = 0
                        if AOI_flag == 1:
                            event_index = tops_list_bs[jj][14][event_ID]
                            vpg_tcf = tops_list_bs[jj][52][event_index][i][j]
                            if vpg_tcf < 0.0:
                                vpg_tcf = 0.0
                            vpl_gob = tops_list_bs[jj][53][event_index][i][j]
                            if vpl_gob < 0.0:
                                vpl_gob = 0.0
                            vsg_tcf = tops_list_bs[jj][59][event_index][i][j]
                            if vsg_tcf < 0.0:
                                vsg_tcf = 0.0
                            sum_splexp = sum_splexp + vpl_gob
                            sum_spgexp = sum_spgexp + vpg_tcf
                            sum_ssgexp = sum_ssgexp + vsg_tcf
            data1.append(sum_splexp)
            tgas = sum_spgexp + sum_ssgexp
            data2.append(tgas)
            if icount > 0:
                data3.append(sum_splexp-data1[icount-1+1])
                data4.append(tgas - data2[icount-1+1])
            else:
                data3.append(0.0)
                data4.append(0.0)
            icount = icount + 1
        str_out = ','.join(map(str, data1)) + "\n"
        fout1.write(str_out)
        str_out = ','.join(map(str, data2)) + "\n"
        fout2.write(str_out)
        str_out = ','.join(map(str, data3)) + "\n"
        fout3.write(str_out)
        str_out = ','.join(map(str, data4)) + "\n"
        fout4.write(str_out)
    fout1.close()
    fout2.close()       


@jit(nopython=True, cache=True)
def calc_incYield_and_GOR_from_masses(
    nx, ny, event_index, AOI_np, vODGEXSR_GOB_xy_np,
    LFVF_xy_np, rho_rfluid_kg_m3_xy_np,
    vPGEXSR_Tcf_xy_np, PGFVF_xy_np, rho_rpgas_kg_m3_xy_np, GOR_liquid_xy_np,
    vSGEXSR_Tcf_xy_np, SGFVF_xy_np, rho_rsgas_kg_m3_xy_np,
    vODGEXSR_GOB_prev_xy_np, LFVF_prev_xy_np, rho_rfluid_kg_m3_prev_xy_np,
    vPGEXSR_Tcf_prev_xy_np, PGFVF_prev_xy_np, rho_rpgas_kg_m3_prev_xy_np,
    vSGEXSR_Tcf_prev_xy_np, SGFVF_prev_xy_np, rho_rsgas_kg_m3_prev_xy_np,
    mPLg_xy_np, mPFGg_xy_np, mPDGg_xy_np, mPOg_xy_np, mSGFg_xy_np,
    GOR_primary_xy_np, GOR_total_xy_np, mPLg_prev_xy_np, mPFGg_prev_xy_np,
    GOR_liquid_prev_xy_np, mPDGg_prev_xy_np, mPOg_prev_xy_np, mSGFg_prev_xy_np,
    GOR_primary_prev_xy_np, GOR_total_prev_xy_np, mPFGg_i_xy_np, mPDGg_i_xy_np,
    mPOg_i_xy_np, mSGFg_i_xy_np
):
    for i in range(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:                
                # Primary liquid mass expulsed from source
                mPLg_xy_np[i][j] = (
                                    vODGEXSR_GOB_xy_np[i][j]
                                    *LFVF_xy_np[i][j]/6.28981*1000000000
                                    *rho_rfluid_kg_m3_xy_np[i][j]*1000.0
                                )
                # Primary free gas mass expulsed from source
                mPFGg_xy_np[i][j] = (
                                    vPGEXSR_Tcf_xy_np[i][j]
                                    *PGFVF_xy_np[i][j]/35.3147*1000000000000
                                    *rho_rpgas_kg_m3_xy_np[i][j]*1000.0
                                )
                # Primary dissolved gas mass expulsed from source
                GORL = GOR_liquid_xy_np[i][j]
                if GORL > 0:
                    mPDGg_xy_np[i][j] = mPLg_xy_np[i][j]/(1+1/GORL)
                else:
                    mPDGg_xy_np[i][j] = 0
                # Primary oil mass expulsed from source
                mPOg_xy_np[i][j] = mPLg_xy_np[i][j] - mPDGg_xy_np[i][j]          
                # Secondary free gas expulsed from source
                mSGFg_xy_np[i][j] = (
                                    vSGEXSR_Tcf_xy_np[i][j]
                                    *SGFVF_xy_np[i][j]/35.3147*1000000000000
                                    *rho_rsgas_kg_m3_xy_np[i][j]*1000.0
                                )
                mPO = mPOg_xy_np[i][j]
                if mPO > 0:
                    GOR_primary_xy_np[i][j] = (
                                                (
                                                    mPFGg_xy_np[i][j] 
                                                    + mPDGg_xy_np[i][j]
                                                )/mPO
                                            )
                    GOR_total_xy_np[i][j] = (
                                                (
                                                    mPFGg_xy_np[i][j] 
                                                    + mPDGg_xy_np[i][j] 
                                                    + mSGFg_xy_np[i][j]
                                                )/mPO
                                            )
                else:
                    GOR_primary_xy_np[i][j] = 0
                    GOR_total_xy_np[i][j] = 0
                if event_index > 0:
                    # Primary liquid mass expulsed from source
                    mPLg_prev_xy_np[i][j] = (
                                    vODGEXSR_GOB_prev_xy_np[i][j]
                                    *LFVF_prev_xy_np[i][j]/6.28981
                                    *1000000000
                                    *rho_rfluid_kg_m3_prev_xy_np[i][j]*1000.0
                                )
                    # Primary free gas mass expulsed from source
                    mPFGg_prev_xy_np[i][j] = (
                                    vPGEXSR_Tcf_prev_xy_np[i][j]
                                    *PGFVF_prev_xy_np[i][j]/35.3147
                                    *1000000000000
                                    *rho_rpgas_kg_m3_prev_xy_np[i][j]*1000.0
                                )
                    # Primary dissolved gas mass expulsed from source
                    GORL = GOR_liquid_prev_xy_np[i][j]
                    if GORL > 0:
                        mPDGg_prev_xy_np[i][j] = (
                                            mPLg_prev_xy_np[i][j]/(1+1/GORL)
                                        )
                    else:
                        mPDGg_prev_xy_np[i][j] = 0                        
                    # Primary oil mass expulsed from source
                    mPOg_prev_xy_np[i][j] = (
                                                mPLg_prev_xy_np[i][j] 
                                                - mPDGg_prev_xy_np[i][j]
                                            )                            
                    # Secondary free gas expulsed from source
                    mSGFg_prev_xy_np[i][j] = (
                                    vSGEXSR_Tcf_prev_xy_np[i][j]
                                    *SGFVF_prev_xy_np[i][j]/35.3147
                                    *1000000000000
                                    *rho_rsgas_kg_m3_prev_xy_np[i][j]*1000.0
                                )                   
                    mPO_prev = mPOg_prev_xy_np[i][j]
                    if mPO_prev > 0:
                        GOR_primary_prev_xy_np[i][j] = (
                                                (
                                                    mPFGg_prev_xy_np[i][j] 
                                                    + mPDGg_prev_xy_np[i][j]
                                                )/mPO_prev
                                            )
                        GOR_total_prev_xy_np[i][j] = (
                                            (
                                                    mPFGg_prev_xy_np[i][j] 
                                                    + mPDGg_prev_xy_np[i][j] 
                                                    + mSGFg_prev_xy_np[i][j]
                                            )/mPO_prev
                                        )
                    else:
                        GOR_primary_prev_xy_np[i][j] = 0
                        GOR_total_prev_xy_np[i][j] = 0
                    mPFGg_i_xy_np[i][j] = (
                                mPFGg_xy_np[i][j] - mPFGg_prev_xy_np[i][j]
                            )
                    mPDGg_i_xy_np[i][j] = (
                                mPDGg_xy_np[i][j] - mPDGg_prev_xy_np[i][j]
                            )
                    mPOg_i_xy_np[i][j] = (
                                mPOg_xy_np[i][j] - mPOg_prev_xy_np[i][j]
                            )
                    mSGFg_i_xy_np[i][j] = (
                            mSGFg_xy_np[i][j] - mSGFg_prev_xy_np[i][j]
                        )            
                else:
                    mPLg_prev_xy_np[i][j] = 0
                    mPFGg_prev_xy_np[i][j] = 0
                    mPDGg_prev_xy_np[i][j] = 0
                    mPOg_prev_xy_np[i][j] = 0
                    mSGFg_prev_xy_np[i][j] = 0
                    GOR_primary_prev_xy_np[i][j] = 0
                    GOR_total_prev_xy_np[i][j] = 0
                    mPFGg_i_xy_np[i][j] =0
                    mPDGg_i_xy_np[i][j] = 0
                    mPOg_i_xy_np[i][j] = 0
                    mSGFg_i_xy_np[i][j] = 0

                    
def output_gen_and_yield_maps(
                                output_path, model,
                                Lx, Ly, nx, ny, dx, dy, 
                                xmin, xmax, ymin, ymax, 
                                AOI_np, src_top_names
):
    oil_api = 30.0
    sgo = 141.5/(131.5+oil_api)
    rho_os = sgo*1e6 # g/m3
    rho_a = 1292.0 # g/m3
    sgg = 0.65
    rho_gs = rho_a*sgg # g/m3
    cm3_scf = 28316.846711688
    cm3_bbl = 158987.2956
    con_fac = rho_os/rho_gs*cm3_bbl/cm3_scf
    keys = list(model.event_dict_bs.keys())
    # Initialize calculation arrays
    mPLg_xy_np = np.zeros((nx,ny))
    mPFGg_xy_np = np.zeros((nx,ny))
    mPDGg_xy_np = np.zeros((nx,ny))
    mPOg_xy_np = np.zeros((nx,ny))
    mSGFg_xy_np = np.zeros((nx,ny))
    GOR_primary_xy_np = np.zeros((nx,ny))
    GOR_total_xy_np = np.zeros((nx,ny))
    mPLg_prev_xy_np = np.zeros((nx,ny))
    mPFGg_prev_xy_np = np.zeros((nx,ny))
    mPDGg_prev_xy_np = np.zeros((nx,ny))
    mPOg_prev_xy_np = np.zeros((nx,ny))
    mSGFg_prev_xy_np = np.zeros((nx,ny))
    GOR_primary_prev_xy_np = np.zeros((nx,ny))
    GOR_total_prev_xy_np = np.zeros((nx,ny))
    mPFGg_i_xy_np = np.zeros((nx,ny))
    mPDGg_i_xy_np = np.zeros((nx,ny))
    mPOg_i_xy_np = np.zeros((nx,ny))
    mSGFg_i_xy_np = np.zeros((nx,ny))
    vODGEXSR_GOB_prev_xy_np = np.zeros((nx,ny))
    vPGEXSR_Tcf_prev_xy_np = np.zeros((nx,ny))
    PGFVF_prev_xy_np = np.zeros((nx,ny))
    LFVF_prev_xy_np = np.zeros((nx,ny))
    rho_rfluid_kg_m3_prev_xy_np = np.zeros((nx,ny))
    rho_rpgas_kg_m3_prev_xy_np = np.zeros((nx,ny))
    GOR_liquid_prev_xy_np = np.zeros((nx,ny))
    vSGEXSR_Tcf_prev_xy_np = np.zeros((nx,ny))
    SGFVF_prev_xy_np = np.zeros((nx,ny))
    rho_rsgas_kg_m3_prev_xy_np = np.zeros((nx,ny))
    ntops = len(model.tops_list_bs)
    for kk, event_ID in enumerate(keys):
        itop_event = model.event_dict_bs[event_ID][2]
        age = model.event_dict_bs[event_ID][0]
        for jj in range(ntops):
            if jj <= itop_event:
                event_index = model.tops_list_bs[jj][14][event_ID]
                if event_index > 0:
                    event_index_prev = event_index - 1
                name = model.tops_list_bs[jj][6] 
                if name in src_top_names:
                    # Primary source expulsion for current event
                    vODGEXSR_GOB_xy_np = np.copy(
                            model.tops_list_bs[jj][53][event_index])
                    vPGEXSR_Tcf_xy_np = np.copy(
                            model.tops_list_bs[jj][52][event_index])                
                    PGFVF_xy_np = np.copy(
                            model.tops_list_bs[jj][56][event_index]) # rcf/scf                  
                    LFVF_xy_np = np.copy(
                            model.tops_list_bs[jj][57][event_index]) # rbbl/stb  
                    rho_rfluid_kg_m3_xy_np = np.copy(
                            model.tops_list_bs[jj][55][event_index])*1000.0
                    rho_rpgas_kg_m3_xy_np = np.copy(
                            model.tops_list_bs[jj][54][event_index])*1000.0                    
                    GOR_liquid_xy_np = np.copy(
                            model.tops_list_bs[jj][70][event_index]) 
                    # Secondary source expulsion for current event
                    vSGEXSR_Tcf_xy_np = np.copy(
                            model.tops_list_bs[jj][59][event_index])               
                    SGFVF_xy_np = np.copy(
                            model.tops_list_bs[jj][61][event_index]) # rcf/scf
                    rho_rsgas_kg_m3_xy_np = np.copy(
                            model.tops_list_bs[jj][60][event_index])*1000.0                    
                    if event_index > 0:
                        # Primary source expulsion for previous event
                        vODGEXSR_GOB_prev_xy_np = np.copy(
                                model.tops_list_bs[jj][53][event_index_prev])                      
                        vPGEXSR_Tcf_prev_xy_np = np.copy(
                                model.tops_list_bs[jj][52][event_index_prev])
                        # rcf/scf
                        PGFVF_prev_xy_np = np.copy(
                                model.tops_list_bs[jj][56][event_index_prev])
                        # rbbl/stb
                        LFVF_prev_xy_np = np.copy(
                                model.tops_list_bs[jj][57][event_index_prev])
                        rho_rfluid_kg_m3_prev_xy_np = np.copy(
                           model.tops_list_bs[jj][55][event_index_prev])*1000.0
                        rho_rpgas_kg_m3_prev_xy_np = np.copy(
                           model.tops_list_bs[jj][54][event_index_prev])*1000.0                        
                        GOR_liquid_prev_xy_np = np.copy(
                                model.tops_list_bs[jj][70][event_index_prev])
                        # Secondary source expulsion for previous event
                        vSGEXSR_Tcf_prev_xy_np = np.copy(
                                model.tops_list_bs[jj][59][event_index_prev])
                        # rcf/scf
                        SGFVF_prev_xy_np = np.copy(
                                model.tops_list_bs[jj][61][event_index_prev])
                        rho_rsgas_kg_m3_prev_xy_np = np.copy(
                           model.tops_list_bs[jj][60][event_index_prev])*1000.0
                    # Reset arrays to zero and calculate new maps
                    calc_incYield_and_GOR_from_masses(
                        nx, ny, event_index, AOI_np,
                        vODGEXSR_GOB_xy_np, LFVF_xy_np, rho_rfluid_kg_m3_xy_np,
                        vPGEXSR_Tcf_xy_np, PGFVF_xy_np, rho_rpgas_kg_m3_xy_np, 
                        GOR_liquid_xy_np,
                        vSGEXSR_Tcf_xy_np, SGFVF_xy_np, rho_rsgas_kg_m3_xy_np,
                        vODGEXSR_GOB_prev_xy_np, LFVF_prev_xy_np, 
                        rho_rfluid_kg_m3_prev_xy_np,
                        vPGEXSR_Tcf_prev_xy_np, PGFVF_prev_xy_np, 
                        rho_rpgas_kg_m3_prev_xy_np, 
                        vSGEXSR_Tcf_prev_xy_np, SGFVF_prev_xy_np, 
                        rho_rsgas_kg_m3_prev_xy_np, 
                        mPLg_xy_np, mPFGg_xy_np, mPDGg_xy_np, 
                        mPOg_xy_np, mSGFg_xy_np,
                        GOR_primary_xy_np, GOR_total_xy_np, 
                        mPLg_prev_xy_np, mPFGg_prev_xy_np,
                        GOR_liquid_prev_xy_np, mPDGg_prev_xy_np, 
                        mPOg_prev_xy_np, mSGFg_prev_xy_np,
                        GOR_primary_prev_xy_np, GOR_total_prev_xy_np, 
                        mPFGg_i_xy_np, mPDGg_i_xy_np,
                        mPOg_i_xy_np, mSGFg_i_xy_np
                    )
                    stype = "cYield_PO"
                    sflag = "Tg"
                    file_name = (stype + "_"+sflag+"_EV_"+str(event_ID)+"_t"
                                 +str(jj)+"_AGE_"+str(age)+"_TOP_"+name)               
                    map_tools.make_output_file_ZMAP_v4(
                                output_path, file_name, mPOg_xy_np/1e12,
                                nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)
                    stype = "cYield_PFG"
                    sflag = "Tg"
                    file_name = (stype + "_"+sflag+"_EV_"+str(event_ID)+"_t"
                                 +str(jj)+"_AGE_"+str(age)+"_TOP_"+name)                
                    map_tools.make_output_file_ZMAP_v4(
                            output_path, file_name, mPFGg_xy_np/1e12,
                            nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)    
                    stype = "cYield_PDG"
                    sflag = "Tg"
                    file_name = (stype + "_"+sflag+"_EV_"+str(event_ID)+"_t"
                                 +str(jj)+"_AGE_"+str(age)+"_TOP_"+name)                
                    map_tools.make_output_file_ZMAP_v4(
                            output_path, file_name, mPDGg_xy_np/1e12,
                            nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)      
                    stype = "cYield_SFG"
                    sflag = "Tg"
                    file_name = (stype + "_"+sflag+"_EV_"+str(event_ID)+"_t"
                                 +str(jj)+"_AGE_"+str(age)+"_TOP_"+name)               
                    map_tools.make_output_file_ZMAP_v4(
                            output_path, file_name, mSGFg_xy_np/1e12,
                            nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)                    
                    if event_index > 0:
                        stype = "iYield_PO"
                        sflag = "Tg"
                        file_name = (stype + "_"+sflag+"_EV_"+str(event_ID)
                                +"_t"+str(jj)+"_AGE_"+str(age)+"_TOP_"+name)             
                        map_tools.make_output_file_ZMAP_v4(
                                output_path, file_name, mPOg_i_xy_np/1e12,
                                nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)        
                        stype = "iYield_PFG"
                        sflag = "Tg"
                        file_name = (stype + "_"+sflag+"_EV_"+str(event_ID)
                                +"_t"+str(jj)+"_AGE_"+str(age)+"_TOP_"+name)              
                        map_tools.make_output_file_ZMAP_v4(
                                output_path, file_name, mPFGg_i_xy_np/1e12,
                                nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)       
                        stype = "iYield_PDG"
                        sflag = "Tg"
                        file_name = (stype + "_"+sflag+"_EV_"+str(event_ID)
                                +"_t"+str(jj)+"_AGE_"+str(age)+"_TOP_"+name)               
                        map_tools.make_output_file_ZMAP_v4(
                                output_path, file_name, mPDGg_i_xy_np/1e12,
                                nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)       
                        stype = "iYield_SFG"
                        sflag = "Tg"
                        file_name = (stype + "_"+sflag+"_EV_"+str(event_ID)
                                +"_t"+str(jj)+"_AGE_"+str(age)+"_TOP_"+name)               
                        map_tools.make_output_file_ZMAP_v4(
                                output_path, file_name, mSGFg_i_xy_np/1e12,
                                nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)                    
                    stype = "GOR_primary"
                    sflag = "g_g"
                    file_name = (stype + "_"+sflag+"_EV_"+str(event_ID)
                                +"_t"+str(jj)+"_AGE_"+str(age)+"_TOP_"+name)             
                    map_tools.make_output_file_ZMAP_v4(
                            output_path, file_name, GOR_primary_xy_np,
                            nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)   
                    stype = "GOR_primary"
                    sflag = "scf_bbl"
                    file_name = (stype + "_"+sflag+"_EV_"+str(event_ID)
                                +"_t"+str(jj)+"_AGE_"+str(age)+"_TOP_"+name)             
                    map_tools.make_output_file_ZMAP_v4(
                            output_path, file_name, GOR_primary_xy_np*con_fac,
                            nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)    
                    stype = "GOR_total"
                    sflag = "g_g"
                    file_name = (stype + "_"+sflag+"_EV_"+str(event_ID)
                                +"_t"+str(jj)+"_AGE_"+str(age)+"_TOP_"+name)              
                    map_tools.make_output_file_ZMAP_v4(
                            output_path, file_name, GOR_total_xy_np,
                            nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)     
                    stype = "GOR_total"
                    sflag = "scf_bbl"
                    file_name = (stype + "_"+sflag+"_EV_"+str(event_ID)
                                +"_t"+str(jj)+"_AGE_"+str(age)+"_TOP_"+name)            
                    map_tools.make_output_file_ZMAP_v4(
                            output_path, file_name, GOR_total_xy_np*con_fac,
                            nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)
    
