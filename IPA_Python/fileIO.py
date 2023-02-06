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
import numpy as np
import csv
import print_funcs


def str2float(str_input):
    try:
        fval = float(str_input)
    except:
        fval = 0.0
    return fval
        

def tofloat(val):
    try:
        val = float(val)
    except:
        val = 0.0
    return val

        
def read_zmap_plot_file_csv(input_file_path):
    plot_dict = {}
    poly_plot_dict = {}
    with open(input_file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(readCSV):
            if i not in [1,22]:
                ifind_param = 0
                ifind_poly = 0
                if i in [0, 11, 14,20]:
                    var = row[1]
                    ifind_param = 1
                elif i >= 23:
                    fname = row[1]
                    size = tofloat(row[2])
                    color = row[3]
                    if color == "":
                        color = "b"
                    if fname != "":
                        var = [fname, size, color]
                        ifind_poly = 1
                else:
                    var = tofloat(row[1])
                    ifind_param = 1
                if ifind_param == 1:
                    vname = row[0]
                    plot_dict[vname] = var
                if ifind_poly == 1:
                    vname = row[0]
                    poly_plot_dict[vname] = var
    return plot_dict, poly_plot_dict


def read_kinetics_library_csv(input_file_path):
    Ea_all = []
    A_Early_all = []
    f_Early_all = []    
    A_Normal_all = []
    f_Normal_all = []
    A_Late_all = []
    f_Late_all = []
    A_OilCrack = 0.0
    f_OilCrack_all = []
    A_HighLOM_TII = 0.0
    f_HighLOM_TII_all = []
    A_HighLOM_TIII = 0.0
    f_HighLOM_TIII_all = []
    with open(input_file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for irow, row in enumerate(readCSV):
            if irow == 15:
                A_Early_TI = str2float(row[1])
                A_Early_TII = str2float(row[2])
                A_Early_TIII = str2float(row[3])
                A_Early_TIIS = str2float(row[4])
                A_Normal_TI = str2float(row[5])
                A_Normal_TII = str2float(row[6])
                A_Normal_TIII = str2float(row[7])
                A_Normal_TIIS = str2float(row[8])
                A_Late_TI = str2float(row[9])
                A_Late_TII = str2float(row[10])
                A_Late_TIII = str2float(row[11])
                A_Late_TIIS = str2float(row[12])
                A_OilCrack = str2float(row[13])
                A_HighLOM_TII = str2float(row[14])
                A_HighLOM_TIII = str2float(row[15])                
                A_Early_all = np.array([
                                 A_Early_TI, 
                                 A_Early_TII,
                                 A_Early_TIII,
                                 A_Early_TIIS
                                 ])
                A_Normal_all = np.array([
                                 A_Normal_TI, 
                                 A_Normal_TII,
                                 A_Normal_TIII,
                                 A_Normal_TIIS
                                 ])
                A_Late_all = np.array([
                                 A_Late_TI, 
                                 A_Late_TII,
                                 A_Late_TIII,
                                 A_Late_TIIS
                                 ])               
            elif irow > 16:
                Ea = str2float(row[0])
                f_Early_TI = str2float(row[1])
                f_Early_TII = str2float(row[2])
                f_Early_TIII = str2float(row[3])
                f_Early_TIIS = str2float(row[4])
                f_Normal_TI = str2float(row[5])
                f_Normal_TII = str2float(row[6])
                f_Normal_TIII = str2float(row[7])
                f_Normal_TIIS = str2float(row[8])
                f_Late_TI = str2float(row[9])
                f_Late_TII = str2float(row[10])
                f_Late_TIII = str2float(row[11])
                f_Late_TIIS = str2float(row[12])
                f_OilCrack = str2float(row[13])
                f_HighLOM_TII = str2float(row[14])
                f_HighLOM_TIII = str2float(row[15])
                Ea_all.append(Ea)
                f_Early_row = [
                              Ea,
                              f_Early_TI, 
                              f_Early_TII,
                              f_Early_TIII,
                              f_Early_TIIS
                              ]
                f_Early_all.append(f_Early_row)
                f_Normal_row = [
                               Ea,
                               f_Normal_TI, 
                               f_Normal_TII,
                               f_Normal_TIII,
                               f_Normal_TIIS
                               ]
                f_Normal_all.append(f_Normal_row) 
                f_Late_row = [
                             Ea, 
                             f_Late_TI, 
                             f_Late_TII,
                             f_Late_TIII,
                             f_Late_TIIS
                             ]
                f_Late_all.append(f_Late_row)
                f_OilCrack_all.append([Ea, f_OilCrack])
                f_HighLOM_TII_all.append([Ea, f_HighLOM_TII])
                f_HighLOM_TIII_all.append([Ea, f_HighLOM_TIII])
    Ea_all = np.asarray(Ea_all)
    nEa = Ea_all.size     
    f_Early_all = np.asarray(f_Early_all)
    f_Normal_all = np.asarray(f_Normal_all)
    f_Late_all = np.asarray(f_Late_all)
    f_OilCrack_all = np.asarray(f_OilCrack_all)
    f_HighLOM_TII_all = np.asarray(f_HighLOM_TII_all)
    f_HighLOM_TIII_all = np.asarray(f_HighLOM_TIII_all)
    return( nEa,
            Ea_all,
            A_Early_all,
            f_Early_all,
            A_Normal_all,
            f_Normal_all,
            A_Late_all,
            f_Late_all,
            A_OilCrack,
            f_OilCrack_all,
            A_HighLOM_TII,
            f_HighLOM_TII_all,
            A_HighLOM_TIII,
            f_HighLOM_TIII_all
            )   

def read_traps_csv(input_file_path):
    """ Read trap csv file
    
     Description of traps_list
     -------------------------
     index: description
      0: Oil API gravity
      1: Gas specific gravity (surface w.r.t. air)
      2: Oil density factor
      3: Gas density factor
      4: Gas density change (kg/m3)
      5: Sea Water Density kg/m3
      6: Brine Density kg/m3
      7: Gas Saturation GOR Parameter A
      8: Gas Saturation GOR Parameter B
      9: sGOR_corr_charp (scf/bbl)
     10: Characteristic Pressure (psi)
     11: Exponential Factor
     12: GOR limit (scf/bbl)
     13: Cracking On/Off
     14: Biodegradation On/Off
     15: r_max: maximum biodegradation rate g/m2/Myr
     16: T1: Temperature (C ) below which biodegradation rate is at maximum
     17: sigma: Temperature range (C )for the decrease in biodegradation rate
     18: Non-biodegradable fraction %
    """
    traps_dict = {}
    traps_list = []
    traps_output_dict = {}
    ntraps = 0
    with open(input_file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(readCSV):
            if i >= 4 and i <= 22:
                if i not in [17,18]:
                    val = float(row[3])
                else:
                    val = row[3] # String
                traps_list.append(val)
            if i >= 26 and i <= 55:
                trap_name = row[3]
                if trap_name != "":
                    try:
                        x = float(row[4])
                    except:
                        x = 0.0
                    try:
                        y = float(row[5])
                    except:
                        y = 0.0
                    try:
                        Vhcp= float(row[6])
                    except:
                        Vhcp = 0.0
                    try:
                        Vgas_sp = float(row[7])
                    except:
                        Vgas_sp = 0.0
                    try:
                        Vliquid_sp = float(row[8])
                    except:
                        Vliquid_sp = 0.0
                    try:
                        Vgas_dp = float(row[9])
                    except:
                        Vgas_dp = 0.0
                    try:
                        Vliquid_dp = float(row[10])
                    except:
                        Vliquid_dp = 0.0
                    try:
                        OWC_area_km2 = float(row[11])
                    except:
                        OWC_area_km2 = 0.0
                    try:
                        over_press_psi = float(row[12])
                    except:
                        over_press_psi = 0.0
                    try:
                        iclass = int(row[13])
                    except:
                        iclass = 2
                    try:
                        mig_eff = float(row[14])
                    except:
                        mig_eff = 1
                    traps_dict[trap_name] = [
                                                x, y, Vhcp, Vgas_sp, 
                                                Vliquid_sp, Vgas_dp, 
                                                Vliquid_dp, OWC_area_km2, 
                                                over_press_psi, iclass, 
                                                mig_eff
                                            ]
                    Msatliq_tot_g = 0.0      
                    Munsatliq_tot_g = 0.0
                    Mfluid_gas_tot_g = 0.0                  
                    Mfluid_gas_trap_g = 0.0
                    Mdisoil_trap_g = 0.0                            
                    Msatliq_trap_g = 0.0       
                    Munsatliq_trap_g = 0.0
                    Mdisgas_trap_g = 0.0        
                    Mfree_oil_trap_g = 0.0     
                    Mfree_gas_trap_g = 0.0                    
                    Moil_trap_g = 0.0
                    Mgas_trap_g = 0.0                    
                    bGOR_trap_g_g = 0.0
                    bGOR_trap_scf_bbl = 0.0
                    GOR_res_oil_scf_bbl = 0.0
                    GOR_res_gas_scf_bbl = 0.0
                    iclass=-1
                    sphase=""
                    satstate=""                    
                    Vhc_final_m3=0.0
                    perc_fill=0.0
                    liq_fill_frac=0.0
                    fluid_gas_fill_frac=0.0
                    Vfree_oil_trap_GOB=0.0
                    Vfree_fluid_gas_Tcf=0.0
                    trap_mul=0.0
                    sfill=""
                    traps_output_dict[trap_name] = [
                                                     Msatliq_tot_g,
                                                     Munsatliq_tot_g,
                                                     Mfluid_gas_tot_g,
                                                     Mfluid_gas_trap_g,
                                                     Mdisoil_trap_g,
                                                     Msatliq_trap_g,
                                                     Munsatliq_trap_g,
                                                     Mdisgas_trap_g,
                                                     Mfree_oil_trap_g,
                                                     Mfree_gas_trap_g,
                                                     Moil_trap_g,
                                                     Mgas_trap_g,
                                                     bGOR_trap_g_g,
                                                     bGOR_trap_scf_bbl,
                                                     GOR_res_oil_scf_bbl,
                                                     GOR_res_gas_scf_bbl,
                                                     iclass,
                                                     sphase,
                                                     satstate,
                                                     Vhc_final_m3,
                                                     perc_fill,
                                                     liq_fill_frac,
                                                     fluid_gas_fill_frac,
                                                     Vfree_oil_trap_GOB,
                                                     Vfree_fluid_gas_Tcf,
                                                     trap_mul,
                                                     sfill
                                                ]
                    ntraps = ntraps + 1
    return ntraps, traps_list, traps_dict, traps_output_dict


def read_riskrun_file(input_dir_path):
    input_file_path = os.path.join(input_dir_path, "monte_carlo.csv")
    with open(input_file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        sub_dir_list = []
        for i, row in enumerate(readCSV):
            if i == 0: nruns = int(row[1])
            if i == 1: input_path = row[1]
            if i == 2: output_path_root = row[1]
            if i >= 4:
                sub_dir_list.append(row[0])
    return nruns, output_path_root, sub_dir_list, input_path


def read_BatchRun_file_csv(input_file_path):
    scenario_delta_dict = {}
    with open(input_file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(readCSV):
            if i >= 2 and i <= 17:
                if i == 2: key_name = "bghf"
                if i == 3: key_name = "xth"
                if i == 4: key_name = "mantlefac"
                if i == 5: key_name = "def_age"
                if i == 6: key_name = "hf_red_fac"
                if i == 7: key_name = "temp_elastic"
                if i == 8: key_name = "por_decay_depth"
                if i == 9: key_name = "surf_por"
                if i == 10: key_name = "TOC"
                if i == 11: key_name = "HI"
                if i == 12: key_name = "thick"
                if i == 13: key_name = "oilapi"
                if i == 14: key_name = "gasgrav"
                if i == 15: key_name = "porth"
                if i == 16: key_name = "adsth"
                if i == 17: key_name = "inert"
                val_list = []
                for mm, elem in enumerate(row):
                    if mm >= 3 and elem not in [""]:
                        try:
                            val = float(elem)
                            val_list.append(val)
                        except:
                            pass
                scenario_delta_dict[key_name] = val_list[:]
            if i == 18:
                key_name = "kinetics"
                val_list = []
                for mm, elem in enumerate(row):
                    if mm >= 3 and elem not in [""]:
                        ifind = 1                      
                        if elem == "base_case_model":
                            val = 2
                        elif elem == "early":
                            val = -1
                        elif elem == "normal":
                            val = 0
                        elif elem == "late":
                            val = 1
                        else:
                            ifind = 0
                        
                        if ifind == 1:
                            val_list.append(val)
                scenario_delta_dict[key_name] = val_list[:]
            if i == 19:
                key_name = "gas_frac"
                val_list = []
                for mm, elem in enumerate(row):
                    if mm >= 3 and elem not in [""]:
                        ifind = 1                      
                        if elem == "base_case_model":
                            val = 2
                        elif elem == "minimum":
                            val = -1
                        elif elem == "base":
                            val = 0
                        elif elem == "maximum":
                            val = 1
                        else:
                            ifind = 0
                        if ifind == 1:
                            val_list.append(val)
                scenario_delta_dict[key_name] = val_list[:]        
            if i == 20:
                key_name = "polar_frac"
                val_list = []
                for mm, elem in enumerate(row):
                    if mm >= 3 and elem not in [""]:
                        ifind = 1                      
                        if elem == "base_case_model":
                            val = 2
                        elif elem == "minimum":
                            val = -1
                        elif elem == "base":
                            val = 0
                        elif elem == "maximum":
                            val = 1
                        else:
                            ifind = 0
                        
                        if ifind == 1:
                            val_list.append(val)
                scenario_delta_dict[key_name] = val_list[:]
            if i >= 23 and i <= 32:
                key_name = "src_index"
                if i == 23:
                    scenario_delta_dict[key_name] = []
                val_list = []
                for mm, elem in enumerate(row):
                    if mm >= 3 and elem not in [""]:
                        ifind = 1
                        if elem == "ALL":
                            val = -1
                        elif elem =="1st":
                            val = 0
                        elif elem =="2nd":
                            val = 1
                        elif elem =="3rd":
                            val = 2
                        elif elem =="4th":
                            val = 3
                        elif elem =="5th":
                            val = 4
                        elif elem =="6th":
                            val = 5
                        else:
                            ifind = 0
                        if ifind == 1:
                            val_list.append(val)
                if len(val_list) > 0:
                    scenario_delta_dict[key_name].append(val_list)            
    return scenario_delta_dict


def read_RiskSensitivty_file_csv(input_file_path):
    """ Read risk sensitivity csv file

     Uncertainty and sensitivity parameters
    ---------------------------------------
     a and b
         Beta distribution inputs controlling the shape of the pdf.
    
     c
         Controls the location of the apex in the triangular 
         distribution and can have values ranging from 0 to 1. 
    
     pdf 
         Proability distribution type: 
         "beta"
         "triangular"
         "uniform"
    
     delta_min and delta_max
         Control the boundaries of the distributions and high-low values for 
         high-low analysis.
    """
    pdf_dict = {}
    discrete_dict = {}
    src_index_list_bulk_gor_scenario1 = []
    src_index_list_bulk_gor_scenario2 =[]
    src_index_list_bulk_gor_scenario3 =[]
    srtype = "Note Used"
    nruns = "Not Used"
    with open(input_file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(readCSV):
            if i >= 5 and i <= 20:                
                min_val = row[2]
                max_val = row[3]
                dtype = row[4]
                a = float(row[5])
                b = float(row[6])
                c = float(row[7])                
                if i == 5: key_name = "bghf"
                if i == 6: key_name = "xth"
                if i == 7: key_name = "mantlefac"
                if i == 8: key_name = "def_age"
                if i == 9: key_name = "hf_red_fac"
                if i == 10: key_name = "temp_elastic"
                if i == 11: key_name = "por_decay_depth"
                if i == 12: key_name = "surf_por"
                if i == 13: key_name = "TOC"
                if i == 14: key_name = "HI"
                if i == 15: key_name = "thick"
                if i == 16: key_name = "oilapi"
                if i == 17: key_name = "gasgrav"
                if i == 18: key_name = "porth"
                if i == 19: key_name = "adsth"
                if i == 20: key_name = "inert"
                if key_name in ["adsth", "inert", "temp_elastic"]:
                    min_val = float(min_val)
                    max_val = float(max_val)
                pdf_dict[key_name] = [dtype, a, b, c, min_val, max_val]
            if i == 23:
                p1 = float(row[2])
                discrete_dict["kinetics"] = [p1]            
            if i == 24:
                p2 = float(row[2])
                discrete_dict["kinetics"].append(p2)
            if i == 25:
                p3 = float(row[2])
                discrete_dict["kinetics"].append(p3)
            if i == 28:
                p1 = float(row[2])
                discrete_dict["gas_frac"] = [p1]            
            if i == 29:
                p2 = float(row[2])
                discrete_dict["gas_frac"].append(p2)
            if i == 30:
                p3 = float(row[2])
                discrete_dict["gas_frac"].append(p3)
            if i == 33:
                p1 = float(row[2])
                discrete_dict["polar_frac"] = [p1]            
            if i == 34:
                p2 = float(row[2])
                discrete_dict["polar_frac"].append(p2)
            if i == 35:
                p3 = float(row[2])
                discrete_dict["polar_frac"].append(p3)
            if i == 39:
                p1 = float(row[2])
                discrete_dict["src_index"] = [p1]            
            if i == 40:
                p2 = float(row[2])
                discrete_dict["src_index"].append(p2)
            if i == 41:
                p3 = float(row[2])
                discrete_dict["src_index"].append(p3)
            if i == 42:
                nvals = len(row)
                for mm in range(nvals):
                    if row[mm] in ["ALL","1st","2nd","3rd","4th","5th","6th"]:
                        ival = -99999
                        if row[mm] == "ALL": ival = -1
                        if row[mm] == "1st": ival = 0
                        if row[mm] == "2nd": ival = 1
                        if row[mm] == "3rd": ival = 2
                        if row[mm] == "4th": ival = 3
                        if row[mm] == "5th": ival = 4
                        if row[mm] == "6th": ival = 5
                        src_index_list_bulk_gor_scenario1.append(ival)
            if i == 43:
                nvals = len(row)
                for mm in range(nvals):
                    if row[mm] in ["ALL","1st","2nd","3rd","4th","5th","6th"]:
                        ival = -99999
                        if row[mm] == "ALL": ival = -1
                        if row[mm] == "1st": ival = 0
                        if row[mm] == "2nd": ival = 1
                        if row[mm] == "3rd": ival = 2
                        if row[mm] == "4th": ival = 3
                        if row[mm] == "5th": ival = 4
                        if row[mm] == "6th": ival = 5
                        src_index_list_bulk_gor_scenario2.append(ival)
            if i == 44:
                nvals = len(row)
                for mm in range(nvals):
                    if row[mm] in ["ALL","1st","2nd","3rd","4th","5th","6th"]:
                        ival = -99999
                        if row[mm] == "ALL": ival = -1
                        if row[mm] == "1st": ival = 0
                        if row[mm] == "2nd": ival = 1
                        if row[mm] == "3rd": ival = 2
                        if row[mm] == "4th": ival = 3
                        if row[mm] == "5th": ival = 4
                        if row[mm] == "6th": ival = 5
                        src_index_list_bulk_gor_scenario3.append(ival)                          
    return (
            srtype, nruns, pdf_dict, discrete_dict, 
            src_index_list_bulk_gor_scenario1,
            src_index_list_bulk_gor_scenario2, 
            src_index_list_bulk_gor_scenario3
        )


def read_polys_open_vs_closed(input_file_path):
    poly_dict = {}
    file_path = input_file_path
    fin = open(file_path, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        data = line.split()
        ndata = len(data)
        if ndata == 3:
            ID = int(data[0])
            x = float(data[1])
            y = float(data[2])
            all_IDs = list(poly_dict.keys())
            if ID not in all_IDs:
                poly_dict[ID] = [[x,y,0]]
            else:
                poly_dict[ID].append([x,y,0])
    fin.close()
    allkeys = list(poly_dict.keys())
    for i in allkeys:
        coors = poly_dict[i]
        ncoors = len(coors)
        p1x = coors[0][0]
        p1y = coors[0][1]
        pfx = coors[ncoors-1][0]
        pfy = coors[ncoors-1][1]
        if p1x != pfx or p1y != pfy:
            coors[ncoors-1][2] = 1
    return poly_dict


def read_polys_simple(input_file_path):
    poly_dict = {}
    file_path = input_file_path
    fin = open(file_path, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        data = line.split()
        ndata = len(data)
        if ndata == 3:
            ID = int(data[0])
            x = float(data[1])
            y = float(data[2])
            all_IDs = list(poly_dict.keys())
            if ID not in all_IDs:
                poly_dict[ID] = [[x,y]]
            else:
                poly_dict[ID].append([x,y])
    fin.close()
    allkeys = list(poly_dict.keys())
    for i in allkeys:
        coors = poly_dict[i]
        ncoors = len(coors)
        p1x = coors[0][0]
        p1y = coors[0][1]
        pfx = coors[ncoors-1][0]
        pfy = coors[ncoors-1][1]
        if p1x != pfx or p1y != pfy:
            print(
                    "!!!! WARNING !!!!: ncoors : ",ncoors, 
                    " : polygon with ID ", i, 
                    " is not closed. Check ", file_path
                )
            print("     ---> p1x, pfx, p1y, pfy : ", p1x, pfx, p1y, pfy)
    return poly_dict
                

def read_well_file_csv(input_file_path):
    wells_dict = {}
    nwells = 0
    with open(input_file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(readCSV):
            well_name = row[0]
            if well_name != "":
                UWI = row[1]
                try:
                    x = float(row[2])
                except:
                    x = 0.0
                try:
                    y = float(row[3])
                except:
                    y = 0.0
                try:
                    td = float(row[4])
                except:
                    td = 0.0
                try:
                    wd = float(row[5])
                except:
                    wd = 0.0
                try:
                    bghf = float(row[6])
                except:
                    bghf = 0.0
                try:
                    com = row[7]
                except:
                    com = ""
                wells_dict[well_name] = [x, y, td, wd, bghf, com, UWI]
                nwells = nwells + 1
    return nwells, wells_dict
            

def read_APWP_file_csv(input_file_path):
    nx = 56
    APWP_dict = {}
    APWP_dict["AUS"]  = [[],[],[]]
    APWP_dict["EUR"]  = [[],[],[]]
    APWP_dict["IND"]  = [[],[],[]]
    APWP_dict["NAM"]  = [[],[],[]]
    APWP_dict["SAM"]  = [[],[],[]]
    APWP_dict["SAFR"]  = [[],[],[]]
    APWP_dict["SIB"]  = [[],[],[]]
    for i in range(3):
        for j in range(nx):
            APWP_dict["AUS"][i].append(0.0)
            APWP_dict["EUR"][i].append(0.0)
            APWP_dict["IND"][i].append(0.0)
            APWP_dict["NAM"][i].append(0.0)
            APWP_dict["SAM"][i].append(0.0)
            APWP_dict["SAFR"][i].append(0.0)
            APWP_dict["SIB"][i].append(0.0)
    with open(input_file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(readCSV):
            row_num = i + 1
            # AUS
            if row_num == 1:
                for ix in range(nx):
                    APWP_dict["AUS"][0][ix] = float(row[ix+2])
            if row_num == 2:
                for ix in range(nx):
                    APWP_dict["AUS"][1][ix] = float(row[ix+2])
            if row_num == 3:
                for ix in range(nx):
                    APWP_dict["AUS"][2][ix] = float(row[ix+2]) 
            # EUR
            if row_num == 8:
                for ix in range(nx):
                    APWP_dict["EUR"][0][ix] = float(row[ix+2])

            if row_num == 9:
                for ix in range(nx):
                    APWP_dict["EUR"][1][ix] = float(row[ix+2])
            if row_num == 10:
                for ix in range(nx):
                    APWP_dict["EUR"][2][ix] = float(row[ix+2]) 
            # IND
            if row_num == 15:
                for ix in range(nx):
                    APWP_dict["IND"][0][ix] = float(row[ix+2])
            if row_num == 16:
                for ix in range(nx):
                    APWP_dict["IND"][1][ix] = float(row[ix+2])
            if row_num == 17:
                for ix in range(nx):
                    APWP_dict["IND"][2][ix] = float(row[ix+2]) 
            # NAM
            if row_num == 22:
                for ix in range(nx):
                    APWP_dict["NAM"][0][ix] = float(row[ix+2])
            if row_num == 23:
                for ix in range(nx):
                    APWP_dict["NAM"][1][ix] = float(row[ix+2])                
            if row_num == 24:
                for ix in range(nx):
                    APWP_dict["NAM"][2][ix] = float(row[ix+2]) 
            # SAM
            if row_num == 29:
                for ix in range(nx):
                    APWP_dict["SAM"][0][ix] = float(row[ix+2])
            if row_num == 30:
                for ix in range(nx):
                    APWP_dict["SAM"][1][ix] = float(row[ix+2])            
            if row_num == 31:
                for ix in range(nx):
                    APWP_dict["SAM"][2][ix] = float(row[ix+2])
            # SAFR
            if row_num == 36:
                for ix in range(nx):
                    APWP_dict["SAFR"][0][ix] = float(row[ix+2])
            if row_num == 37:
                for ix in range(nx):
                    APWP_dict["SAFR"][1][ix] = float(row[ix+2])                
            if row_num == 38:
                for ix in range(nx):
                    APWP_dict["SAFR"][2][ix] = float(row[ix+2])
            # SIB
            if row_num == 43:
                for ix in range(nx):
                    APWP_dict["SIB"][0][ix] = float(row[ix+2])
            if row_num == 44:
                for ix in range(nx):
                    APWP_dict["SIB"][1][ix] = float(row[ix+2])
            if row_num == 45:
                for ix in range(nx):
                    APWP_dict["SIB"][2][ix] = float(row[ix+2])   
    return APWP_dict


def read_surf_temp_file_csv(input_file_path):
    nx = 499
    ny = 181
    surf_temp_age = np.zeros((nx))
    surf_temp_lat = np.zeros((ny))
    surf_temp_xy = np.zeros((nx, ny))
    with open(input_file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        iy = 0
        for i, row in enumerate(readCSV):
            if i == 0:
                ix = 0 
                for jcol in range(nx+1):
                    if jcol > 0:
                        age = float(row[jcol])
                        surf_temp_age[ix] = age
                        ix = ix + 1
            elif i > 0:
                surf_temp_lat[iy] = float(row[0])
                ix = 0 
                for jcol in range(nx+1):
                    if jcol > 0:
                        surf_temp_xy[ix][iy] = float(row[jcol])
                        ix = ix + 1
                iy = iy + 1
    return surf_temp_age, surf_temp_lat, surf_temp_xy


def read_lithology_file_csv(input_file_path):
    lith_dict = {}
    icount = 0
    with open(input_file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(readCSV):
            if i > 0: # Skip headers
                e1 = row[0]
                if e1 != "":
                    try: 
                        e2 = float(row[1])
                    except:
                        e2 = 4000.0
                    try:
                        e3 = float(row[2])
                    except:
                        e3 = 0.0
                    try:
                        e4 = float(row[3])
                    except:
                        e4 = 2.0
                    try:
                        e5 = float(row[4])
                    except:
                        e5 = 850
                    try:
                        e6 = float(row[5])
                    except:
                        e6 = 2700 
                    try: 
                        e7 = float(row[6])
                    except:
                        e7 = 40.0
                    try:
                        e8 = float(row[7])
                    except:
                        e8 = 1.5
                    try: 
                        e9 = float(row[8])
                    except:
                        e9 = 0.0
                    try:
                        e10 = float(row[9])
                    except:
                        e10 = 0.0  
                    e11 = row[10]
                    try:
                        eID = int(row[11])
                    except:
                        eID = 1
                    # OMT
                    e12 = row[12]
                    try:
                        # HI
                        e13 = float(row[13])
                    except:
                        e13 = 500
                    try:
                        # TOC
                        e14 = float(row[14])
                    except:
                        e14 = 5.0
                    # Kinetics
                    e15 = row[15]
                    try:
                        # tS
                        e16 = float(row[16])
                    except:
                        e16 = 1.0
                    # as
                    e17 = row[17]                  
                    try:
                        # OIL_API
                        e18 = float(row[18])
                    except:
                        e18 = 35
                    try:
                        # gas_grav
                        e19 = float(row[19])
                    except:
                        e19 = 0.8
                    try:
                        # porosity threshold frac
                        e20 = float(row[20])
                    except:
                        e20 = 0.06
                    try:
                         # gas fraction scenario
                        e21 = row[21]
                    except:
                        e21 = "base"
                    try:
                        # polar fraction scenario
                        e22 = row[22]
                    except:
                        e22 = "base"
                    lith_dict[eID] = [
                                        e1, e2, e3, e4, 
                                        e5, e6, e7, e8, 
                                        e9, e10, e11, e12, 
                                        e13, e14, e15, e16, 
                                        e17, e18, e19, e20, 
                                        e21, e22
                                    ] 
                    icount = icount + 1
    print_funcs.print_lithology_table(lith_dict)
    return lith_dict


def read_main_input_file_csv(
                                input_file_path, 
                                strat_input_dict, param_input_dict
):
    Ntops = 30 # 30 is the maximum number of allowed tops
    icount_layers = 0  
    # Using tmp dict to flip the order of events
    strat_input_dict_tmp = {}
    deltaSL_list_tmp = []
    deltaSL_list = [] # change in sea level for each event
    for i in range(Ntops):
        deltaSL_list_tmp.append(0.0)
        deltaSL_list.append(0.0)
    swit_list_tmp = []
    swit_list = []
    for i in range(Ntops):
        swit_list_tmp.append(0.0)
        swit_list.append(0.0)
    # Initialize some parameters for old input files that do not include new
    # parameters    
    param_input_dict["TRmin"] = 0
    param_input_dict["TRmax"] = 1
    param_input_dict["gas_frac_adj_TRmin"] = 1
    param_input_dict["gas_frac_adj_TRmax"] = 1
    param_input_dict["ahf_max"] = 250
    param_input_dict["iuse_high_lom_gas"] = 1
    with open(input_file_path) as csvfile:        
        readCSV = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(readCSV):            
            if i == 0:                
                Ntops = int(row[1])
                itop_max = Ntops-1            
            if i >= 2 and i < 2 + Ntops:                
                ID = itop_max - icount_layers
                try:
                    # Age
                    e1 = float(row[1])
                except:
                    e1 = 0.0
                # Name
                e2 = str(row[2])
                # Event Type
                e3 = str(row[3])
                if e3 not in ["Deposition","Erosion_and_Deposition","Erosion"]:
                    e3 = "Deposition"
                    print("!!! Event type not found for", e2, 
                          ". Assuming Deposition ... !!!")
                # TVD
                e4 = str(row[4])
                 # Shift
                e5 = str(row[5])
                # WD
                e6 = str(row[6])
                # salt thickness 
                e7 = str(row[7])
                # TWT
                e8 = str(row[8])
                # Lith
                e9 = str(row[9])
                try:
                    e10 = float(row[10]) # Int veloc
                except:
                    e10 = 2500.0
                try:
                    e11 = float(row[11]) # SL
                except:
                    e11 = 0.0
                try:
                    e12 = str(row[12]) # petroleum systems element
                except:
                    e12 = "None"
                try:
                    e13 = str(row[13]) # fetch poly file name
                except:
                    e13 = "None" 
                strat_input_dict_tmp[ID] = [
                                            e1, e2, e3, e4, 
                                            e5, e6, e7, e8, 
                                            e9, e10, e12, e13
                                        ]
                deltaSL_list_tmp[ID] = e11
                icount_layers = icount_layers + 1
            if i == 32: param_input_dict["nphases_name"] = row[1]     
            if i == 33: param_input_dict["start_age_rift1_name"] = row[1]
            if i == 34: param_input_dict["start_age_rift2_name"] = row[1]
            if i == 35: param_input_dict["start_age_rift3_name"] = row[1]
            if i == 36: param_input_dict["end_age_rift1_name"] = row[1]
            if i == 37: param_input_dict["end_age_rift2_name"] = row[1]
            if i == 38: param_input_dict["end_age_rift3_name"] = row[1]
            if i == 39: param_input_dict["mantlefac_rift1_name"] = row[1]
            if i == 40: param_input_dict["mantlefac_rift2_name"] = row[1]
            if i == 41: param_input_dict["mantlefac_rift3_name"] = row[1]
            if i == 42: param_input_dict["riftmag_rift1_name"] = row[1]
            if i == 43: param_input_dict["riftmag_rift2_name"] = row[1]
            if i == 44: param_input_dict["riftmag_rift3_name"] = row[1]
            if i == 45: param_input_dict["background_hf_map_file"] = row[1]
            if i ==46: param_input_dict["idepth_model"] = int(row[1])
            if i ==47: param_input_dict["niter_t2d_comp"] = int(row[1]) 
            if i ==48: param_input_dict["icompact"] = int(row[1])
            if i ==49: param_input_dict["dt_age_PWD"] = 10
            if i ==50: param_input_dict["age_start_PWD"] = 130
            if i ==51: param_input_dict["age_end_PWD"] = 0.0
            if i ==52: param_input_dict["inv_itype"] = int(row[1])
            if i ==53: param_input_dict["shift_fac"] = float(row[1])
            if i ==54: param_input_dict["TOL_delta_bisec"] = float(row[1])
            if i ==55: param_input_dict["iuse_numerical_rift"] = int(row[1])
            if i ==56: param_input_dict["itype_rho_a"] = int(row[1])
            if i ==57: param_input_dict["dt_Myr"] = float(row[1])
            if i ==58: param_input_dict["dt_rift_Myr"] = float(row[1])
            if i ==59: param_input_dict["dz_lith"] = float(row[1])
            if i ==60: param_input_dict["rift_itype"] = int(row[1])
            if i ==61: param_input_dict["tc_initial"] = float(row[1])
            if i ==62: param_input_dict["tm_initial"] = float(row[1])
            if i ==63: param_input_dict["rho_water"] = float(row[1]) 
            if i ==64: param_input_dict["rho_crust"] = float(row[1])
            if i ==65: param_input_dict["rho_mantle"] = float(row[1])
            if i ==66: param_input_dict["k_bulk"] = float(row[1])
            if i ==67: param_input_dict["cp_bulk"] = float(row[1])
            if i ==68: param_input_dict["alpha_bulk"] = float(row[1])
            if i ==69: param_input_dict["T_base"] = float(row[1])
            if i ==70: param_input_dict["HP_itype"] = int(row[1])
            if i ==71: param_input_dict["Q_crust"] = float(row[1])
            if i ==72: param_input_dict["Q_mantle"] = float(row[1])
            if i ==73: param_input_dict["Ao"] = float(row[1])
            if i ==74: param_input_dict["ar"] = float(row[1])
            if i ==75: param_input_dict["L_crust_ref"] = float(row[1])
            if i ==76: param_input_dict["L_lith_ref"] = float(row[1])
            if i ==77: param_input_dict["rho_crust_ref"] = float(row[1])
            if i ==78: param_input_dict["rho_mantle_ref"] = float(row[1])
            if i ==79: param_input_dict["iuse_trans"] = int(row[1])
            if i ==80: param_input_dict["kappa_lith"] = float(row[1])
            if i ==81: param_input_dict["iuse_temp_dep_k"] = int(row[1])
            if i ==82: param_input_dict["nrelax"] = int(row[1])
            if i ==83: param_input_dict["nsublayers"] = int(row[1])
            if i ==84: param_input_dict["iuse_anomalous_heatflow"]= int(row[1])
            if i ==85: param_input_dict["hf_reduc_fac_map_file"] = row[1]
            if i ==86: param_input_dict["k_water"] = float(row[1])
            if i ==87: param_input_dict["Q_water"] = float(row[1])
            if i ==88: param_input_dict["icalc_SWIT"] = int(row[1])
            if i ==89: param_input_dict["iupdate_PWD"] = int(row[1])
            if i ==90: param_input_dict["T_top_input"] = float(row[1])
            if i ==91: param_input_dict["lat_LL"] = float(row[1])
            if i ==92: param_input_dict["lon_LL"] = float(row[1])
            if i ==93: param_input_dict["splate"] = row[1]
            if i ==94: param_input_dict["inert_frac"] = float(row[1])
            if i ==95: param_input_dict["adsorption_perc"] = float(row[1])
            if i ==96: param_input_dict["ioutput_burial"] = int(row[1])
            if i ==97: param_input_dict["ioutput_TTS"] = int(row[1])
            if i ==98: param_input_dict["ioutput_FW_TTS"] = int(row[1])
            if i ==99: param_input_dict["ioutput_HFTOT"] = int(row[1])
            if i ==100: param_input_dict["ioutput_HF_ANOM"] = int(row[1])
            if i ==101: param_input_dict["icalc_temp"] = int(row[1])
            if i ==102: param_input_dict["icalc_LOM"] = int(row[1])
            if i ==103: param_input_dict["imass_gen"] = int(row[1])
            if i ==104: param_input_dict["isalt_restore"] = int(row[1])
            if i ==105: param_input_dict["salt_layer_index"] = int(row[1])
            if i ==106: param_input_dict["rad_search_m"] = float(row[1])
            if i ==121: param_input_dict["iuse_flexure"] = int(row[1])
            if i ==122: param_input_dict["temp_elastic"] = float(row[1])
            if i ==123: param_input_dict["dist_taper"] = float(row[1])
            if i ==124: param_input_dict["xth_file_name"] = row[1]
            if i ==125: param_input_dict["ioutput_TEMP"] = int(row[1])
            if i ==126: param_input_dict["ioutput_Ro"] = int(row[1])
            if i ==127: param_input_dict["ioutput_LOM"] = int(row[1])
            if i ==128: param_input_dict["ioutput_TR"] = int(row[1])
            if i ==129: param_input_dict["ioutput_mHC"] = int(row[1])
            if i ==130: param_input_dict["ioutput_mODG"] = int(row[1])
            if i ==131: param_input_dict["ioutput_mFG"] = int(row[1])
            if i ==132: param_input_dict["ioutput_SEC_mFG"] = int(row[1])
            if i ==133: param_input_dict["ioutput_EXPRATE"] = int(row[1])
            if i ==134: param_input_dict["ioutput_SEC_EXPRATE"] = int(row[1])
            if i ==135: param_input_dict["TRmin"] = float(row[1])
            if i ==136: param_input_dict["TRmax"] = float(row[1])
            if i ==137: param_input_dict["gas_frac_adj_TRmin"] = float(row[1])
            if i ==138: param_input_dict["gas_frac_adj_TRmax"] = float(row[1])
            if i ==139: param_input_dict["ahf_max"] = float(row[1])
            if i ==140: param_input_dict["iuse_high_lom_gas"] = int(row[1])
    # Calculate ages for PWD interpolation...
    age_start_PWD = param_input_dict["age_start_PWD"]
    age_end_PWD = param_input_dict["age_end_PWD"]
    dt_age_PWD = param_input_dict["dt_age_PWD"]
    nages = int((age_start_PWD-age_end_PWD)/dt_age_PWD)
    PWD_interp_ages = []
    for i in range(nages):
        age_PWD = age_start_PWD - dt_age_PWD*float(i)
        PWD_interp_ages.append(age_PWD)
    keys = list(strat_input_dict_tmp.keys())
    nkeys = len(keys)
    ilast = nkeys - 1
    for i, key in enumerate(keys):
        ID = ilast - i
        #print("i, ID, key : ", i, ID, key)
        # Need to make sure that key for first entry is 0
        strat_input_dict[i] = strat_input_dict_tmp[i][:]
        # We flip the sign for the TTS equation 
        # So now negatives refer to above present-day and positives refer to below
        deltaSL_list[i] = -deltaSL_list_tmp[i]
    param_input_dict["deltaSL_list"] = deltaSL_list
    param_input_dict["PWD_interp_ages"] = PWD_interp_ages
    param_input_dict["swit_list"] = swit_list     


def make_riskrun_csv(
                        output_path, ir, file_name, 
                        nruns, src_index_list_bulk_gor
):
    file_name = os.path.join(output_path, file_name+".csv")
    fout = open(file_name, 'w')
    nameList = ["Run#", "nruns", "src_index_scenario_bulk_gor"]
    nsrc = len(src_index_list_bulk_gor)
    if nsrc > 0:
        for i in range(nsrc):
            sindex = src_index_list_bulk_gor[i]
            if sindex == -1:
                str_tmp = "ALL_SOURCES_USED_FOR_BULK_GOR"
            elif sindex == 0:
                str_tmp = "1st_SOURCE_USED_FOR_BBULK_GOR"
            elif sindex == 1:
                str_tmp = "2nd_SOURCE_USED_FOR_BULK_GOR"
            elif sindex == 2:
                str_tmp = "3rd_SOURCE_USED_FOR_BULK_GOR"
            elif sindex == 3:
                str_tmp = "4th_SOURCE_USED_FOR_BULK_GOR"
            elif sindex == 4:
                str_tmp = "5th_SOURCE_USED_FOR_BULK_GOR"
            elif sindex == 5:
                str_tmp = "6th_SOURCE_USED_FOR_BULK_GOR"
            elif sindex >= 6:
                str_tmp = "7_or_MORE_SOURCES_USED_FOR_BULK_GOR"
            if i == 0:
                str_main = str_tmp
            else:
                str_main = str_tmp +" AND "+ str_tmp
    else:
        str_tmp = "NO_SOURCES_USED_FOR_BULK_GOR"
    valList = [ir, nruns, str_main]
    unitlist = ["unit", "unit", "Sources Used in Bulk GOR Calc"]
    for i, val in enumerate(valList):
        name = nameList[i]
        str_out = name + ","+ str(val) + "," + unitlist[i] + " \n"
        fout.write(str_out)
    fout.close()


def make_trap_csv(
                    output_path, trap_name, 
                    nages, output_strs_list, 
                    output_list, trap_mul_tot
):
    file_name = os.path.join(output_path, "trap_" + trap_name+".csv")
    fout = open(file_name, 'w')
    for j, val in enumerate(output_strs_list):
        if j == 0:
            str_out = str(val)
        else:
            str_out = str_out + "," + str(val)
    str_out = str_out + " \n"    
    fout.write(str_out)    
    for i, elem in enumerate(output_list):
        for j, val in enumerate(elem):
            if j == 0:
                str_out = str(val)
            else:
                str_out = str_out + "," + str(val)
        str_out = str_out + " \n"        
        fout.write(str_out)
    str_out = "Total_Trap_Multiple, " + str(trap_mul_tot) + " \n"
    fout.write(str_out)
    fout.close()