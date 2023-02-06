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
import math
import time
import numpy as np
import csv
import map_tools
import print_funcs


def calc_1D_rift_params(model):
    nx = model.nx
    ny = model.ny
    input1D_dict = model.input1D_dict
    rift_mag1_xy = model.rift_mag1_xy
    rift_mag2_xy = model.rift_mag2_xy
    mantle_fac1_xy = model.mantle_fac1_xy
    mantle_fac2_xy = model.mantle_fac2_xy
    tc_ini = model.tc_initial
    xth_xy = np.zeros((nx,ny))
    # Not currently setup for multiple events
    delta1 = input1D_dict["ev1_delta"]
    beta1 = input1D_dict["ev1_beta"]
    delta2 = input1D_dict["ev2_delta"]
    beta2 = input1D_dict["ev2_beta"]
    for i in range(nx): # Columns
        for j in range(ny): # Rows    
            xth = tc_ini/delta1
            xth_xy[i][j] = xth
            rift_mag1_xy[i][j] = 1.0
            rift_mag2_xy[i][j] = 1.0
            mantle_fac1_xy[i][j] = beta1/delta1
            mantle_fac2_xy[i][j] = beta2/delta2
    return xth_xy


def extract_for_1D_model(model, ioutput_main, process):
    rho_water = model.rho_water
    k_water = model.k_water
    output_path = model.output_path
    inode = model.inode
    jnode = model.jnode
    if model.itype3D == 0:
        tt1 = time.time()
        naflag = -99999.0
        if model.imass_gen > 0 and model.icalc_temp ==1:
            extract_at_node_history_for_IPA1D(
                                        inode, jnode, naflag, output_path,
                                        model, rho_water, k_water
                                    )
        tt2 = time.time()
        print_funcs.print_finfo(
                            ioutput_main, process, 
                            "Extracted history for 1D model", tt2-tt1)
        
        
def rift_params_1D(model, ioutput_main, process, xth_xy):
    if model.inv_itype < 0 and model.itype3D == 0:
        tt1 = time.time()
        # compute xth_xy based on deltas etc...
        xth_xy = calc_1D_rift_params(model)
        tt2 = time.time()
        print_funcs.print_finfo(
                    ioutput_main, process, 
                    "Calculated 1D rift parameters for 1D model", tt2-tt1)
    return xth_xy
    

def make_ipa_wells_file_for1D(input_path, xmax, ymax):
    file_name = input_path + "ipa_wells.csv"
    fout = open(file_name, 'w')
    xwell = xmax/2.0
    ywell = ymax/2.0
    td = 15000.0
    wd = 0.0
    for i in range(200):
        if i == 0:
            data = ["Well_Dum","", xwell, ywell, td, wd, 
                    55, "1D extraction point"]
            str_out = ','.join(map(str, data)) + "\n"
            fout.write(str_out)
        else:
            data = ["","", "", "", "", "", 55, ""]
            str_out = ','.join(map(str, data)) + "\n"
            fout.write(str_out)            
    fout.close()    
    

def define_basic_geometry_for1D(input1D_dict):
    nx = 4
    ny = 4    
    # Adjust dx and dy
    area = input1D_dict["src_area"]
    print("cell area (km2) : ", area)    
    dx = math.sqrt(area)*1000
    dy = dx    
    xmax = dx*(nx-1)
    ymax = dy*(ny-1)
    xmin = 0.0
    ymin = 0.0
    print("xmax, ymax, dx, dy : ", xmax, ymax, dx, dy)
    base_xy = np.zeros((nx, ny)) 
    AOI_np_L = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            if i == 0 and j == 0:
                AOI_np_L[i][j] = 1
            else:
                AOI_np_L[i][j] = 1
    return nx, ny, area, dx, dy, xmax, ymax, xmin, ymin, base_xy, AOI_np_L


def make_rift_maps_for1D(
                            input1D_dict, input_path, base_xy, 
                            nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                            AOI_np_L
):
    map_names = [
                 "nphases", 
                 "start_age_rift1", 
                 "start_age_rift2",
                 "start_age_rift3",
                 "end_age_rift1", 
                 "end_age_rift2",
                 "end_age_rift2",
                 "mantlefac_rift1",
                 "mantlefac_rift2",
                 "mantlefac_rift3",
                 "riftmag_rift1",
                 "riftmag_rift2",
                 "riftmag_rift3",
                 "background_hf", 
                 "hf_reduc_fac"
                 ] 
    nphases = 1 # input1D_dict["nevents"]
    # If 2 events are added make sure to update 
    start_age_rift1 = input1D_dict["ev1_start"]
    start_age_rift2 = input1D_dict["ev1_start"]
    start_age_rift3 = input1D_dict["ev1_start"]
    end_age_rift1 = input1D_dict["ev1_end"]
    end_age_rift2 = input1D_dict["ev1_end"]
    end_age_rift2 = input1D_dict["ev1_end"]
    mantlefac_rift1 = 1
    mantlefac_rift2 = 1
    mantlefac_rift3 = 1
    riftmag_rift1 = 1
    riftmag_rift2 = 1
    riftmag_rift3 = 1
    background_hf = input1D_dict["bghf"]
    hf_reduc_fac = input1D_dict["ev1_hf_reduc"]    
    map_vals = [nphases, start_age_rift1, start_age_rift2, 
                start_age_rift3, end_age_rift1,
                end_age_rift2, end_age_rift2,
                mantlefac_rift1, mantlefac_rift2, 
                mantlefac_rift3,
                riftmag_rift1, riftmag_rift2,
                riftmag_rift3, background_hf, hf_reduc_fac]
    for i, mname in enumerate(map_names):
        fname = mname
        val = map_vals[i]
        output_zmap(input_path, base_xy, fname, val, 
                    nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np_L)
    return map_names


def make_event_maps_and_append_list(
                                    evL, i, age, Lname, evt_type, input_path,
                                    int_veloc, dSL, itop, swit,
                                    base_xy, nx, ny, dx, dy, 
                                    xmin, xmax, ymin, ymax, AOI_np_L, 
                                    depth_tvd, shift, pwd, depth_twt, lith_ID,
                                    pse
):
    tvd_map_name = "top_tvd_" + str(i+1)
    if evt_type == "Erosion_and_Deposition":
        shift_map_name = "shift_ero_" + str(i+1)
        pwd_map_name = "pwd_ero_" + str(i+1)
    else:
        shift_map_name = "shift_depo_" + str(i+1)
        pwd_map_name = "pwd_depo_" + str(i+1)
    twt_map_name = "top_twt_" + str(i+1)
    lith_map_name = "lith_" + str(i+1)
    
    output_zmap(input_path, base_xy, tvd_map_name, depth_tvd, 
                nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np_L)
    output_zmap(input_path, base_xy, shift_map_name, shift, 
                nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np_L)
    output_zmap(input_path, base_xy, pwd_map_name, pwd, 
                nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np_L)
    output_zmap(input_path, base_xy, twt_map_name, depth_twt, 
                nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np_L)
    output_zmap(input_path, base_xy, lith_map_name, lith_ID,
                nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np_L)
    evL.append(
               [
                age, # 0
                Lname, # 1
                evt_type, # 2
                tvd_map_name+".dat", # 3
                depth_tvd, # 4
                shift_map_name + ".dat", # 5
                shift, # 6
                pwd_map_name + ".dat", # 7
                pwd, # 8
                twt_map_name + ".dat", # 9
                depth_twt, # 10
                lith_map_name + ".dat", # 11
                lith_ID, # 12
                int_veloc, # 13
                dSL, # 14
                itop, # 15
                swit, # 16
                pse # 17
                ]
                )
            
            
def make_event_list_and_maps_for1D(
                                    input1D_dict, input_path, base_xy, 
                                    nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                    AOI_np_L
):
    # Make event list
    evL = []    
    # loop over all layers and define all events including erosion
    for i in range(13):        
        Lname = input1D_dict["layer_name"][i]
        idepth = input1D_dict["idepth"]
        if idepth == 0:
            depth_tvd = input1D_dict["top_depths"][i]
            depth_twt = 0.0
        else:
            depth_tvd = 0.0
            depth_twt = input1D_dict["top_depths"][i]            
        depo_end = input1D_dict["depo_end"][i]
        ero_end = input1D_dict["ero_end"][i]
        int_veloc = input1D_dict["int_veloc"][i]        
        pwd_depo = input1D_dict["pwd_depo"][i]
        pwd_ero = input1D_dict["pwd_ero"][i]        
        dSL_depo = input1D_dict["dSL_depo"][i]
        dSL_ero = input1D_dict["dSL_ero"][i]
        swit_depo = input1D_dict["swit_depo"][i]
        swit_ero = input1D_dict["swit_ero"][i]        
        ero_thick = input1D_dict["ero_thick"][i]
        pse = input1D_dict["pse"][i]
        if pse in ["Source (Type I)", "Source (Type II)", 
                   "Source (Type III)", "Source (Type IIS)"]:
            pse = "Source"
        else:
            pse = ""            
        lith_ID = i + 1
        itop = i
        if ero_end == depo_end:
            evt_type = "Deposition"
            age = depo_end
            dSL = dSL_depo
            pwd = pwd_depo
            swit = swit_depo
            shift = 0.0
            swit = swit_depo
            make_event_maps_and_append_list(
                                    evL, i, age, Lname, evt_type, input_path,
                                    int_veloc, dSL, itop, swit,
                                    base_xy, nx, ny, dx, dy, 
                                    xmin, xmax, ymin, ymax, AOI_np_L, 
                                    depth_tvd, shift, pwd, depth_twt, lith_ID,
                                    pse
                                    )
        else:
            if i != 12:
                evt_type = "Erosion_and_Deposition"
                age = ero_end
                dSL = dSL_ero
                pwd = pwd_ero
                swit = swit_ero
                shift = 0.0
                make_event_maps_and_append_list(
                                    evL, i, age, Lname, evt_type, input_path,
                                    int_veloc, dSL, itop, swit,
                                    base_xy, nx, ny, dx, dy, 
                                    xmin, xmax, ymin, ymax, AOI_np_L, 
                                    depth_tvd, shift, pwd, depth_twt, lith_ID,
                                    pse
                                    )
            evt_type = "Deposition"            
            age = depo_end
            dSL = dSL_depo
            pwd = pwd_depo
            swit = swit_depo            
            shift = ero_thick
            make_event_maps_and_append_list(
                                    evL, i, age, Lname, evt_type, input_path,
                                    int_veloc, dSL, itop, swit,
                                    base_xy, nx, ny, dx, dy, 
                                    xmin, xmax, ymin, ymax, AOI_np_L, 
                                    depth_tvd, shift, pwd, depth_twt, lith_ID,
                                    pse
                                    )
    return evL
 
               
def initialize_laoded_maps_and_cal_files_for1D(input_path):
    file_name = input_path + "ipa_loaded_maps.csv"
    fout = open(file_name, 'w')
    fout.close()
    file_name = input_path + "ipa_calibration.csv"
    fout = open(file_name, 'w')
    fout.close()   
    

def make_ipa_input_csv_for1D(input1D_dict, input_path, evL, map_names):
    param_names = [
       "idepth_model",
       "niter_t2d_comp",
       "icompact",
       "dt_age_PWD",
       "age_start_PWD",
       "age_end_PWD",
       "inv_itype",
       "shift_fac",
       "TOL_delta_bisec",
       "iuse_numerical_rift",
       "itype_rho_a",
       "dt_Myr",
       "dt_rift_Myr",
       "dz_lith",
       "rift_itype",
       "tc_initial",
       "tm_initial",
       "rho_water",
       "rho_crust",
       "rho_mantle",
       "k_bulk",
       "cp_bulk",
       "alpha_bulk",
       "T_base",
       "HP_itype",
       "Q_crust",
       "Q_mantle",
       "Ao",
       "ar",
       "L_crust_ref",
       "L_lith_ref",
       "rho_crust_ref",
       "rho_mantle_ref",
       "iuse_trans",
       "kappa_lith",
       "iuse_temp_dep_k",
       "nrelax",
       "nsublayers",
       "iuse_anomalous_heatflow",
       "hf_reduc_fac",
       "k_water",
       "Q_water",
       "icalc_SWIT",
       "iupdate_PWD",
       "T_top_input",
       "lat_LL",
       "lon_LL",
       "splate",
       "inert_frac",
       "adsorption_perc",
       "ioutput_burial",
       "ioutput_FW_TTS",
       "ioutput_HFTOT",
       "ioutput_HF_ANOM",
       "ioutput_HF_ANOM",
       "icalc_temp",
       "icalc_LOM",
       "imass_gen",
       "isalt_restore",
       "salt_layer_index",
       "search_rad",
       "NotUsed1",
       "NotUsed2",
       "NotUsed3",
       "NotUsed4",
       "NotUsed5",
       "NotUsed6",
       "NotUsed7",
       "NotUsed8",
       "NotUsed9",
       "NotUsed10",
       "NotUsed11",
       "NotUsed12",
       "NotUsed13",
       "NotUsed14",
       "iuse_flexure",
       "temp_elastic",
       "dist_taper",
       "xth_file_name",
       "ioutput_TEMP",
       "ioutput_Ro",
       "ioutput_LOM",
       "ioutput_TR",
       "ioutput_mHC",
       "ioutput_mODG",
       "ioutput_mFG",
       "ioutput_SEC_mFG",
       "ioutput_EXPRATE",
       "ioutput_SEC_EXPRATE",
       "TRmin",
       "TRmax",
       "gas_frac_adj_TRmin",
       "gas_frac_adj_TRmax",
       "ahf_max",
       "iuse_high_lom_gas"
       ]

    idepth_model = input1D_dict["idepth"]
    if idepth_model == 0 or idepth_model == 1:
        niter_t2d_comp = 1
    else:
        niter_t2d_comp = 6
    icompact = 1
    dt_age_PWD = 5.0
    age_start_PWD= 60.0
    age_end_PWD = 0.0
    inv_ityp = -1
    shift_fac = 1.0
    TOL_delta_bisec = 1e-2
    iuse_numerical_rift = 0
    itype_rho_a = 1
    dt_Myr = 2.0
    dt_rift_Myr = 2.0
    dz_lith = 5000.0
    rift_itype = 1
    tc_initial = input1D_dict["crust_thick"]
    tm_initial = input1D_dict["lith_thick"] - tc_initial
    rho_water = input1D_dict["rho_sea"]
    rho_crust = input1D_dict["rho_crust"]
    rho_mantle = input1D_dict["rho_mantle"]
    k_bulk = input1D_dict["k_mantle"]
    cp_bulk = input1D_dict["cp_mantle"]
    alpha_bulk = input1D_dict["alpha_mantle"]
    T_base = input1D_dict["ev1_atemp"]
    HP_itype = 0
    Q_crust = 0
    Q_mantle = 0
    Ao = 0
    ar = 8400
    L_crust_ref = input1D_dict["crust_thick"]
    L_lith_ref = input1D_dict["lith_thick"]
    rho_crust_ref = input1D_dict["rho_crust"]
    rho_mantle_ref = input1D_dict["rho_mantle"]
    iuse_trans = input1D_dict["trans_type"]
    kappa_lith = input1D_dict["lith_dif"]
    iuse_temp_dep_k = input1D_dict["cond_type"]
    nrelax = 4
    nsublayers = 5
    iuse_anomalous_heatflow = 1
    hf_reduc_fac = "hf_reduc_fac.dat"
    k_water = input1D_dict["pore_k"]
    Q_water = input1D_dict["pore_hp"]
    icalc_SWIT = 0
    iupdate_PWD = 0
    T_top_input = 1
    lat_LL = input1D_dict["lat"]
    lon_LL = input1D_dict["lon"]
    splate = input1D_dict["plateID"]
    inert_frac = input1D_dict["inhert_frac"]
    adsorption_perc = input1D_dict["adsor"]
    ioutput_burial = 0
    ioutput_FW_TTS = 0
    ioutput_HFTOT = 0
    ioutput_HF_ANOM = 0
    icalc_temp = 1
    icalc_LOM = 1
    imass_gen = 1
    isalt_restore = 0
    salt_layer_index = 0
    search_rad = 20
    NotUsed1 = 0
    NotUsed2 = 0
    NotUsed3 = 0
    NotUsed4 = 0
    NotUsed5 = 0
    NotUsed6 = 0
    NotUsed7 = 0
    NotUsed8 = 0
    NotUsed9 = 0
    NotUsed10 = 0
    NotUsed11 = 0
    NotUsed12 = 0
    NotUsed13 = 0
    NotUsed14 = 0
    iuse_flexure = 0
    temp_elastic = 200
    dist_taper = 250
    xth_file_name = "None"
    ioutput_TEMP = 0
    ioutput_Ro = 0
    ioutput_LOM = 0
    ioutput_TR = 0
    ioutput_mHC = 0
    ioutput_mODG = 0
    ioutput_mFG = 0
    ioutput_SEC_mFG = 0
    ioutput_EXPRATE = 0
    ioutput_SEC_EXPRATE = 0
    TRmin = 0
    TRmax = 1
    gas_frac_adj_TRmin = 1
    gas_frac_adj_TRmax = 1
    ahf_max = 250
    iuse_high_lom_gas = 1        
    param_vals = [
        idepth_model, 
        niter_t2d_comp, 
        icompact, 
        dt_age_PWD,
        age_start_PWD, 
        age_end_PWD, 
        inv_ityp, 
        shift_fac, 
        TOL_delta_bisec, 
        iuse_numerical_rift,
        itype_rho_a, 
        dt_Myr, 
        dt_rift_Myr,
        dz_lith, 
        rift_itype, 
        tc_initial,
        tm_initial,
        rho_water, 
        rho_crust, 
        rho_mantle,
        k_bulk, 
        cp_bulk, 
        alpha_bulk, 
        T_base, 
        HP_itype,
        Q_crust,
        Q_mantle, 
        Ao, 
        ar, 
        L_crust_ref, 
        L_lith_ref, 
        rho_crust_ref,
        rho_mantle_ref,
        iuse_trans, 
        kappa_lith, 
        iuse_temp_dep_k, 
        nrelax,nsublayers,
        iuse_anomalous_heatflow, 
        hf_reduc_fac, 
        k_water, 
        Q_water,
        icalc_SWIT,
        iupdate_PWD, 
        T_top_input,
        lat_LL, 
        lon_LL,
        splate, 
        inert_frac, 
        adsorption_perc,
        ioutput_burial,
        ioutput_FW_TTS,
        ioutput_HFTOT,
        ioutput_HF_ANOM,
        ioutput_HF_ANOM,
        icalc_temp,
        icalc_LOM,
        imass_gen,
        isalt_restore,
        salt_layer_index,
        search_rad,
        NotUsed1,
        NotUsed2,
        NotUsed3,
        NotUsed4,
        NotUsed5,
        NotUsed6,
        NotUsed7,
        NotUsed8,
        NotUsed9,
        NotUsed10,
        NotUsed11,
        NotUsed12,
        NotUsed13,
        NotUsed14,
        iuse_flexure,
        temp_elastic,
        dist_taper,
        xth_file_name,
        ioutput_TEMP,
        ioutput_Ro,
        ioutput_LOM,
        ioutput_TR,
        ioutput_mHC,
        ioutput_mODG,
        ioutput_mFG,
        ioutput_SEC_mFG,
        ioutput_EXPRATE,
        ioutput_SEC_EXPRATE,
        TRmin,
        TRmax,
        gas_frac_adj_TRmin,
        gas_frac_adj_TRmax,
        ahf_max,
        iuse_high_lom_gas,
        ]    
    # ipa_input.csv
    file_name = input_path + "ipa_input.csv"
    fout = open(file_name, 'w')    
    ntops_total = len(evL)
    data = ["Ntops", ntops_total]
    str_out = ','.join(map(str, data)) + "\n"
    fout.write(str_out)   
    data = [
            "","Age (Ma)","Top Names", "Event Type", "TVD Maps (subsea: m)",
            "Erosion Shift Maps (m)", "Water Depth Maps (m)", "", 
            "TWT Maps (subsea: ms)",
            "Lithology ID Maps", "Interval Velocity Maps (m/s)",
            "Sea Level Change Maps (m)", "Petroleum Systems Element", 
            "Fetch Polygon ASCII File"
            ]    
    str_out = ','.join(map(str, data)) + "\n"
    fout.write(str_out)
    nevent = len(evL)
    for i in range(30):
        if i < nevent:
            data = ["",evL[i][0], evL[i][1], evL[i][2], evL[i][3], evL[i][5],
                    evL[i][7], "", evL[i][9], evL[i][11], evL[i][13], 
                    evL[i][14], evL[i][17], ""]
            str_out = ','.join(map(str, data)) + "\n"
            fout.write(str_out)
        else:
            data = ["","","","","","","","","","","","","",""]
            str_out = ','.join(map(str, data)) + "\n"
            fout.write(str_out)
    nmaps = len(map_names)
    for ii, name in enumerate(map_names):
        if ii < nmaps - 1:
            fname = name + ".dat"
            data = [name, fname]
            str_out = ','.join(map(str, data)) + "\n"
            fout.write(str_out)
    for i, vname in enumerate(param_names):
        data = [vname, param_vals[i]]
        str_out = ','.join(map(str, data)) + "\n"
        fout.write(str_out)
    fout.close()


def make_ipa_lithology_csv_for1D(input_path, lith1D_final_dict):
    # ipa_lithology.csv
    data = ["Name", "Matrix Velocity m/s","Matrix HP: microW/m3", 
            "Grain k: W/m/K",
            "Grain Cp: J/kg/K","Grain rho: kg/m3","Surface Porosity %",
            "Porosity Decay Depth (km)","d","f","Reference","ID","OMT","HI",
            "TOC","Kinetics","Source Thickness (m)","Not Used","OIL_API",
            "Gas_Grav", "Pore_Threshold_Frac","Gas_Frac_Scenario",
            "Polar_Frac_Scenario"
            ]
    file_name = input_path + "ipa_lithology.csv"
    fout = open(file_name, 'w')
    str_out = ','.join(map(str, data)) + "\n"
    fout.write(str_out)
    keys = lith1D_final_dict.keys()
    for key in keys:    
        rlist = lith1D_final_dict[key]    
        name = rlist[0]
        v = rlist[1]
        hp = rlist[2]
        k = rlist[3]
        cp = rlist[4]
        rho = rlist[5]
        phi = rlist[6]
        lam = rlist[7]
        d = rlist[8]
        f = rlist[9]
        ref = rlist[10]
        ID = rlist[11]
        OMT = rlist[12]
        HI = rlist[13]
        TOC = rlist[14]
        kinetics = rlist[15]
        thick = rlist[16]
        area = rlist[17]
        api = rlist[18]
        gas_grav = 0.8
        por_threshold = 0.06
        gas_frac_scenario = "base"
        polar_frac_scenario = "base"
        
        data = [name, v, hp, k, cp, rho, phi, lam, d, f, ref, 
                ID, OMT, HI, TOC, kinetics, thick, area, api,
                gas_grav, por_threshold, gas_frac_scenario, 
                polar_frac_scenario]
        str_out = ','.join(map(str, data)) + "\n"
        fout.write(str_out)
    fout.close()

    
def make_multi1D_input_files(
                                input_path, output_path, input1D_dict,
                                lith1D_dict, rockID_dict, lith1D_final_dict
):
    (
        nx, ny, area, dx, dy, 
        xmax, ymax, xmin, ymin,
        base_xy, AOI_np_L
    ) = define_basic_geometry_for1D(input1D_dict)
    make_ipa_wells_file_for1D(input_path, xmax, ymax)
    initialize_laoded_maps_and_cal_files_for1D(input_path)
    evL = make_event_list_and_maps_for1D(
                                        input1D_dict, input_path, base_xy, 
                                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                        AOI_np_L
                                        )
    map_names = make_rift_maps_for1D(
                                     input1D_dict, input_path, base_xy, 
                                     nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                     AOI_np_L
                                     )
    make_ipa_input_csv_for1D(input1D_dict, input_path, evL, map_names)
    make_ipa_lithology_csv_for1D(input_path, lith1D_final_dict)

    
def read_IPA1D_file_csv(input_file_path):
    input1D_dict = {}
    input1D_dict["layer_name"] = []
    input1D_dict["pse"] = []
    input1D_dict["liths"] = []
    input1D_dict["top_depths"] = []
    input1D_dict["int_veloc"] = []
    input1D_dict["depo_end"] = []
    input1D_dict["ero_thick"] = []
    input1D_dict["ero_end"] = []
    input1D_dict["pwd_depo"] = []
    input1D_dict["pwd_ero"] = []
    input1D_dict["dSL_depo"] = []
    input1D_dict["dSL_ero"] = []
    input1D_dict["swit_depo"] = []
    input1D_dict["swit_ero"] = []
    input1D_dict["hf_depo"] = []
    input1D_dict["hf_ero"] = []
    input1D_dict["HIo"] = []
    input1D_dict["TOCo"] = []
    input1D_dict["TypeI_kinetics"] = []
    input1D_dict["TypeII_kinetics"] = []
    input1D_dict["TypeIII_kinetics"] = []
    input1D_dict["TypeIIS_kinetics"] = []
    lith1D_dict = {}
    lith1D_final_dict = {}
    rockID_dict = {}
    idrock = 0
    with open(input_file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        # Loop through csv and define inputs
        for i, row in enumerate(readCSV):
            ie = i + 1
            # Make sure empty cells have a value of zero
            irow_len = len(row)
            for ic in range(irow_len):
                v = row[ic]
                if v == '':
                    row[ic] = '0'
            if ie == 2:
                input1D_dict["idepth"] = int(row[1])
            if ie == 3:
                input1D_dict["iveloc"] = int(row[1])
            if ie == 4:
                input1D_dict["start_trap1"] = float(row[1])                
            if ie == 5:
                input1D_dict["end_trap1"] = float(row[1]) 
            if ie == 6:
                input1D_dict["start_trap2"] = float(row[1])                
            if ie == 7:
                input1D_dict["end_trap2"] = float(row[1]) 
            if ie == 8:
                input1D_dict["bghf"] = float(row[27])
            if ie > 8 and ie < 23:
                if ie < 22:
                    # add to arrays
                    input1D_dict["layer_name"].append(row[0])
                    input1D_dict["pse"].append(row[1])
                    input1D_dict["liths"].append(row[2])
                    input1D_dict["top_depths"].append(float(row[3]))
                    #if ie < 21:
                    input1D_dict["int_veloc"].append(float(row[6]))
                    input1D_dict["depo_end"].append(float(row[11]))
                    #if ie < 21:
                    input1D_dict["ero_thick"].append(float(row[13]))
                    input1D_dict["ero_end"].append(float(row[15]))
                    input1D_dict["pwd_depo"].append(float(row[16]))
                    input1D_dict["pwd_ero"].append(float(row[17]))
                    input1D_dict["dSL_depo"].append(float(row[20]))
                    #if ie < 21:
                    input1D_dict["dSL_ero"].append(float(row[21]))
                    input1D_dict["swit_depo"].append(float(row[22]))
                    input1D_dict["swit_ero"].append(float(row[23]))
                    input1D_dict["hf_depo"].append(float(row[24]))
                    input1D_dict["hf_ero"].append(float(row[25]))
                # Define scalars
                if ie == 9:
                    input1D_dict["nevents"] = float(row[27])
                    input1D_dict["pore_veloc"] = float(row[34])
                    input1D_dict["pore_hp"] = float(row[35])
                    input1D_dict["pore_k"] = float(row[36])
                    input1D_dict["pore_rho"] = float(row[37])
                    input1D_dict["pore_cp"] = float(row[38])
                if ie == 10:
                    input1D_dict["def_type_ev1"] = int(row[27])
                    input1D_dict["trans_type"] = int(row[38])
                if ie == 11:
                    input1D_dict["ev1_start"] = float(row[27])
                    input1D_dict["cond_type"] = int(row[38])
                if ie == 12:
                    input1D_dict["ev1_end"] = float(row[27])
                    input1D_dict["crust_veloc"] = float(row[38])
                if ie == 13:
                    input1D_dict["ev1_delta"] = float(row[27])
                    input1D_dict["trans_fac"] = float(row[33])
                    input1D_dict["lith_thick"] = float(row[38])
                if ie == 14:
                    input1D_dict["ev1_beta"] = float(row[27])
                    input1D_dict["crust_thick"] = float(row[38])
                if ie == 15:
                    input1D_dict["ev1_atemp"] = float(row[27])
                    input1D_dict["biodeg"] = float(row[33])
                    input1D_dict["rho_crust"] = float(row[38])
                if ie == 16:
                    input1D_dict["ev2_delay"] = float(row[27])
                    input1D_dict["hf_limit"] = float(row[33])
                    input1D_dict["rho_sea"] = float(row[38])  
                if ie == 17:
                    input1D_dict["ev2_dur"] = float(row[27])
                    input1D_dict["ev1_hf_reduc"] = float(row[33])
                    input1D_dict["rho_mantle"] = float(row[38])
                if ie == 18:
                    input1D_dict["ev2_delta"] = float(row[27])
                    input1D_dict["ev2_hf_reduc"] = float(row[33])
                    input1D_dict["k_mantle"] = float(row[38])
                if ie == 19:
                    input1D_dict["ev2_beta"] = float(row[27])
                    input1D_dict["cp_mantle"] = float(row[38])
                if ie == 20:
                    input1D_dict["ev2_atemp"] = float(row[27])
                    input1D_dict["alpha_mantle"] = float(row[38])
                if ie == 21:
                    input1D_dict["tts_cor"] = float(row[27])
                    input1D_dict["rho_asth"] = float(row[38])
                if ie == 22:
                    input1D_dict["moho"] = float(row[3])
                    input1D_dict["lith_dif"] = float(row[33])
                    input1D_dict["sed_hp_reduc"] = float(row[38])            
            if ie == 39:
                input1D_dict["lat"] = float(row[1])
            if ie == 40:
                input1D_dict["lon"] = float(row[1])
            if ie == 41:
                input1D_dict["plateID"] = row[1]               
            if ie > 42 and ie < 93:               
                name = row[0]
                tmp_list = [
                            idrock, float(row[1]), 
                            float(row[2]), float(row[3]), 
                            float(row[4]), float(row[5]), float(row[6]), 
                            float(row[7]), float(row[8]), float(row[9])
                            ]
                if name not in ['0']:
                    lith1D_dict[name] = tmp_list
                    rockID_dict[name] = idrock
                    idrock = idrock + 1
            if ie == 873:
                input1D_dict["src_thick"] = float(row[1])
            if ie == 874:
                input1D_dict["src_area"] = float(row[1])
            if ie == 875:
                input1D_dict["oil_api"] = float(row[1])                
            if ie == 876:
                input1D_dict["adsor"] = float(row[1])*100.0     
            if ie == 877:
                input1D_dict["inhert_frac"] = float(row[1])            
            if ie > 878 and ie < 892:                
                input1D_dict["HIo"].append(float(row[2]))
                input1D_dict["TOCo"].append(float(row[3]))           
            if ie == 892:
                input1D_dict["kinetic_type"] = row[1]
            if ie == 894:
                input1D_dict["mix_type"] = row[1]            
            if ie == 895:
                input1D_dict["TypeI_fracs"] = [
                        float(row[1]), float(row[2]), 
                        float(row[3]), float(row[4])
                        ]              
            if ie == 896:
                input1D_dict["TypeII_fracs"] = [
                        float(row[1]), float(row[2]), 
                        float(row[3]), float(row[4])
                        ]
            if ie == 897:
                input1D_dict["TypeIII_fracs"] = [
                        float(row[1]), float(row[2]), 
                        float(row[3]), float(row[4])
                        ]
            if ie == 898:
                input1D_dict["TypeIIS_fracs"] = [
                        float(row[1]), float(row[2]), 
                        float(row[3]), float(row[4])
                        ]
            if ie == 899:
                input1D_dict["pre_expo"] = [
                        float(row[1]), float(row[2]), 
                        float(row[3]), float(row[4])
                        ]
            if ie > 900 and ie < 942:
                input1D_dict["TypeI_kinetics"].append(float(row[1]))
                input1D_dict["TypeII_kinetics"].append(float(row[2]))
                input1D_dict["TypeIII_kinetics"].append(float(row[3]))
                input1D_dict["TypeIIS_kinetics"].append(float(row[4]))           
    layer_names = input1D_dict["layer_name"]
    for i, lname in enumerate(layer_names):
        rname = input1D_dict["liths"][i]
        rlist = lith1D_dict[rname]

        ID = i+1
        v = rlist[1]
        hp = rlist[2]
        k = rlist[3]
        cp = rlist[4]
        rho = rlist[5]
        phi = rlist[6]
        lam = rlist[7]
        d = rlist[8]
        f = rlist[9]
        ref = rname
        
        pe = input1D_dict["pse"][i]
        if pe == "Source (Type I)":
            omt = "Type_I"
        elif pe == "Source (Type II)":
            omt = "Type_II"
        elif pe == "Source (Type III)":
            omt = "Type_III"
        elif pe == "Source (Type IIS)":
            omt = "Type_IIS"
        else:
            omt = "Type_II"
            
        HI = input1D_dict["HIo"][i]
        TOC = input1D_dict["TOCo"][i]
        
        kin = input1D_dict["kinetic_type"]
        print("kin : ", kin)
        if kin == "Middle":
            kinetics = "normal"
        elif kin == "Early":
            kinetics = "early"
        elif kin == "Late":
            kinetics = "late"
        thick = input1D_dict["src_thick"]
        oil_api = input1D_dict["oil_api"]
        area = input1D_dict["src_area"] 
        
        lith1D_final_dict[lname] = [
                lname, v, hp, k, cp, rho, phi, lam, d, f,
                ref, ID, omt, HI, TOC, kinetics, thick, area, oil_api
                ]
        
    return input1D_dict, lith1D_dict, rockID_dict, lith1D_final_dict


def output_zmap(
        direc_path, base_xy, file_name, val, 
        nx, ny, dx, dy, xmin, xmax, ymin,
        ymax, AOI_np_L
): 
    for i in range(nx):
        for j in range(ny):
            valf = val
            base_xy[i][j] = valf
    
    map_tools.make_output_file_ZMAP_v4(
            direc_path, file_name, base_xy, nx, ny, dx,
            dy, xmin, xmax, ymin, ymax, AOI_np_L
            )


def ipa1Dmain(path1D, input_path, output_path, csv1D_path):
    (
    input1D_dict, lith1D_dict, 
    rockID_dict, lith1D_final_dict
    ) = read_IPA1D_file_csv(csv1D_path)
    make_multi1D_input_files(
            input_path, output_path, input1D_dict,
            lith1D_dict, rockID_dict, lith1D_final_dict
            )
    
    return input1D_dict, lith1D_dict, rockID_dict, lith1D_final_dict


def extract_at_node_history_for_IPA1D(
                                        inode, jnode, naflag, output_path,
                                        model, rho_w, kw
):
    na_num = 0.0
    na_num_orig = na_num
    keys = list(model.event_dict_bs.keys())
    event_ID_ilast = len(keys)-1
    # Make paleo-water depth array
    pwd_xy_event_list = []
    # Loop over events from old to young
    for event_ID, key in enumerate(keys):
        s = model.event_dict_bs[event_ID][5][inode][jnode]
        pwd_xy_event_list.append(s)        
    file_name = output_path + "IPA1Doutput.csv"
    fout = open(file_name, 'w')   
    # Event specific output
    data_itypes = [
                    0, 
                    3, 
                    5, 
                    13, 
                    0,
                    8, 
                    14, 
                    9, 
                    15, 
                    12
                   ]
    stype_list = [
                    "Age (Ma)", 
                    "Erosion (m)", 
                    "PWD (m)", 
                    "PWD Correction (m)", 
                    "dSL (m)",
                    "TTS (m)", 
                    "Forward TTS(m)", 
                    "Total HF mW/m2", 
                    "Anomalous HF mW/m2", 
                    "Bulk Sediment Density (kg/m3)"
                ]
    for im, itype in enumerate(data_itypes):
        stype = stype_list[im]
        data = [stype]
        # Loop over events from old to young
        for event_ID, key in enumerate(keys):
            event_type = model.event_dict_bs[event_ID][1]
            itop_event = model.event_dict_bs[event_ID][2]
            if itype not in [0]:
                s1 = model.event_dict_bs[event_ID][itype][inode][jnode]
            else:
                if stype == "Age (Ma)":
                    s1 = model.event_dict_bs[event_ID][itype]
                elif stype == "dSL (m)":
                    s1 = -model.deltaSL_list[event_ID]
            if event_ID < event_ID_ilast:
                event_type_next = model.event_dict_bs[event_ID+1][1]
                if event_type_next != "Deposition":
                    if itype not in [0]:
                        s2 = model.event_dict_bs[event_ID + 1][itype]\
                                                                [inode][jnode]
                    else:
                        if stype == "Age (Ma)":
                            s2 = model.event_dict_bs[event_ID+1][itype]
                        elif stype == "dSL (m)":
                            s2 = -model.deltaSL_list[event_ID+1]
                else:
                    s2 = s1
            else:
                s2 = s1
            if event_type == "Deposition":
                if event_ID > 0:
                    data.append(s1)
                    data.append(s2)
                else:
                    data.append(s1)
        str_out = ','.join(map(str, data)) + "\n"
        fout.write(str_out)
    # Top specific output
    data_itypes = [
                    38, 
                    37, 
                    39, 
                    40, 
                    1,
                    2, 
                    1, 
                    41, 
                    54,
                    55, 
                    56, 
                    57, 
                    42,
                    44, 
                    43,
                    45, 
                    52,
                    53,
                    60,
                    61,
                    58,
                    59,
                    62, 
                    3,
                    63, 
                    64,
                    65,
                    66,
                    66,
                    68,
                    69,
                    1, 
                    1,
                    1,
                    1, 
                    1
                ]
    stype_list = [
        "Transient T(C)", 
        "Steady State T(C)", 
        "Ro", 
        "LOM", 
        "Burial (TVDsmkm)", 
        "Thickness (km)", 
        "Burial (TVDsskm)", 
        "TR", 
        "rho_g (g/cm3)",
        "rho_l (g/cm3)", 
        "Bg (rcf/scf)", 
        "Bo oil (rbbl/stb)",
        "HC Gen (mgHC/gOC)",
        "Free Gas Pore Expulsion (mgHC/gOC)", 
        "Liquid Pore Expulsion (mgHC/gOC)",
        "Expulsion rate (mgHC/gOC/Myr)", 
        "Volume of gas expelled from SRC(surface conditions, Tcf)",
        "Volume of liquid expelled from SRC (surface conditions, GOB)", 
        "rho_g_sec (g/cm3)", 
        "Bg (rcf/scf)", 
        "Secondary Free Gas Pore Expulsion (mgHC/gOC)",
        "Volume of secondary gas expelled from SRC (surface conditions, Tcf)", 
        "Secondary Expulsion rate (mgHC/gOC/Myr)",
        "Erosion (m)",
        "Primary Gas Gen. (g)",
        "Primary Oil Gen. (g)",
        "Primary Coke Gen. (g)",
        "Secondary mass generation (mgHC/gOC/Myr)", 
        "Secondary Gas Gen. (g)",
        "Mass of initial organic carbon (primary) (g)",
        "Mass of initial organic carbon (secondary) (g)",
        "Porosity (frac)", 
        "Thermal Conductivity (W/m/K)", 
        "Sed. Density (kg/m3)",
        "Interval Velocity From Compaction (m/s)", 
        "Two-way-time (ms)"
    ]             
    con_fac = 1.0
    for im, itype_data in enumerate(data_itypes):
        stype = stype_list[im]
        str_out = stype + "\n"
        fout.write(str_out)
        keys = list(model.event_dict_bs.keys())
        ntops = len(model.tops_list_bs)
        for mm in range(ntops): # Loop over tops from young to old
            jj = ntops-1-mm
            # Name
            name_top = model.tops_list_bs[jj][6]
            icheck_erosion = 0
            event_type_top = model.event_dict_bs[jj][1]
            #print("name_top, event_type_top : ", name_top, event_type_top)
            if event_type_top != "Deposition":
                icheck_erosion = 1
            if icheck_erosion == 0:
                data = [name_top]
                event_ID_ilast = len(keys)-1
                 # Loop over events from old to young
                for event_ID, key in enumerate(keys):
                    event_type = model.event_dict_bs[event_ID][1]
                    itop_event = model.event_dict_bs[event_ID][2]
                    base_level = -model.deltaSL_list[event_ID]
                    pwd = pwd_xy_event_list[event_ID]
                    # Use water depth for na_num if data type is subsea depth
                    if stype in ["Burial (TVDsskm)"]:
                        na_num_1 = pwd-base_level
                        if event_ID < event_ID_ilast:
                            (
                                event_type_next
                            ) = model.event_dict_bs[event_ID+1][1]
                            if event_type_next != "Deposition":
                                base_level2 = -model.deltaSL_list[event_ID+1]
                                pwd2 = pwd_xy_event_list[event_ID+1]
                            else:
                                base_level2 = base_level
                                pwd2 = pwd
                            na_num_2 = pwd2-base_level2
                        else:
                            na_num_2 = na_num_1
                    else:
                        na_num_1 = na_num_orig
                        na_num_2 = na_num_orig
                    if stype in ["Burial (TVDsmkm)", 
                                 "Burial (TVDsskm)","Thickness (km)"]:
                        con_fac = 1e-3
                    else:
                        con_fac = 1.0
                    if jj <= itop_event:
                        event_index = model.tops_list_bs[jj][14][event_ID]
                        s1 = model.tops_list_bs[jj][itype_data]\
                                                    [event_index][inode][jnode]
                        base_level = -model.deltaSL_list[event_ID]
                        pwd = pwd_xy_event_list[event_ID]
                        twt = model.tops_list_bs[jj][18][inode][jnode]
                        vint = model.tops_list_bs[jj][24][inode][jnode]
                        rho_grain1 = model.tops_list_bs[jj][32][inode][jnode]
                        # W/m/K
                        k_grain1 = model.tops_list_bs[jj][30][inode][jnode]
                        # Matrix HP: microW/m^3 converted to W/m/m/m
                        #(
                        #    Q_grain1
                        #) = model.tops_list_bs[jj][29][inode][jnode]/1e6
                        # Surface Porosity % converted to fraction
                        phi_o1 = model.tops_list_bs[jj][33][inode][jnode]/100.0
                        # Porosity Decay Depth (km) converted to 1/m
                        (
                            c1
                        ) = 1.0/(model.tops_list_bs[jj][34][inode][jnode]*1000)
                         # maximum forward burial in meters
                        maxfb1 = model.tops_list_bs[jj][46]\
                                                    [event_index][inode][jnode]
                        #z_top_subsea1 = model.tops_list_bs[jj][1]\
                        #                           [event_index][inode][jnode]
                        #z_surf1 = -base_level + pwd
                        # submud depth of layer top in meters
                        #z_top1 = z_top_subsea1-z_surf1
                        phi1 = phi_o1*math.exp(-maxfb1*c1)
                        k1 = (k_grain1**(1-phi1))*(kw**(phi1))
                        rho1 = phi1*rho_w + (1-phi1)*rho_grain1
                        if event_ID < event_ID_ilast:
                            (
                                event_index_next
                            ) = model.tops_list_bs[jj][14][event_ID+1]
                            (
                                event_type_next
                            ) = model.event_dict_bs[event_ID+1][1]
                            if event_type_next != "Deposition":
                                s2 = model.tops_list_bs[jj][itype_data]\
                                               [event_index_next][inode][jnode]
                                base_level2 = -model.deltaSL_list[event_ID+1]
                                pwd2 = pwd_xy_event_list[event_ID+1]
                                (
                                    rho_grain2
                                ) = model.tops_list_bs[jj][32][inode][jnode]
                                # W/m/K
                                (
                                    k_grain2
                                ) = model.tops_list_bs[jj][30][inode][jnode]
                                # Matrix HP: microW/m^3 converted to W/m/m/m
                                #Q_grain2 = model.tops_list_bs[jj][29]\
                                #                            [inode][jnode]/1e6
                                # Surface Porosity % converted to fraction
                                phi_o2 = model.tops_list_bs[jj][33]\
                                                           [inode][jnode]/100.0
                                # Porosity Decay Depth (km) converted to 1/m
                                c2 = 1.0/(model.tops_list_bs[jj][34]\
                                                           [inode][jnode]*1000)
                                 # maximum forward burial in meters
                                maxfb2 = model.tops_list_bs[jj][46]\
                                            [event_index_next][inode][jnode]
                                #z_top_subsea2 = model.tops_list_bs[jj][1]\
                                #            [event_index_next][inode][jnode]
                                #z_surf2 = -base_level2 + pwd2
                                # submud depth of layer top in meters
                                #z_top2 = z_top_subsea2-z_surf2
                                phi2 = phi_o2*math.exp(-maxfb2*c2)
                                k2 = (k_grain2**(1-phi2))*(kw**(phi2))
                                rho2 = phi2*rho_w + (1-phi2)*rho_grain2
                                
                            else:
                                s2 = s1
                                base_level2 = base_level
                                pwd2 = pwd
                                # W/m/K
                                k_grain2 = k_grain1
                                # Matrix HP: microW/m^3 converted to W/m/m/m
                                #Q_grain2 = Q_grain1
                                # Surface Porosity % converted to fraction
                                phi_o2 = phi_o1
                                # Porosity Decay Depth (km) converted to 1/m
                                c2 = c1
                                # maximum forward burial in meters
                                maxfb2 = maxfb1
                                #z_top_subsea2 = z_top_subsea1
                                #z_surf2 = z_surf1
                                #z_top2 = z_top1                                
                                phi2 = phi1
                                k2 = k1
                                rho2 = rho1
                        else:
                            s2 = s1
                            base_level2 = base_level
                            pwd2 = pwd
                             # W/m/K
                            k_grain2 = k_grain1
                            # Matrix HP: microW/m^3 converted to W/m/m/m
                            #Q_grain2 = Q_grain1
                            # Surface Porosity % converted to fraction
                            phi_o2 = phi_o1
                            # Porosity Decay Depth (km) converted to 1/m
                            c2 = c1
                            # maximum forward burial in meters
                            maxfb2 = maxfb1
                            #z_top_subsea2 = z_top_subsea1
                            #z_surf2 = z_surf1
                            #z_top2 = z_top1
                            phi2 = phi1
                            k2 = k1
                            rho2 = rho1
                        if stype == "Burial (TVDsmkm)":
                            s1 = s1-pwd+base_level
                            s2 = s2-pwd2+base_level2
                        elif stype == "Porosity (frac)":  
                            s1 = phi1
                            s2 = phi2
                        elif stype == "Thermal Conductivity (W/m/K)":  
                            s1 = k1
                            s2 = k2
                        elif stype == "Sed. Density (kg/m3)":  
                            s1 = rho1
                            s2 = rho2
                        elif stype == (
                                "Interval Velocity From Compaction (m/s)"
                        ):  
                            s1 = vint
                            s2 = vint
                        elif stype == "Two-way-time (ms)":  
                            s1 = twt
                            s2 = twt
                    else:
                        s1 = na_num_1
                        s2 = na_num_2
                    if event_type == "Deposition":
                        if event_ID > 0:
                            data.append(s1*con_fac)
                            data.append(s2*con_fac)
                        else:
                            data.append(s1*con_fac)
                str_out = ','.join(map(str, data)) + "\n"
                fout.write(str_out)
    fout.close()