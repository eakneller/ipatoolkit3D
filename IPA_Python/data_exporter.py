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
import time
import map_tools
import numpy as np
import fileIO
import source_kinetics
import print_funcs
        
        
def export_at_well_locations(model, ioutput_main, process, iextract_at_wells):
    output_path = model.output_path
    Lx = model.Lx
    Ly = model.Ly
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np
    imass_gen = model.imass_gen
    icalc_LOM = model.icalc_LOM
    icalc_temp = model.icalc_temp
    tt1 = time.time()
    if iextract_at_wells == 1 and model.itype3D == 1:
        input_file_path =  os.path.join(model.input_path, "ipa_wells.csv")
        nwells, wells_dict = fileIO.read_well_file_csv(input_file_path)
        wkeys = list(wells_dict.keys())
        for key in wkeys:
            well_name = key
            xx = wells_dict[key][0]
            yy = wells_dict[key][1]
            naflag = -99999.0
            map_tools.extract_at_xy_at_0Ma(
                                    well_name, xx, yy, naflag, output_path, 
                                    model.tops_list_bs, model.event_dict_bs, 
                                    model.deltaSL_list, 
                                    Lx, Ly, nx, ny, dx, dy, 
                                    xmin, xmax, ymin, ymax, 
                                    AOI_np, imass_gen, icalc_LOM, icalc_temp
                                )
            itype_list = [
                            1, 38, 39, 40, 41, 42, 45, 52, 53, 54, 55, 56, 
                            57, 58, 59, 60, 61, 62
                        ]
            if model.imass_gen > 0 and model.icalc_temp == 1:
                for itype_data in itype_list:
                    map_tools.extract_at_xy_history(
                                                itype_data, well_name, xx, yy, 
                                                naflag, output_path, 
                                                model.tops_list_bs, 
                                                model.event_dict_bs, 
                                                model.deltaSL_list, Lx, Ly, nx, 
                                                ny, dx, dy, xmin, 
                                                xmax, ymin, ymax, AOI_np
                                            )
    tt2 = time.time()
    print_funcs.print_finfo(
                            ioutput_main, process, 
                            "Extracted information at well locations", tt2-tt1)  
                

def export_flexural_baselevel_maps(
                                    model, ioutput, process,
                                    iterID, event_age, event_ID,
                                    AOI_clean_np, delta_q_BL, pwd_n_xy
):
    output_path = model.output_path
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np
    if ioutput == 1:
        tt1 = time.time()                                            
        my_file_name = "Flex_BL_Load_MPa_Age_"+str(event_age)
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, my_file_name, 
                                        delta_q_BL,
                                        nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_clean_np
                                    )
        file_name = ("PWD_flex_wBL_iter"+str(iterID)+"_"
                     +str(event_age)+"_event_"+str(event_ID))
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, 
                                        pwd_n_xy, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax,
                                        AOI_np
                                    )
        tt2 = time.time()
        print_funcs.print_finfo(1, process, 
                    "----> Exported BL load and PWD flex wBL maps", tt2-tt1)


def export_debug_bsmt_flex(
                                    model, ioutput, event_age, 
                                    event_ID, iterID, process,
                                    bsmt_next_np, w_inc_next_np,
                                    delta_crust_np, pwd_next_np,
                                    sed_thick_next_np
                                    
):
    output_path = model.output_path
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np
    if ioutput == 1:
        tt1 = time.time()
        file_name = ("DEBUG_bsmt_next_np_flex_iter"+str(iterID)+"_"
                     +str(event_age)+"_event_"+str(event_ID))
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, 
                                        bsmt_next_np, nx,
                                        ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np
                                    )
        file_name = ("DEBUG_w_inc_next_flex_iter"+str(iterID)
                    +"_"+str(event_age)+"_event_"+str(event_ID))
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, 
                                        w_inc_next_np, nx,
                                        ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np
                                    )
        file_name = ("DEBUG_delta_crust_np_flex_iter"+str(iterID)
                    +"_"+str(event_age)+"_event_"+str(event_ID))
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, 
                                        delta_crust_np, nx,
                                        ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np
                                    )
        file_name = ("DEBUG_pwd_next_np_flex_iter"+str(iterID)
                    +"_"+str(event_age)+"_event_"+str(event_ID))
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, 
                                        pwd_next_np, nx,
                                        ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np
                                    )
        file_name = ("DEBUG_psed_thick_next_np_flex_iter"+str(iterID)
                    +"_"+str(event_age)+"_event_"+str(event_ID))
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, 
                                        sed_thick_next_np, nx,
                                        ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np
                                    )
        tt2 = time.time()
        print_funcs.print_finfo(
                1, process, 
                "----> Exported BSMT Debug flex maps", tt2-tt1)    

                    
def export_restored_bsmt_and_pwd_iter(
                                    model, ioutput, event_age, 
                                    event_ID, iterID, process,
                                    bsmt_new_xy, pwd_flex_xy
):
    output_path = model.output_path
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np
    if ioutput == 1:
        tt1 = time.time()
        file_name = ("Flex_BSMT_iter"+str(iterID)+"_"
                     +str(event_age)+"_event_"+str(event_ID))
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, 
                                        bsmt_new_xy, nx,
                                        ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np
                                    )
        file_name = ("PWD_flex_WL_iter"+str(iterID)
                    +"_"+str(event_age)+"_event_"+str(event_ID))
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, 
                                        pwd_flex_xy, nx,
                                        ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np
                                    )
        tt2 = time.time()
        print_funcs.print_finfo(
                1, process, 
                "----> Exported BSMT and PWD flex WL maps", tt2-tt1)    


def export_degug_q_uc(
                        model, ioutput, process,
                        iterID, event_age, event_ID,
                        AOI_clean_np,
                        q_uc_np, q_uc_prev_np,
                        delta_q_uc
):
    output_path = model.output_path
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    #AOI_np = model.AOI_np
    
    if ioutput == 1:
        tt1 = time.time()
        my_file_name = "DEBUG_q_uc_np_Age_iter" + str(event_age)
        map_tools.make_output_file_ZMAP_v4(
                                         output_path, my_file_name, 
                                         q_uc_np,
                                         nx, ny, dx, dy, 
                                         xmin, xmax, ymin, ymax, 
                                         AOI_clean_np
                                        )
        my_file_name = "DEBUG_q_uc_prev_np_Age_"+str(event_age)
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, my_file_name, 
                                        q_uc_prev_np,
                                        nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_clean_np
                                        )
        my_file_name = ("DEBUG_delta_q_uc_Age_"+str(event_age))
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, my_file_name,
                                        delta_q_uc, nx, ny, 
                                        dx, dy, xmin, xmax, 
                                        ymin, ymax, AOI_clean_np
                                        )
        tt2 = time.time()
        print_funcs.print_finfo(
                            1, process, 
                            "----> Exported q_uc Debug maps", tt2-tt1)
                    
                    
def export_load_and_incremental_flex_maps(
                                        model, ioutput, process,
                                        iterID, event_age, event_ID,
                                        AOI_clean_np, AOI_tmp_np,
                                        delta_q_sed_xy, delta_q_uc_xy, 
                                        delta_q_thermal_xy, Te_map_xy, w_xy, 
                                        Flex_w_xy, xn_exp, yn_exp, Te_map,
                                        q_thermal
):
    output_path = model.output_path
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np
    
    if ioutput == 1:
        tt1 = time.time()
        my_file_name = (
                             "Flex_delta_Sed_Load_MPa_Age_iter" 
                            + str(iterID) 
                            + "_" 
                            + str(event_age)
                        )
        map_tools.make_output_file_ZMAP_v4(
                                         output_path, my_file_name, 
                                         delta_q_sed_xy,
                                         nx, ny, dx, dy, 
                                         xmin, xmax, ymin, ymax, 
                                         AOI_clean_np
                                        )
        my_file_name = "Flex_delta_UC_Load_MPa_Age_"+str(event_age)
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, my_file_name, 
                                        delta_q_uc_xy,
                                        nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_clean_np
                                    )
        my_file_name = ("Flex_delta_Thermal_Load_MPa_Age_"
                        +str(event_age))
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, my_file_name,
                                        delta_q_thermal_xy, nx, ny, 
                                        dx, dy, xmin, xmax, 
                                        ymin, ymax, AOI_clean_np
                                    )
        my_file_name = ("Flex_Thermal_Load_orig_MPa_Age_"
                        +str(event_age))
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, my_file_name,
                                        q_thermal, nx, ny, 
                                        dx, dy, xmin, xmax, 
                                        ymin, ymax, AOI_clean_np
                                    )
        my_file_name = "Flex_Te_clean_Age_"+str(event_age)
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, my_file_name,
                                        Te_map_xy, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_clean_np
                                    )
        my_file_name = "Flex_Te_orig_Age_"+str(event_age)
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, my_file_name,
                                        Te_map, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_clean_np
                                    )
        tt2 = time.time()
        print_funcs.print_finfo(
                            1, process, 
                            "----> Exported Te and load maps", tt2-tt1)
        tt1 = time.time()
        file_name = (
                          "Flex_w_inc_iter"
                        + str(iterID)
                        + "_"+str(event_age) 
                        + "_event_"+str(event_ID)
                    )
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, 
                                        w_xy, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np
                                    )
        file_name = (
                    "Flex_Flex_w_inc_iter" 
                    + str(iterID) 
                    + "_" 
                    + str(event_age) 
                    + "_event_"+str(event_ID)
                )
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, 
                                        Flex_w_xy, xn_exp,
                                        yn_exp, dx, dy, 0, xn_exp*dx, 
                                        0, yn_exp*dy, AOI_tmp_np
                                    )
        tt2 = time.time()
        print_funcs.print_finfo(
                    1, process, 
                    "----> Exported incremental flex maps", tt2-tt1)

        
def export_interpolated_PWD(
                            PWD_age, output_path, pwd_new_xy,
                            nx, ny, dx, dy, xmin, xmax, 
                            ymin, ymax, AOI_np
):
    file_name = "PWD_Interp_Age_" + str(PWD_age)
    map_tools.make_output_file_ZMAP_v4(
                                    output_path, file_name, pwd_new_xy,
                                    nx, ny, dx, dy, xmin, xmax, 
                                    ymin, ymax, AOI_np
                                )   

    
def export_sed_load_and_tts(imake_output, event_age, key, model, q_sed_xy, 
                            ttsub_xy):
    # Unpack model object
    iuse_flexure = model.iuse_flexure
    ioutput_TTS = model.ioutput_TTS
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np    
    output_path = model.output_path
    
    if imake_output == 1:
        if iuse_flexure == 1:
            my_file_name = "Flex_Sed_Load_MPa_Age_"+str(event_age)
            map_tools.make_output_file_ZMAP_v4(
                                            output_path, my_file_name, 
                                            q_sed_xy,
                                            nx, ny, dx, dy, 
                                            xmin, xmax, ymin, ymax, AOI_np
                                        )        
        if ioutput_TTS == 1:
            file_name = "TTS_"+str(event_age)+"_event_"+str(key)
            map_tools.make_output_file_ZMAP_v4(
                                            output_path, file_name, 
                                            ttsub_xy, 
                                            nx, ny, dx, dy, 
                                            xmin, xmax, ymin, ymax, AOI_np
                                        )    

    
def export_inverted_pwd_and_residual_sub(model, event_age, 
                                         pwd_new_xy, res_sub_xy):
    # Unpack model objects
    output_path = model.output_path
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np
    rad_search_m = model.rad_search_m
    
    file_name = (
                    "PWD_LocalIsostasy_Age_LPF_" 
                    + str(int(rad_search_m/1000)) 
                    + "km_"+str(event_age) 
                )            
    map_tools.make_output_file_ZMAP_v4(
                                output_path, file_name, pwd_new_xy,
                                nx, ny, dx, dy, xmin, 
                                xmax, ymin, ymax, AOI_np
                            )            
    file_name = "ResidualSub_Age_" + str(event_age)            
    map_tools.make_output_file_ZMAP_v4(
                                output_path, file_name, res_sub_xy,
                                nx, ny, dx, dy, 
                                xmin, xmax, ymin, ymax, AOI_np
                            )

            
def export_sub_and_hf_maps(model, ioutput_main, process):
    if model.itype3D == 1:
        output_path = model.output_path
        nx = model.nx
        ny = model.ny
        dx = model.dx
        dy = model.dy
        xmin = model.xmin
        xmax = model.xmax
        ymin = model.ymin
        ymax = model.ymax
        AOI_np = model.AOI_np    
        iuse_numerical_rift = model.iuse_numerical_rift
        hf_reduc_fac_xy = model.hf_reduc_fac_xy
        bghf_xy = model.bghf_xy
        ioutput_FW_TTS = model.ioutput_FW_TTS
        ioutput_HFTOT = model.ioutput_HFTOT
        ioutput_HF_ANOM = model.ioutput_HF_ANOM
                                
        tt1 = time.time()
        event_IDs = model.event_dict_bs.keys()
        for event_ID in event_IDs:
            age = model.event_dict_bs[event_ID][0]
            ttsub_xy = np.copy(model.event_dict_bs[event_ID][14])
            hf_xy = np.copy(model.event_dict_bs[event_ID][9])
            hf_anom_xy = np.copy(model.event_dict_bs[event_ID][15])
            q_crustal_xy = np.copy(model.event_dict_bs[event_ID][16])
            q_thermal_xy = np.copy(model.event_dict_bs[event_ID][17])
            Te_xy = np.copy(model.event_dict_bs[event_ID][19])
            if iuse_numerical_rift == 0:
                my_file_name = "Flex_Crustal_Load_MPa_Age_"+str(age)
                map_tools.make_output_file_ZMAP_v4(
                                                    output_path, my_file_name, 
                                                    q_crustal_xy, nx, 
                                                    ny, dx, dy, xmin, 
                                                    xmax, ymin, ymax, AOI_np
                                                    )
                my_file_name = "Flex_Thermal_Load_MPa_Age_"+str(age)
                map_tools.make_output_file_ZMAP_v4(
                                                    output_path, my_file_name, 
                                                    q_thermal_xy, nx, 
                                                    ny, dx, dy, xmin, 
                                                    xmax, ymin, ymax, AOI_np
                                                    )
                my_file_name = "Flex_Te_m_Age_"+str(age)
                map_tools.make_output_file_ZMAP_v4(
                                                    output_path, my_file_name, 
                                                    Te_xy, nx, 
                                                    ny, dx, dy, xmin, 
                                                    xmax, ymin, ymax, AOI_np
                                                    )
            if ioutput_FW_TTS == 1:
                my_file_name = "TTS_FW_Age_"+str(age)
                map_tools.make_output_file_ZMAP_v4(
                                                    output_path, my_file_name, 
                                                    ttsub_xy, nx, 
                                                    ny, dx, dy, xmin, 
                                                    xmax, ymin, ymax, AOI_np
                                                    )
            if ioutput_HFTOT == 1:
                my_file_name = "HF_TOT_Age_"+str(age)
                for i in range(nx): 
                    for j in range(ny):
                        hf_xy[i][j] = ( 
                                         hf_anom_xy[i][j]/hf_reduc_fac_xy[i][j] 
                                       + bghf_xy[i][j]
                                      )
                map_tools.make_output_file_ZMAP_v4(
                                                    output_path, my_file_name, 
                                                    hf_xy, nx, 
                                                    ny, dx, dy, xmin, 
                                                    xmax, ymin, ymax, AOI_np
                                                    )
            if ioutput_HF_ANOM == 1:
                my_file_name = "HF_ANOM_FW_Age_"+str(age)
                map_tools.make_output_file_ZMAP_v4(
                                                    output_path, my_file_name, 
                                                    hf_anom_xy, nx, 
                                                    ny, dx, dy, xmin, 
                                                    xmax, ymin, ymax, AOI_np
                                                    )
        tt2 = time.time()
        print_funcs.print_finfo(
                        ioutput_main, process, 
                        "Exported subsidence and heat flow maps", tt2-tt1)
    

def export_interpolated_sub_hf_maps(model, ioutput_main, process):
    if model.itype3D == 1:
        output_path = model.output_path
        nx = model.nx
        ny = model.ny
        dx = model.dx
        dy = model.dy
        xmin = model.xmin
        xmax = model.xmax
        ymin = model.ymin
        ymax = model.ymax
        AOI_np = model.AOI_np    
        hf_reduc_fac_xy = np.copy(model.hf_reduc_fac_xy)
        bghf_xy = np.copy(model.bghf_xy)
        ioutput_FW_TTS = model.ioutput_FW_TTS
        ioutput_HFTOT = model.ioutput_HFTOT
        ioutput_HF_ANOM = model.ioutput_HF_ANOM
    
        tt1 = time.time()
        interp_IDs = model.TTS_FW_dict.keys()
        for interp_ID in interp_IDs:
            age = model.TTS_FW_dict[interp_ID][0]
            ttsub_xy = np.copy(model.TTS_FW_dict[interp_ID][1])
            hf_xy = np.copy(model.TTS_FW_dict[interp_ID][2])
            hf_anom_xy = np.copy(model.TTS_FW_dict[interp_ID][3])
            if ioutput_FW_TTS == 1:
                my_file_name = "TTS_FW_Age_"+str(age)
                map_tools.make_output_file_ZMAP_v4(
                                                    output_path, my_file_name, 
                                                    ttsub_xy, nx, 
                                                    ny, dx, dy, xmin, 
                                                    xmax, ymin, ymax, AOI_np
                                                )
            if ioutput_HFTOT == 1:
                my_file_name = "HF_TOT_Age_"+str(age)
                for i in range(nx):
                    for j in range(ny): 
                        hf_xy[i][j] = (
                                        hf_anom_xy[i][j]/hf_reduc_fac_xy[i][j] 
                                        + bghf_xy[i][j]
                                    )
                map_tools.make_output_file_ZMAP_v4(
                                                    output_path, my_file_name, 
                                                    hf_xy, nx, 
                                                    ny, dx, dy, xmin, 
                                                    xmax, ymin, ymax, AOI_np
                                                )
            if ioutput_HF_ANOM == 1:
                my_file_name = "HF_ANOM_FW_Age_"+str(age)
                map_tools.make_output_file_ZMAP_v4(
                                                    output_path, my_file_name, 
                                                    hf_anom_xy, nx, 
                                                    ny, dx, dy, xmin, 
                                                    xmax, ymin, ymax, AOI_np
                                                )
                
        tt2 = time.time()
        print_funcs.print_finfo(
                ioutput_main, process, 
                "Exported interpolated subsidence and heat flow maps", tt2-tt1)
    

def export_stretching_maps(ioutput_main, process, model, 
                           delta_best_fit_xy_np, crustal_thick_xy_np):
    output_path = model.output_path
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np
    
    if model.inv_itype >= 0:
        tt1 = time.time()
        file_name = "best_fit_deltas"
        map_tools.make_output_file_ZMAP_v4(
                                            output_path, file_name, 
                                            delta_best_fit_xy_np,
                                            nx, ny, dx, dy, 
                                            xmin, xmax, ymin, ymax, AOI_np
                                        )
        file_name = "best_fit_xth"
        map_tools.make_output_file_ZMAP_v4(
                                            output_path, file_name, 
                                            crustal_thick_xy_np, 
                                            nx, ny, dx, dy, 
                                            xmin, xmax, ymin, ymax, AOI_np
                                        )
        tt2 = time.time()
        print_funcs.print_finfo(
            ioutput_main, process, 
            "Exported best-fit deltas and crustal thickness maps", tt2-tt1)
        

def export_moho_maps(ioutput_main, process, model, moho_xy_np, moho_twt_xy_np):
    output_path = model.output_path
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np
    
    if model.inv_itype >= 0:
        tt1 = time.time()
        file_name = "Moho_TWT"        
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, moho_twt_xy_np,
                                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                        AOI_np
                                    )
        file_name = "Moho"
        map_tools.make_output_file_ZMAP_v4(
                                            output_path, file_name, moho_xy_np,
                                            nx, ny, dx, dy, 
                                            xmin, xmax, ymin, ymax, AOI_np
                                        )
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Exported moho maps", tt2-tt1) 

    
def export_initial_burial_history(ioutput_main, process, model):
    ioutput_burial = model.ioutput_burial
    itype3D = model.itype3D
    if ioutput_burial == 1 and itype3D == 1:
        tt1 = time.time()
        sflag = "Initial"
        output_burial_history(model, sflag)
        tt2 = time.time()
        print_funcs.print_finfo(
                            ioutput_main, process, 
                            "Exported initial burial history maps", tt2-tt1)


def export_updated_burial_history(ioutput_main, process, model):
    ioutput_burial = model.ioutput_burial
    itype3D = model.itype3D
    inv_itype = model.inv_itype
    iupdate_PWD = model.iupdate_PWD
    if inv_itype >= -1 or itype3D == 0:
        if ioutput_burial == 1 and iupdate_PWD == 1:
            tt1 = time.time()
            sflag = "Updated"
            output_burial_history(model, sflag)
            tt2 = time.time()
            print_funcs.print_finfo(
                            ioutput_main, process, 
                            "Exported updated burial history maps", tt2-tt1)
        

def output_burial_history(model, sflag):
    output_path = model.output_path
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    
    AOI_np = model.AOI_np
    keys = list(model.event_dict_bs.keys())
    ntops = len(model.tops_list_bs)    
    # Looping over all events in from oldest to youngest
    for kk, event_ID in enumerate(keys):        
        # Getting the top ID for this event
        itop_event = model.event_dict_bs[event_ID][2]        
        age = model.event_dict_bs[event_ID][0]        
        # Looping over all tops from oldest to youngest        
        for jj in range(ntops):            
            # We only want tops that with indices that are 
            # less than or equal to current top
            if jj <= itop_event:                
                event_index = model.tops_list_bs[jj][14][event_ID]
                name = model.tops_list_bs[jj][6]
                depth_xy = np.copy(model.tops_list_bs[jj][1][event_index])
                file_name = (
                               "DEPTH_" 
                             + sflag + "_EV_" 
                             + str(event_ID) 
                             + "_t" 
                             + str(jj) 
                             + "_AGE_" 
                             + str(age)
                             + "_TOP_"+name
                             )
                map_tools.make_output_file_ZMAP_v4(
                                                output_path, file_name, 
                                                depth_xy, nx, ny, dx, dy, 
                                                xmin, xmax, ymin, ymax, AOI_np
                                            )

        
def export_temperature_maps(model, ioutput_main, process):
    output_path = model.output_path
    Lx = model.Lx
    Ly = model.Ly
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np
    src_top_names = model.src_top_names
    
    if model.icalc_temp == 1:
        if model.itype3D == 1:
            if model.ioutput_TEMP > 0:
                tt1 = time.time()
                stype="TEMP"
                sflag = "C"
                if model.ioutput_TEMP == 1:
                    imake_all_history_tmp = 1
                    imake_all_tops_tmp = 0
                elif model.ioutput_TEMP == 2:
                    imake_all_history_tmp = 1
                    imake_all_tops_tmp = 1 
                map_tools.output_zmap_history(
                                            stype, output_path,
                                            model, 
                                            Lx, Ly, nx, ny, dx, dy, 
                                            xmin, xmax, ymin, ymax, 
                                            AOI_np, sflag, 
                                            imake_all_history_tmp, 
                                            imake_all_tops_tmp, 
                                            src_top_names
                                        )
                tt2 = time.time()
                print_funcs.print_finfo(
                                    ioutput_main, process, 
                                    "Exported temperature maps", tt2-tt1)

                
def export_maturity_maps(model, process, ioutput_main):
    output_path = model.output_path
    Lx = model.Lx
    Ly = model.Ly
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np
    src_top_names = model.src_top_names
    if model.icalc_temp == 1 and model.itype3D == 1:
        if model.ioutput_Ro > 0:
            tt1 = time.time()
            if model.ioutput_Ro == 1:
                imake_all_history_tmp = 0
                imake_all_tops_tmp = 1
            elif model.ioutput_Ro == 2:
                imake_all_history_tmp = 1
                imake_all_tops_tmp = 1
            stype="EasyRo"
            sflag = "PER"
            map_tools.output_zmap_history(
                                        stype, output_path, 
                                        model,
                                        Lx, Ly, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np, sflag, 
                                        imake_all_history_tmp, 
                                        imake_all_tops_tmp, src_top_names
                                    )
            tt2 = time.time()
            print_funcs.print_finfo(
                                    ioutput_main, process, 
                                    "Exported EasyRo maps", tt2-tt1)        
    if model.icalc_LOM  == 1 and model.itype3D == 1:
        if model.ioutput_LOM > 0:
            tt1 = time.time()
            if model.ioutput_LOM == 1:
                imake_all_history_tmp = 0
                imake_all_tops_tmp = 1
            elif model.ioutput_LOM == 2:
                imake_all_history_tmp = 1
                imake_all_tops_tmp = 1
            stype="LOM"
            sflag = "UNIT"
            map_tools.output_zmap_history(
                                        stype, output_path, 
                                        model,
                                        Lx, Ly, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np, sflag, 
                                        imake_all_history_tmp, 
                                        imake_all_tops_tmp, src_top_names
                                    )
            tt2 = time.time()
            print_funcs.print_finfo(
                                    ioutput_main, process, 
                                    "Exported LOM maps", tt2-tt1)
            

def export_gen_maps_and_expulsion(model, ioutput_main, process):
    output_path = model.output_path
    Lx = model.Lx
    Ly = model.Ly
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np
    itype3D = model.itype3D
    inode = model.inode
    jnode = model.jnode
    src_top_names = model.src_top_names
    
    if model.itype3D == 1 and model.imass_gen > 0:
        tt1 = time.time()
        source_kinetics.source_expulsion_history(
                                output_path, model.tops_list_bs, 
                                model.event_dict_bs,
                                Lx, Ly, nx, ny, dx, dy, 
                                xmin, xmax, ymin, ymax, 
                                AOI_np, itype3D, 
                                inode, jnode, src_top_names
                                )
        tt2 = time.time()
        print_funcs.print_finfo(
            ioutput_main, process, 
            "Exported source expulsion history", tt2-tt1)
        tt1 = time.time()
        source_kinetics.output_gen_and_yield_maps(
                                output_path, model,
                                Lx, Ly, nx, ny, dx, dy, xmin, xmax, ymin,
                                ymax, AOI_np, src_top_names
                                )
        tt2 = time.time()
        print_funcs.print_finfo(
            ioutput_main, process, 
            "Exported yield maps", tt2-tt1)
        
        if model.ioutput_TR > 0:
            tt1 = time.time()
            if model.ioutput_TR == 1:
                imake_all_history_tmp = 0
                imake_all_tops_tmp = 1
            elif model.ioutput_TR == 2:
                imake_all_history_tmp = 1
                imake_all_tops_tmp = 1                
            stype="TR"
            sflag = "frac"
            map_tools.output_zmap_history(
                                        stype, output_path, 
                                        model,
                                        Lx, Ly, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np, sflag, 
                                        imake_all_history_tmp, 
                                        imake_all_tops_tmp, src_top_names
                                        )                
            tt2 = time.time()
            print_funcs.print_finfo(
                                    ioutput_main, process, 
                                    "Exported TR maps", tt2-tt1)            
        if model.ioutput_mHC > 0:
            tt1 = time.time()
            if model.ioutput_mHC == 1:
                imake_all_history_tmp = 0
                imake_all_tops_tmp = 1
            elif model.ioutput_mHC == 2:
                imake_all_history_tmp = 1
                imake_all_tops_tmp = 1
            stype="mHC"
            sflag = "mg_gOC"
            map_tools.output_zmap_history(
                                        stype, output_path,
                                        model, 
                                        Lx, Ly, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np, sflag, 
                                        imake_all_history_tmp, 
                                        imake_all_tops_tmp, src_top_names
                                        )
            tt2 = time.time()
            print_funcs.print_finfo(
                                    ioutput_main, process, 
                                    "Exported mgHC_gTOC maps", tt2-tt1)
        if model.ioutput_mODG > 0:
            tt1 = time.time()
            if model.ioutput_mODG == 1:
                imake_all_history_tmp = 0
                imake_all_tops_tmp = 1
            elif model.ioutput_mODG == 2:
                imake_all_history_tmp = 1
                imake_all_tops_tmp = 1
            stype="mODG"
            sflag = "mg_gOC"
            map_tools.output_zmap_history(
                                        stype, output_path, 
                                        model,
                                        Lx, Ly, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np, sflag, 
                                        imake_all_history_tmp, 
                                        imake_all_tops_tmp, src_top_names
                                        )
            tt2 = time.time()
            print_funcs.print_finfo(
                                    ioutput_main, process, 
                                    "Exported mgODG_gTOC maps", tt2-tt1)    
        if model.ioutput_mFG > 0:
            tt1 = time.time()
            if model.ioutput_mFG == 1:
                imake_all_history_tmp = 0
                imake_all_tops_tmp = 1
            elif model.ioutput_mFG == 2:
                imake_all_history_tmp = 1
                imake_all_tops_tmp = 1
            
            stype="mFG"
            sflag = "mg_gOC"
            map_tools.output_zmap_history(
                                        stype, output_path, 
                                        model, 
                                        Lx, Ly, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np, sflag, 
                                        imake_all_history_tmp, 
                                        imake_all_tops_tmp, src_top_names
                                        )        
            tt2 = time.time()
            print_funcs.print_finfo(
                                    ioutput_main, process, 
                                    "Exported mgFG_gTOC maps", tt2-tt1)    
        if model.ioutput_SEC_mFG > 0:
            tt1 = time.time()
            if model.ioutput_SEC_mFG == 1:
                imake_all_history_tmp = 0
                imake_all_tops_tmp = 1
            elif model.ioutput_SEC_mFG:
                imake_all_history_tmp = 1
                imake_all_tops_tmp = 1
            stype="SEC_mFG"
            sflag = "mg_gOC"
            map_tools.output_zmap_history(
                                        stype, output_path, 
                                        model,
                                        Lx, Ly, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np, sflag, 
                                        imake_all_history_tmp, 
                                        imake_all_tops_tmp, src_top_names
                                        )
            tt2 = time.time()
            print_funcs.print_finfo(
                        ioutput_main, process, 
                        "Exported secondary cracking mgFG_gTOC maps", tt2-tt1)    
        if model.ioutput_EXPRATE > 0:     
            tt1 = time.time()
            if model.ioutput_EXPRATE == 1:
                imake_all_history_tmp = 0
                imake_all_tops_tmp = 1
            elif model.ioutput_EXPRATE == 2:
                imake_all_history_tmp = 1
                imake_all_tops_tmp = 1
            stype="EXPRATE"
            sflag = "mg_gOC_Myr"
            map_tools.output_zmap_history(
                                        stype, output_path, 
                                        model,
                                        Lx, Ly, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np, sflag, 
                                        imake_all_history_tmp, 
                                        imake_all_tops_tmp, src_top_names
                                        )
            tt2 = time.time()
            print_funcs.print_finfo(
                            ioutput_main, process, 
                            "Exported primary expulsion rate maps", tt2-tt1)    
        if model.ioutput_SEC_EXPRATE > 0:
            tt1 = time.time()
            if model.ioutput_SEC_EXPRATE == 1:
                imake_all_history_tmp = 0
                imake_all_tops_tmp = 1
            elif model.ioutput_SEC_EXPRATE == 2:
                imake_all_history_tmp = 1
                imake_all_tops_tmp = 1
            stype="SEC_EXPRATE"
            sflag = "mg_gOC_Myr"
            map_tools.output_zmap_history(
                                        stype, output_path, 
                                        model,
                                        Lx, Ly, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, 
                                        AOI_np, sflag, 
                                        imake_all_history_tmp,
                                        imake_all_tops_tmp, src_top_names
                                        )
            
            tt2 = time.time()
            print_funcs.print_finfo(
                            ioutput_main, process, 
                            "Exported secondary expulsion rate maps", tt2-tt1)

