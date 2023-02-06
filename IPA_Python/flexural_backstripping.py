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
import psutil
import math
import time
import numpy as np
from numba import jit
import map_tools
import flex_restore
import data_exporter
import print_funcs


def flexural_backstripping(model, ioutput_main, process):
    if model.iuse_flexure == 1:
        tt1 = time.time()            
        if model.isalt_restore == 1: 
            iter_flag = "a"
        else:
            iter_flag = "final"
        ioutput = 1
        flexural_backstripping_method1(model, ioutput, iter_flag)
        tt2 = time.time()
        print_funcs.print_finfo(
                ioutput_main, process, 
                "Calculated PWD using flexural backstripping", tt2-tt1)    
    
def thermo_tectonic_sub_flexure(
                                tops_list_bs, event_dict_bs, 
                                rho_water, rho_crust, rho_mantle, 
                                alpha_bulk, T_base, ioutput, output_path, 
                                Lx, Ly, nx, ny, dx, dy, 
                                xmin, xmax, ymin, ymax, 
                                itype_rho_a, AOI_np, ioutput_TTS, 
                                dist_taper, iuse_flexure, 
                                deltaSL_list, tc_initial, 
                                xth_xy, ioutput_main, process
):
    if iuse_flexure == 1:            
        icalc_tts_flex = 0
        if icalc_tts_flex == 1:                
            tt1 = time.time()        
            ioutput = 1
            calculate_thermo_tectonic_sub_flexure(
                                tops_list_bs, event_dict_bs, 
                                rho_water, rho_crust, rho_mantle, 
                                alpha_bulk, T_base, ioutput, output_path, 
                                Lx, Ly, nx, ny, dx, dy, 
                                xmin, xmax, ymin, ymax, 
                                itype_rho_a, AOI_np, ioutput_TTS, 
                                dist_taper, iuse_flexure, 
                                deltaSL_list, tc_initial, 
                                xth_xy
                            )                
            tt2 = time.time()
            print_funcs.print_finfo(
                ioutput_main, process, 
                "Finished calculating TTS using flexural backstripping", 
                tt2-tt1)    
    
    
def calculate_thermo_tectonic_sub_flexure(
                                            tops_list_bs, event_dict_bs, 
                                            rho_water, rho_crust, rho_mantle, 
                                            alpha_bulk, T_base, ioutput, 
                                            output_path,
                                            Lx, Ly, nx, ny, dx, dy, 
                                            xmin, xmax, ymin, ymax, 
                                            itype_rho_a, AOI_np, ioutput_TTS, 
                                            dist_taper, iuse_flexure, 
                                            deltaSL_list, tc_initial, 
                                            crustal_thick_xy
):
    # Flexure - Poisson's ratio.
    nu = 0.25
    # Young's modulus Pa
    E = 7e10
    # Set to zero to force local isostasy; set to 1 to allow normal Te
    Te_fac = 1.0
    rad_search_m = 5000.0
    keys = event_dict_bs.keys()
    nevents = len(keys)
    event_ID_last = nevents - 1
    AOI_clean_np = np.ones((nx, ny))    
    w_tts = np.zeros((nx,ny))
    for k in range(nevents):
        # Looping over events from youngest to oldest
        event_ID = nevents - k - 1
        if event_ID > 0:
            event_age = event_dict_bs[event_ID][0]
            print ("\n Calculating incremental sediment flexure for event : ",
                   event_ID, " : event_age : ", event_age, "\n")
            all_zeros = np.zeros((nx,ny))
            AOItmp = np.zeros((nx,ny))
            #************************
            # Change in sediment load
            #************************            
            q_sed = event_dict_bs[event_ID][20]            
            q_sed_np = np.copy(q_sed)
            fill_undefined(q_sed_np, AOItmp, nx, ny, dx, dy)
            q_sed_prev = event_dict_bs[event_ID-1][20]
            q_sed_prev_np = np.copy(q_sed_prev)
            fill_undefined(q_sed_prev_np, AOItmp, nx, ny, dx, dy)
            delta_q_sed = q_sed_np - q_sed_prev_np          
            #****************************
            # Effective elastic thickness
            #****************************
            Te_map = event_dict_bs[event_ID][19]
            Te_map_np_tmp = np.copy(Te_map)*Te_fac
            fill_undefined(Te_map_np_tmp, AOItmp, nx, ny, dx, dy)
            flex_restore.clean_load_boundaries(nx, ny, Te_map_np_tmp)            
            Te_map_np = np.zeros((nx, ny))
            map_tools.low_pass_filter_zmap(
                                            dx, dy, nx, ny, AOI_clean_np, 
                                            rad_search_m, Te_map_np_tmp, 
                                            Te_map_np
                                        )
            #****************************
            # Mantle density
            #****************************            
            rho_m_map = np.ones((nx,ny))*rho_mantle
            fill_undefined(rho_m_map, AOItmp, nx, ny, dx, dy)
            AOI = np.zeros((nx,ny))            
            AOI_cells = []
            for i in range(nx):
                for j in range(ny):              
                    AOI[i,j] = 1
                    AOI_cells.append([i, j])
            
            n_AOIs = len(AOI_cells)
            #*****************************************
            # Calculating incremental total deflection
            #*****************************************
            xn = nx
            yn = ny            
            # Clean boundaries
            flex_restore.clean_load_boundaries(xn, yn, Te_map_np)
            flex_restore.clean_load_boundaries(xn, yn, delta_q_sed)            
            # Convert loads to Pa from MPa
            (
                w_inv, Flex_w, xn_exp, yn_exp
            ) = flex_restore.calc_flex_sub(
                                                rho_crust, rho_water, 
                                                rho_m_map, delta_q_sed*1e6, 
                                                all_zeros, all_zeros,
                                                all_zeros, all_zeros, 
                                                dx, dy, AOI, n_AOIs,
                                                Te_map_np, nu, E, 
                                                dist_taper, event_ID
                                            )
            w_tts = w_tts + w_inv          
    pwd_final_np = np.copy(event_dict_bs[event_ID_last][5])    
    sed_thick_final_np = np.copy(event_dict_bs[event_ID_last][21])
    bsmt_final_np = pwd_final_np + sed_thick_final_np    
    tts_flex_np = bsmt_final_np + w_tts
    tts_flex_xy = np.copy(tts_flex_np)
    my_file_name = "Flex_TTS_0Ma_"
    map_tools.make_output_file_ZMAP_v4(
                                    output_path, my_file_name, tts_flex_xy,
                                    nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                    AOI_clean_np
                                )
    return tts_flex_xy


@jit(nopython=True)
def fill_undefined(scalars, AOI, xn, yn, dx, dy):
    xmin = 0.0
    ymin = 0.0    
    # Loop over main grid
    for i in range(xn):
        for j in range(yn):
            xm = xmin + dx*i
            ym = ymin + dy*j
            vm = scalars[i,j]
            if vm == -99999.0:                
                AOI[i,j] = 0
            else:
                AOI[i,j] = 1
    # Loop over main grid
    for i in range(xn):
        for j in range(yn):
            xm = xmin + dx*i
            ym = ymin + dy*j
            vm = scalars[i,j]
            AOI_flag = AOI[i,j]
            if AOI_flag == 0:
                # Loop over small grid and find closest small grid node
                ii_c = -9999
                jj_c = -9999
                dist_c = 1e32
                ifind = 0
                for ii in range(xn):
                    for jj in range(yn):
                        AOI_flagB = AOI[ii,jj]
                         # Only consider nodes that had original values defined
                        if AOI_flagB == 1:
                            ifind = 1
                            xs = xmin + dx*ii
                            ys = ymin + dy*jj
                            ddx = xs-xm
                            ddy = ys-ym
                            dist = math.sqrt(ddx*ddx+ddy*ddy)
                            if dist < dist_c:
                                dist_c = dist
                                ii_c = ii
                                jj_c = jj
                if ifind == 1:
                    scalars[i,j] = scalars[ii_c,jj_c]
                    

@jit(nopython=True)
def calc_change_baselevel_load(nx, ny, rho_water, dSL, pwd_flex, delta_q_BL):
    for i in range(nx):
        for j in range(ny):
            pwd = pwd_flex[i][j]
            if pwd > 0:
                if dSL <= pwd:
                    dSL_final = dSL
                else:
                    dSL_final = pwd
            else:
                dSL_final = 0.0
            delta_q_BL[i,j] = rho_water*dSL_final*9.81/1e6
            

@jit(nopython=True)
def calc_nAOIs(AOI, nx, ny):
    n_AOIs = 0
    for i in range(nx):   
        for j in range(ny):      
            AOI[i,j] = 1
            n_AOIs = n_AOIs + 1                
    return n_AOIs
              

def flexural_backstripping_method1(model, ioutput, iterID):
    deltaSL_list = model.deltaSL_list
    rho_water = model.rho_water
    output_path = model.output_path
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    rho_water = model.rho_water
    rho_crust = model.rho_crust
    rho_mantle = model.rho_mantle
    dist_taper = model.dist_taper
    deltaSL_list = model.deltaSL_list
    rad_search_m = model.rad_search_m
    PWD_overwrite_flag_dict = model.PWD_overwrite_flag_dict

    keys = list(model.event_dict_bs.keys())    
    nevents_bs = len(keys)
    event_ID_list = keys[:]
    event_ID_last = event_ID_list[nevents_bs-1]
                                    
    process = psutil.Process(os.getpid())
    if iterID == "final":
        ioutput = 1
    else:
        ioutput = 0
    # Flexure - Poisson's ratio.
    nu = 0.25
     # Flexure - Young's modulus N/m^2.
    E = 7e10
    # used for thermal smoothing
    rad_search_m_local = 5000.0    
    keys = model.event_dict_bs.keys()
    nevents = len(keys)    
    AOI_clean_np = np.ones((nx, ny))    
    # List of crustal loads for each event
    delta_q_uc_dict = {}    
    for k in range(nevents):        
        # Looping over events from youngest to oldest
        event_ID = nevents - k - 1       
        if event_ID == 0:
            q_dum = np.zeros((nx,ny))
            delta_q_uc_dict[event_ID] = np.copy(q_dum)
        if event_ID > 0:
            event_age = model.event_dict_bs[event_ID][0]
            print ("Calculating incremental flexure for event : ",
                   event_ID, " : event_age : ", event_age)
            AOItmp = np.zeros((nx,ny))
            tt1 = time.time()            
            #************************
            # Change in sediment load
            #************************            
            q_sed_np = np.copy(model.event_dict_bs[event_ID][20])
            # Fill all undefined cells with closest defined value
            fill_undefined(q_sed_np, AOItmp, nx, ny, dx, dy)
            q_sed_prev_np = np.copy(model.event_dict_bs[event_ID-1][20])
            # Fill all undefined cells with closest defined value
            fill_undefined(q_sed_prev_np, AOItmp, nx, ny, dx, dy)
            delta_q_sed = q_sed_np - q_sed_prev_np
            flex_restore.clean_load_boundaries(nx, ny, delta_q_sed)
            delta_q_sed_xy = np.copy(delta_q_sed)
            #***************************************
            # Change in upper crustal thickness load
            #***************************************
            q_uc_np = np.copy(model.event_dict_bs[event_ID][16])
            # Fill all undefined cells with closest defined value
            fill_undefined(q_uc_np, AOItmp, nx, ny, dx, dy)
            q_uc_prev_np = np.copy(model.event_dict_bs[event_ID-1][16])
            # Fill all undefined cells with closest defined value
            fill_undefined(q_uc_prev_np, AOItmp, nx, ny, dx, dy)
            delta_q_uc = q_uc_np - q_uc_prev_np
            flex_restore.clean_load_boundaries(nx, ny, delta_q_uc)
            delta_q_uc_dict[event_ID] = np.copy(delta_q_uc)
            delta_q_uc_xy = np.copy(delta_q_uc)
            #************************
            # Change in thermal load
            #***********************
            q_thermal_np = np.copy(model.event_dict_bs[event_ID][17])
            # Fill all undefined cells with closest defined value
            fill_undefined(q_thermal_np, AOItmp, nx, ny, dx, dy)            
            q_thermal_prev = model.event_dict_bs[event_ID-1][17]
            q_thermal_prev_np = np.copy(q_thermal_prev)
            # Fill all undefined cells with closest defined value
            fill_undefined(q_thermal_prev_np, AOItmp, nx, ny, dx, dy)
            delta_q_thermal = q_thermal_np - q_thermal_prev_np
            # Apply low pass filter for 3D thermal effect
            delta_q_thermal_tmp = q_thermal_np - q_thermal_prev_np
            flex_restore.clean_load_boundaries(nx, ny, delta_q_thermal_tmp)
            delta_q_thermal = np.zeros((nx,ny))
            map_tools.low_pass_filter_zmap(dx, dy, nx, ny, AOI_clean_np,
                    rad_search_m_local, delta_q_thermal_tmp, delta_q_thermal)
            flex_restore.clean_load_boundaries(nx, ny, delta_q_thermal)
            delta_q_thermal_xy = np.copy(delta_q_thermal)
            #*************************************************************
            # Set change in lower crustal load and base level load to zero
            #*************************************************************
            tt1 = time.time()
            delta_q_lc = np.zeros((nx,ny))
            delta_q_BL = np.zeros((nx,ny))
            #****************************
            # Effective elastic thickness
            #****************************
            Te_map_np_tmp = np.copy(model.event_dict_bs[event_ID][19])
            # Fill all undefined cells with closest defined value
            fill_undefined(Te_map_np_tmp, AOItmp, nx, ny, dx, dy)
            flex_restore.clean_load_boundaries(nx, ny, Te_map_np_tmp)
            # Apply low pass filter for 3D thermal effect
            Te_map_np = np.zeros((nx, ny))
            map_tools.low_pass_filter_zmap(
                                            dx, dy, nx, ny, 
                                            AOI_clean_np,
                                            rad_search_m_local, 
                                            Te_map_np_tmp, Te_map_np
                                            )
            flex_restore.clean_load_boundaries(nx, ny, Te_map_np)
            Te_map_xy = np.copy(Te_map_np)
            tt2 = time.time()
            print_funcs.print_finfo(
                                1, process, 
                                "----> Calculated load and Te maps", tt2-tt1)
            #****************************
            # Mantle density
            #****************************
            # For now let's have constant density in mantle, 
            # no lower crustal stretching and no base level changes
            rho_m_map = np.ones((nx,ny))*rho_mantle
            # Fill all undefined cells with closest defined value
            fill_undefined(rho_m_map, AOItmp, nx, ny, dx, dy)
            AOI = np.zeros((nx,ny))
            n_AOIs = calc_nAOIs(AOI, nx, ny)
            #*****************************************
            # Calculating incremental total deflection
            #*****************************************
            tt1 = time.time()
            (
                w, 
                Flex_w, 
                xn_exp, 
                yn_exp
            ) = flex_restore.calc_flex_sub(
                                                rho_crust, rho_water, 
                                                rho_m_map, delta_q_sed*1e6, 
                                                delta_q_uc*1e6, delta_q_lc*1e6,
                                                delta_q_thermal*1e6, 
                                                delta_q_BL*1e6, 
                                                dx, dy, AOI, n_AOIs,
                                                Te_map_np, nu, E, 
                                                dist_taper, event_ID
                                                )
            w_xy = np.copy(w)
            model.event_dict_bs[event_ID][18] = np.copy(w_xy)
            AOI_tmp_np = np.ones((xn_exp,yn_exp))
            Flex_w_xy = np.copy(Flex_w)
            tt2 = time.time()
            print_funcs.print_finfo(
                    1, process, 
                    "----> Calculated incremental total deflection", tt2-tt1)
            data_exporter.export_load_and_incremental_flex_maps(
                                        model, ioutput, process,
                                        iterID, event_age, event_ID,
                                        AOI_clean_np,  AOI_tmp_np,
                                        delta_q_sed_xy, delta_q_uc_xy, 
                                        delta_q_thermal_xy, Te_map_xy, w_xy, 
                                        Flex_w_xy, xn_exp, yn_exp, Te_map_np,
                                        q_thermal_np
                                        )
    #**************************************************************************
    # Reconstructing water depth for water loaded configuration (i.e. dSL = 0)
    #**************************************************************************
    event_ID_last = nevents - 1
    wd_0Ma_xy = np.copy(model.event_dict_bs[event_ID_last][5])
    model.event_dict_bs[event_ID_last][22] = np.copy(wd_0Ma_xy)
    AOItmp = np.zeros((nx,ny))
    for k in range(nevents):
        # Looping over events from youngest to oldest
        event_ID = nevents - k - 1
        if event_ID > 0:
            event_age = model.event_dict_bs[event_ID][0]
            # Skip the last event
            if event_ID < event_ID_last:
                print ("Reconstructing PWD with flexure for event : ",
                       event_ID, " : event_age : ", event_age)
                tt1 = time.time()
                pwd_next_np = np.copy(model.event_dict_bs[event_ID+1][22])
                fill_undefined(pwd_next_np, AOItmp, nx, ny, dx, dy)
                sed_thick_next_np = np.copy(
                                           model.event_dict_bs[event_ID+1][21])
                fill_undefined(sed_thick_next_np, AOItmp, nx, ny, dx, dy)
                bsmt_next_np = pwd_next_np + sed_thick_next_np
                w_inc_next_np = np.copy(model.event_dict_bs[event_ID+1][18])
                # Get upper crustal thickness load from next event and 
                # convert to Pa from MPa)
                delta_q_uc_next_np = np.copy(delta_q_uc_dict[event_ID+1])*1e6
                # crustal thickness change in m
                delta_crust_np = delta_q_uc_next_np/9.81/(rho_crust-rho_water)
                sed_thick_np = np.copy(model.event_dict_bs[event_ID][21])
                fill_undefined(sed_thick_np, AOItmp, nx, ny, dx, dy)
                # need to add the crustal thickness change to reconstruct 
                # basement
                # Shallow the bsmt with the change in crustal thickness
                bsmt_new = bsmt_next_np + w_inc_next_np - delta_crust_np 
                bsmt_new_xy = np.copy(bsmt_new)
                model.event_dict_bs[event_ID][26] = np.copy(bsmt_new_xy)
                pwd_flex_np = bsmt_new - sed_thick_np
                pwd_flex_xy = np.copy(pwd_flex_np)
                model.event_dict_bs[event_ID][22] = np.copy(pwd_flex_xy)
                tt2 = time.time()
                print_funcs.print_finfo(
                            1, process, 
                            "----> Calculated BSMT and PWD flex maps", tt2-tt1)
                data_exporter.export_restored_bsmt_and_pwd_iter(
                                    model, ioutput, event_age, 
                                    event_ID, iterID, process,
                                    bsmt_new_xy, pwd_flex_xy
                                    )
    #**************************************************************************
    # Reconstructing base level shifts given pwd prediction starting from
    # water loaded configuration
    #**************************************************************************    
    AOItmp = np.zeros((nx,ny))
    for k in range(nevents):
        # Looping over events from youngest to oldest
        event_ID = nevents - k - 1
        dSL = deltaSL_list[event_ID]
        if event_ID > 0:
            event_age = model.event_dict_bs[event_ID][0]
            # Skip the last event and events with no base level changes
            if event_ID < event_ID_last and abs(dSL) > 0:
                print ("Reconstructing flexural base level shift for event : ",
                       event_ID, " : event_age : ", event_age)
                tt1 = time.time()
                delta_q_thermal = np.zeros((nx,ny))
                delta_q_uc = np.zeros((nx,ny))
                delta_q_lc = np.zeros((nx,ny))
                delta_q_BL = np.zeros((nx,ny))                
                pwd_flex = model.event_dict_bs[event_ID][22]
                pwd_flex_np = np.copy(pwd_flex)
                fill_undefined(pwd_flex_np, AOItmp, nx, ny, dx, dy)
                calc_change_baselevel_load(
                                            nx, ny, rho_water, dSL, 
                                            pwd_flex, delta_q_BL
                                            )
                #****************************
                # Effective elastic thickness
                #****************************
                Te_map_np_tmp = np.copy(model.event_dict_bs[event_ID][19])
                # Fill all undefined cells with closest defined value
                fill_undefined(Te_map_np_tmp, AOItmp, nx, ny, dx, dy)
                flex_restore.clean_load_boundaries(nx, ny, Te_map_np_tmp)
                Te_map_np = np.zeros((nx, ny))
                map_tools.low_pass_filter_zmap(
                                                dx, dy, nx, ny, AOI_clean_np,
                                                rad_search_m_local, 
                                                Te_map_np_tmp, Te_map_np
                                                )
                #****************************
                # Mantle density
                #****************************
                rho_m_map = np.ones((nx,ny))*rho_mantle
                # Fill all undefined cells with closest defined value
                fill_undefined(rho_m_map, AOItmp, nx, ny, dx, dy)
                xn = nx
                yn = ny
                # Clean boundaries
                flex_restore.clean_load_boundaries(xn, yn, Te_map_np)
                flex_restore.clean_load_boundaries(xn, yn, delta_q_BL)  
                AOI = np.zeros((nx,ny))
                n_AOIs = calc_nAOIs(AOI, nx, ny)
                all_zeros = np.zeros((nx,ny))
                # update base level load array                
                (
                    w, 
                    Flex_w, 
                    xn_exp, 
                    yn_exp
                ) = flex_restore.calc_flex_sub(
                                                    rho_crust, rho_water, 
                                                    rho_m_map, all_zeros, 
                                                    all_zeros, all_zeros,
                                                    all_zeros, delta_q_BL*1e6, 
                                                    dx, dy, AOI, n_AOIs,
                                                    Te_map_np, nu, E, 
                                                    dist_taper, event_ID
                                                    )
                # Correct water loaded pwd using base level change
                pwd_n_np = np.zeros((nx,ny))
                for i in range(nx):
                    for j in range(ny):
                        # water loaded pwd
                        pwd_flex_tmp = pwd_flex_np[i,j]
                        w_tmp = w[i,j]
                        # q_load = delta_q_BL[i,j]
                        # Need to account for flexural rebound from air load 
                        # Need to define pwd with respect to BL
                        pwd_n = pwd_flex_tmp - w_tmp - dSL
                        pwd_n_np[i,j] = pwd_n
                pwd_n_xy = np.copy(pwd_n_np)
                model.event_dict_bs[event_ID][22] = np.copy(pwd_n_xy)
                tt2 = time.time()
                print_funcs.print_finfo(
                                    1, process, 
                                    "----> Corrected for base level", tt2-tt1)
                data_exporter.export_flexural_baselevel_maps(
                                    output_path, ioutput, process,
                                    iterID, event_age, event_ID,
                                    AOI_clean_np, delta_q_BL, pwd_n_xy
                                    )
    # Apply low-pass filter to pwd based on flexure
    print ("----> Apply low pass filter to pwd based on flexure")
    tt1 = time.time()
    event_ID_last = nevents - 1
    wd_0Ma_xy = np.copy(model.event_dict_bs[event_ID_last][5])
    model.event_dict_bs[event_ID_last][22] = np.copy(wd_0Ma_xy)
    for k in range(nevents):    
        # Looping over events from youngest to oldest
        event_ID = nevents - k - 1
        if event_ID > 0:
            event_age = model.event_dict_bs[event_ID][0]
            if event_ID < event_ID_last: 
                # Skip the last event
                ioverwrite = PWD_overwrite_flag_dict[event_ID]
                pwd_overwrite_np = np.copy(model.event_dict_bs[event_ID][5])
                fill_undefined(pwd_overwrite_np, AOItmp, nx, ny, dx, dy)
                dSL = deltaSL_list[event_ID]
                pwd_flex_xy = np.copy(model.event_dict_bs[event_ID][22])
                # Actually define z_surf
                pwd_flex = pwd_flex_xy + dSL
                fill_undefined(pwd_flex, AOItmp, nx, ny, dx, dy)
                pwd_flex_LPF = np.zeros((nx, ny))
                map_tools.low_pass_filter_zmap(
                                                dx, dy, nx, ny, AOI_clean_np,
                                                rad_search_m, pwd_flex, 
                                                pwd_flex_LPF
                                                )
                # Now convert back to actual pwd
                pwd_flex_LPF = pwd_flex_LPF - dSL
                pwd_flex_LPF_xy = np.copy(pwd_flex_LPF)
                if ioverwrite == 0:
                    model.event_dict_bs[event_ID][25] = np.copy(
                                                            pwd_flex_LPF_xy)                    
                else:
                    model.event_dict_bs[event_ID][25] = np.copy(
                                                            pwd_overwrite_np)
    tt2 = time.time()
    print_funcs.print_finfo(1, process, "Applied low pass filters", tt2-tt1)