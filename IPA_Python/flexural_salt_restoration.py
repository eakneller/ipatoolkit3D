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
import time
import numpy as np
import flexural_backstripping
import map_tools
from numba import jit
import burial_history
import backstripping_tools
import print_funcs


def salt_restoration(model, ioutput_main, process):
    if model.iuse_flexure == 1:
        if model.isalt_restore == 1:
            tt1 = time.time()
            for isalt_iter in range(4):
                if ioutput_main == 1:
                    print_funcs.print_statement(
                        "Flexural salt thickness iteration " + str(isalt_iter))
                    print_funcs.print_statement(
                        "Update flexural pwd using latest sediment thickness "
                        "and estimated paleo-salt thickness")
                # update pwd_flex and pwd_flex_LP and estimate salt 
                # thickness change
                tt1 = time.time()
                ioutput = 1
                restore_salt_flex(model, ioutput, isalt_iter)                    
                tt2 = time.time()
                print_funcs.print_finfo(
                                ioutput_main, process, 
                                "-----> Restored salt using flexure ", tt2-tt1)
                # Perform backstripping without compaction 
                # using new salt thicknesses
                ioutput = 1
                isalt_restore_local = model.isalt_restore
                burial_history.calculate_burial_history(
                                            0, ioutput, 
                                            isalt_restore_local, model
                                            )
                print_funcs.print_finfo(
                    ioutput_main, process, 
                    "-----> Re-calculated burial history without compaction", 
                    tt2-tt1)
                # Calculate forward max burial
                tt1 = time.time()
                burial_history.calculate_forward_max_burial(model)
                tt2 = time.time()
                print_funcs.print_finfo(
                        ioutput_main, process, 
                        "-----> Re-calculated forward maximum burial", tt2-tt1)
                ioutput = 1
                # Calculate final backstripping with compaction
                tt1 = time.time()
                isalt_restore_local = model.isalt_restore
                burial_history.calculate_burial_history(
                                            model.icompact, ioutput, 
                                            isalt_restore_local, model
                                            )
                tt2 = time.time()
                print_funcs.print_finfo(
                    ioutput_main, process, 
                  "-----> Re-calculated burial history and sediment thickness", 
                    tt2-tt1)
                tt1 = time.time()
                imake_output = 1
                backstripping_tools.update_sed_thick_rho_load_tts(
                                                        imake_output, ioutput, 
                                                        model
                                                        )
                tt2 = time.time()
                print_funcs.print_finfo(
                    ioutput_main, process, 
                   "-----> Re-calculated sediment thickness and load", tt2-tt1)
            if ioutput_main == 1:
                print_funcs.print_statement(
                        "Re-calculate incremental flexure and pwd using "
                        "updated sediment loads")
            ioutput = 1
            flexural_backstripping.flexural_backstripping_method1(
                                                                model, ioutput, 
                                                                "final"
                                                                )                
            tt2 = time.time()
            print_funcs.print_finfo(
                ioutput_main, process, 
             "-----> Calculated salt thickness using flexural reconstructions", 
                tt2-tt1)

            
@jit(nopython=True)
def initialize_salt_thickness(
                                nx, ny, dx, dy, AOI_np, 
                                salt_thick_start_xy, 
                                salt_thick_final_xy
):
    # area of each element in km^3
    dA = dx/1000*dy/1000
    # volume in km^3
    vol_tot = 0.0
    for i in range(nx):
        for j in range(ny):            
            AOI_flag = AOI_np[i,j]            
            if AOI_flag == 1:                
                (
                    salt_thick
                 ) = salt_thick_start_xy[i,j]
            else:                
                salt_thick = -99999.0            
            if salt_thick >= 0.0:
                vol_tot = vol_tot + dA*salt_thick/1000.0
            salt_thick_final_xy[i,j] = salt_thick
    return vol_tot


@jit(nopython=True)
def update_salt_thickenss(
                            nx, ny, dx, dy, AOI_np, DS,
                            salt_thick_o, salt_thick_xy
):
    dA = dx/1000*dy/1000 # area of each element in km^3
    vol_tot = 0.0 # volume in km^3
    for i in range(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:
                salt_thick_tmp = salt_thick_o[i,j]
                DS_tmp = DS[i,j]
#                        # do not allow salt to inflate if overburden 
#                        #  is beyond a threshold
#                        top_salt_depth_sm = TOS[i,j] - pwd_flex[i,j]
#                        if (top_salt_depth_sm > 
#                                          inflation_threshold and DS_tmp > 0):
#                            DS_tmp = 0
                salt_thick = salt_thick_tmp + DS_tmp
                if salt_thick < 0:
                    salt_thick = 0.0
            else:
                salt_thick = 0.0
            salt_thick_xy[i][j] = salt_thick
            if salt_thick >= 0.0:
                vol_tot = vol_tot + dA*salt_thick/1000.0
    return vol_tot

                        
def restore_salt_flex(model, ioutput, iterID):
    output_path = model.output_path
    PWD_overwrite_flag_dict = model.PWD_overwrite_flag_dict
    deltaSL_list = model.deltaSL_list
    rad_search_m = model.rad_search_m
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np
    salt_layer_index = model.salt_layer_index
    
    ioutput = 0    
    ntops = len(model.tops_list_bs)    
    # event_ID for pre-salt deposition event; event ID is zero for first event
    event_ID_salt = ntops - salt_layer_index    
    event_age_salt = model.event_dict_bs[event_ID_salt][0]
    index_TOS = ntops - salt_layer_index
    index_BOS = index_TOS - 1
    salt_thick_final_xy = np.zeros((nx, ny))
    # The first event is the oldest and has an ID of 0
    event_ID_last = ntops-1
    itop_salt_final = ntops - salt_layer_index
    event_list_index = model.tops_list_bs[itop_salt_final][14][event_ID_last]
    (
        salt_thick_start_xy
    ) = np.copy(model.tops_list_bs[itop_salt_final][2][event_list_index])
    vol_final = initialize_salt_thickness(
                                nx, ny, dx, dy, AOI_np, 
                                salt_thick_start_xy, 
                                salt_thick_final_xy
                            )
    # Loop over each event starting from present day until salt time
    AOItmp = np.zeros((nx,ny))
    AOI_clean_np = np.ones((nx, ny))
    for ii in range(ntops):
        # move from younger to older
        event_ID = ntops - ii - 1
        event_age = model.event_dict_bs[event_ID][0]
        if event_age <= event_age_salt and event_age > 0.0: 
            #Skip present-day
            dSL = deltaSL_list[event_ID]
            ioverwrite = PWD_overwrite_flag_dict[event_ID]
            pwd_overwrite_xy = np.copy(model.event_dict_bs[event_ID][5])
            flexural_backstripping.fill_undefined(
                                                    pwd_overwrite_xy, AOItmp, 
                                                    nx, ny, dx, dy
                                                )
            bsmt_flex_xy = np.copy(model.event_dict_bs[event_ID][26])
            flexural_backstripping.fill_undefined(
                                                    bsmt_flex_xy, AOItmp, 
                                                    nx, ny, dx, dy
                                                )
            sed_thick_xy = np.copy(model.event_dict_bs[event_ID][21])
            flexural_backstripping.fill_undefined(
                                                    sed_thick_xy, AOItmp, 
                                                    nx, ny, dx, dy
                                                )
            # Here we are updating pwd
            # But we need to account for base level changes
            pwd_flex_xy = bsmt_flex_xy - sed_thick_xy - dSL
            if ioutput == 1:
                file_name = (
                                  "PWD_Flex_iter"+str(iterID)+"_"
                                + str(event_age)
                                + "_event_"+str(event_ID)
                            )
                map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, pwd_flex_xy, 
                                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                        AOI_np
                                    )
            if ioverwrite == 0:
                pwd_flex_LPF_xy = np.zeros((nx, ny))
                map_tools.low_pass_filter_zmap(
                                                dx, dy, nx, ny, AOI_clean_np,
                                                rad_search_m, pwd_flex_xy, 
                                                pwd_flex_LPF_xy
                                            )
                if ioutput == 1:
                    file_name = (
                                      "PWD_Flex_LPF_iter"+str(iterID) + "_"
                                    + str(int(rad_search_m/1000)) + "km_"
                                    + str(event_age)+"_event_" + str(event_ID)
                                )
                    map_tools.make_output_file_ZMAP_v4(
                                    output_path, file_name, pwd_flex_LPF_xy, 
                                    nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                    AOI_np
                                )
                DS = pwd_flex_xy - pwd_flex_LPF_xy
            else:  
                DS = pwd_flex_xy - pwd_overwrite_xy
                if ioutput == 1:
                    file_name = (
                                      "PWD_Flex_OVERWRITE_iter"+str(iterID)+"_"
                                    + str(event_age)+"_event_"+str(event_ID)
                                )
                    map_tools.make_output_file_ZMAP_v4(
                                output_path, file_name, pwd_overwrite_xy, 
                                nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                AOI_np
                            )
            # Get the top of salt for this event
            event_index = model.tops_list_bs[index_TOS][14][event_ID]
            TOS_xy = np.copy(model.tops_list_bs[index_TOS][1][event_index])
            flexural_backstripping.fill_undefined(
                                                    TOS_xy, AOItmp, 
                                                    nx, ny, dx, dy
                                                )         
            event_index = model.tops_list_bs[index_BOS][14][event_ID]
            BOS_xy =np.copy(model.tops_list_bs[index_BOS][1][event_index])
            flexural_backstripping.fill_undefined(
                                                    BOS_xy, AOItmp, 
                                                    nx, ny, dx, dy
                                                )
            salt_thick_o = BOS_xy - TOS_xy
            salt_thick_xy = np.zeros((nx,ny))
            #salt_thick_np = salt_thick_i + DS
            vol_current = update_salt_thickenss(
                                                nx, ny, dx, dy, AOI_np, DS,
                                                salt_thick_o, salt_thick_xy
                                            )            
            print("---> Volume (km3) of salt = ", vol_current, 
                                                         " at age ", event_age)
            print("---> Volume (km3) of final salt = ", vol_final)
            print("---> Difference % : ", 
                                         (vol_final-vol_current)/vol_final*100)
            # Update salt thickness
            model.event_dict_bs[event_ID][23] = np.copy(salt_thick_xy)
            #if ioutput == 1:
            file_name = (
                            "Salt_Thick_fromFLEX_iter"+str(iterID)+"_"
                            +str(event_age)+"_event_"+str(event_ID)
                        )
            map_tools.make_output_file_ZMAP_v4(
                                    output_path, file_name, salt_thick_xy, 
                                    nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                    AOI_np
                                )