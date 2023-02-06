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
import numba
from numba import jit
import backstripping_tools
import compaction
import bulk_props
import manage_parallel
import print_funcs


def burial_history(ioutput_main, process, model):
    tt1 = time.time()
    ioutput = 1
    isalt_restore_local = 0
    calculate_burial_history(0, ioutput, isalt_restore_local, model)
    tt2 = time.time()
    print_funcs.print_finfo(
                        ioutput_main, process, 
                        "Calculated 1st pass burial history", tt2-tt1)
    
    tt1 = time.time()
    calculate_forward_max_burial(model)
    tt2 = time.time()
    print_funcs.print_finfo(
                ioutput_main, process, 
                "Calculated 1st pass forward maximum burial history", tt2-tt1)
    
    tt1 = time.time()
    ioutput = 1
    isalt_restore_local = 0
    calculate_burial_history(model.icompact, ioutput, isalt_restore_local, 
                             model)
    tt2 = time.time()
    print_funcs.print_finfo(
                            ioutput_main, process, 
                            "Calculated 2nd pass burial history", tt2-tt1)

    tt1 = time.time()
    imake_output = 0
    ioutput = 1
    backstripping_tools.update_sed_thick_rho_load_tts(imake_output, ioutput, 
                                                      model)
    tt2 = time.time()
    print_funcs.print_finfo(
            ioutput_main, process, 
            "Updated bulk sediment thickness, density, load and tts", tt2-tt1)

    tt1 = time.time()
    calculate_forward_max_burial(model)
    tt2 = time.time()
    print_funcs.print_finfo(
                ioutput_main, process, 
                "Calculated 2nd pass forward maximum  burial history", tt2-tt1)

    tt1 = time.time()
    imake_output = 1
    ioutput = 1
    backstripping_tools.update_sed_thick_rho_load_tts(imake_output, ioutput, 
                                                      model)
    tt2 = time.time()
    print_funcs.print_finfo(
            ioutput_main, process, 
            "Updated bulk sediment thickness, density, load and tts", tt2-tt1)


def residual_subsidence_and_pwd(ioutput_main, process, model):
    
    tt1 = time.time()
    # Residual subsidence is calculated and if 
    # iupdate_PWD == 1 PWD maps and depth maps are updated
    ioutput = 1
    backstripping_tools.calculate_res_sub_and_corrected_PWD(ioutput, model)
    tt2 = time.time()
    print_funcs.print_finfo(
                            ioutput_main, process, 
                            "Updated PWD models", tt2-tt1)    

                           
@jit(nopython=True, cache=True)
def forward_max_burial_loop(
                            jj, nx, ny, ntops, AOI_np, pwd_xy, base_level, 
                            depth_xy, emag_top_xy, icount_grid, 
                            z_submud_all_grid, 
                            maxb_final_A_xy, maxb_final_B_xy):
    for i in range(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            # Only perform calc if within AOI
            if AOI_flag == 1:
                pwd = pwd_xy[i][j]
                # subsea depth of sediment surface (meters)
                z_surf = base_level + pwd
                z_top = depth_xy[i][j] - z_surf                            
                emag_top = emag_top_xy[i][j]                
                ic = icount_grid[i,j,jj]                
                z_submud_all_grid[i,j,jj,ic] = z_top
                icount_grid[i,j,jj] = icount_grid[i,j,jj] + 1                
                z_submud_all_grid[i,j,jj,ic+1] = emag_top
                icount_grid[i,j,jj] = icount_grid[i,j,jj] + 1
                # This includes the effects of erosion and treats 
                # the top as an unconformity
                maxb_final = -1e32
                for m in range(ntops*2):
                    val = z_submud_all_grid[i,j,jj,m]
                    if val > maxb_final:
                        maxb_final = val
                # Updated main arrays
                maxb_final_A_xy[i][j] = maxb_final
                maxb_final_B_xy[i][j] = maxb_final

                            
def calculate_forward_max_burial(model):
    """ Calculate forward maximum burial using Numba
    
    Key Arrays
    ----------
    z_submud_all_grid : float array (nx,ny,ntops,ntops)
        for each grid location and top, an array is defined with submud depth 
        (meters) and erosion magnitude (meters) for all relevent events.
        Note the maximum number of exvents is ntops.
    
    icount_grid : int array (nx,ny,ntops)
        the number of z_top and emag entries for each grid location and top.
    """
    # Unpack model objects for jitted functions
    nx = model.nx
    ny = model.ny
    AOI_np = model.AOI_np
    
    keys = list(model.event_dict_bs.keys())
    ntops = len(model.tops_list_bs)
    z_submud_all_grid = np.zeros((nx, ny, ntops, ntops*2))
    icount_grid = np.zeros((nx, ny, ntops), dtype=int)
    # Loop over all tops
    for mm in range(ntops):
        # Loop from oldest to yoigest tops
        jj = ntops - 1 - mm
        # Loop over each event
        # Loop over events from old to young
        for event_ID, key in enumerate(keys):
            # Get the upper most top for event
            itop_event = model.event_dict_bs[event_ID][2]
            # Does the top exist during this event?
            # We only consider tops that have been deposited
            if jj <= itop_event:
                pwd_xy = model.event_dict_bs[event_ID][5]
                event_index = model.tops_list_bs[jj][14][event_ID]
                depth_xy = model.tops_list_bs[jj][1][event_index]
                emag_top_xy = model.tops_list_bs[jj][3][event_index]
                # meters
                base_level = model.deltaSL_list[event_ID]
                # Main arrays that will be updated
                maxb_final_A_xy = np.copy(
                                       model.tops_list_bs[jj][46][event_index])
                # Note this array controls the thickness of layers
                maxb_final_B_xy = np.copy(model.tops_list_bs[jj][12])
                forward_max_burial_loop(
                            jj, nx, ny, ntops, AOI_np, pwd_xy, base_level, 
                            depth_xy, emag_top_xy, icount_grid, 
                            z_submud_all_grid, 
                            maxb_final_A_xy, maxb_final_B_xy
                        )
                # Update main data structures
                model.tops_list_bs[jj][46][event_index] = np.copy(
                                                               maxb_final_A_xy)
                model.tops_list_bs[jj][12] = np.copy(maxb_final_B_xy)                
    itest_output = 0
    if itest_output == 1:                
        for mm in range(ntops):
            jj = ntops - 1 - mm
            if jj <= itop_event:
                file = open('opt_max_burial_'+str(jj)+".txt", "w+")
                content = str(model.tops_list_bs[jj][12])
                file.write(content)
                file.close()

    
@jit(nopython=True, cache=True)
def reconstruct_from_deposition_event(
                                        iuse_compact, jj, itop_event, 
                                        z_surf, z_top_p1_sm, maxb_event,
                                        thick_p1, emag_top_p1, maxb_event_p1,
                                        z_prev, t_prev, phi_o, c, rho_grain, 
                                        rho_water, iuse_salt, salt_thick
):
    """ Backstrip and reconstruct layer starting from a deposition event
    """
    if jj == itop_event: 
        # Top at the top of sediment pile for deposition event       
        if z_top_p1_sm >= maxb_event_p1 and iuse_compact == 1: 
            
            (
                z_top_n, thick_n, 
                maxb_n, rho_bulk
             ) = decompact(
                            maxb_event_p1, thick_p1, 
                            phi_o, c, rho_grain, rho_water, 
                            z_surf, z_surf, 
                            iuse_salt, salt_thick
                        )
        else:           
            (
                z_top_n, thick_n, 
                maxb_n, rho_bulk
             ) = no_decompact(
                                maxb_event, thick_p1, 
                                phi_o, c, rho_grain, rho_water, 
                                z_surf/2.0, z_surf/2.0, 
                                iuse_salt, salt_thick
                            )       
    else:
        # Top within sediment pile for deposition event
        z_base_prev = z_prev + t_prev       
        if z_top_p1_sm >= maxb_event_p1 and iuse_compact == 1:
            (
                z_top_n, thick_n,
                maxb_n, rho_bulk
             ) = decompact(
                            maxb_event_p1, thick_p1, 
                            phi_o, c, rho_grain, rho_water, 
                            z_surf, z_base_prev, 
                            iuse_salt, salt_thick
                        )
        else:
            (
                z_top_n, thick_n, 
                maxb_n, rho_bulk
             ) = no_decompact(
                                maxb_event, thick_p1, 
                                phi_o, c, rho_grain, rho_water, 
                                z_prev, t_prev, 
                                iuse_salt, salt_thick
                            )    
    return z_top_n, thick_n, maxb_n, rho_bulk


@jit(nopython=True, cache=True)
def reconstruct_from_erosion_event(
                                    jj, itop_event, z_surf, 
                                    thick_p1, emag_top_p1, maxb_event_p1,
                                    z_prev, t_prev, phi_o, c, rho_grain, 
                                    rho_water, iuse_salt, salt_thick
):
    #**********************************
    # Reconstructing from erosion event
    #**********************************
    # We are eroding as we move from event m to m + 1
    # Add eroded thicknesses to copied m + 1 information
    # No need to decompact
    # Get eroded thickness for this top
                    
    # Copy m + 1 information to present event 
    # and add eroded thickness
    if jj == itop_event:
        z_top_n = z_surf
        thick_n = thick_p1 + emag_top_p1
        maxb_n = maxb_event_p1
    else:
        z_top_n = z_prev + t_prev
        thick_n = thick_p1+emag_top_p1
        if iuse_salt == 1:
            thick_n = salt_thick
        maxb_n = maxb_event_p1
    z1n = z_top_n - z_surf
    z2n = z1n + thick_n
    phi_bulk = bulk_props.bulk_porosity_layer(phi_o, c, z1n, z2n)
    rho_bulk = bulk_props.bulk_rho_layer(phi_bulk, rho_grain, rho_water)
    return z_top_n, thick_n, maxb_n, rho_bulk


@jit(nopython=True, cache=True)    
def no_decompact(
                    maxb_event, thick_p1, 
                    phi_o, c, rho_grain, rho_water, 
                    z_prev, t_prev, 
                    iuse_salt, salt_thick
):
    z_top_n = z_prev + t_prev
    thick_n = thick_p1
    if iuse_salt == 1:
        thick_n = salt_thick
    maxb_n = maxb_event
    z1n = maxb_n
    z2n = z1n + thick_n
    phi_bulk = bulk_props.bulk_porosity_layer(
                                            phi_o, c, 
                                            z1n, z2n
                                        )    
    rho_bulk = bulk_props.bulk_rho_layer(
                                            phi_bulk, 
                                            rho_grain, 
                                            rho_water
                                        )    
    return z_top_n, thick_n, maxb_n, rho_bulk


@jit(nopython=True, cache=True)
def decompact(
                maxb_event_p1, thick_p1, 
                phi_o, c, rho_grain, rho_water, 
                z_surf, z_base_prev, 
                iuse_salt, salt_thick
):
    z1o = maxb_event_p1 # use maximum compaction state
    z2o = z1o + thick_p1
    # get submud depth
    z1n = z_base_prev - z_surf
    decom_thick = compaction.compact_or_decompact(phi_o, c, z1o, z2o, z1n)
    if iuse_salt == 1:
        decom_thick = salt_thick
    z2n = z1n + decom_thick
    phi_bulk = bulk_props.bulk_porosity_layer(phi_o, c, z1n, z2n)
    rho_bulk = bulk_props.bulk_rho_layer(phi_bulk, rho_grain, rho_water)
    z_top_n = z1n + z_surf
    thick_n = decom_thick                            
    maxb_n = z_top_n - z_surf
    return z_top_n, thick_n, maxb_n, rho_bulk


#@jit(nopython=True, parallel=True)
def burial_history_final_event(
                                nx, ny, rho_water, base_level, 
                                AOI_xy, pwd_xy, phi_o_xy, 
                                z_top_xy, thick_xy, rho_grain_xy, 
                                decay_depth_xy, maxb_xy, 
                                maxb_update_xy, rho_bulk_xy
):
    """
    Initialize last event maximum burial with final maximum burial and 
    calculate bulk density for the layer. 
    """
    for i in numba.prange(nx):         
        for j in range(ny):            
            AOI_flag = AOI_xy[i][j]            
            if AOI_flag == 1:
                pwd = pwd_xy[i][j]
                z_surf = base_level + pwd                
                # input in %, function needs fraction
                phi_o = phi_o_xy[i][j]/100.0
                # entry is in km, function needs 1/m
                c = 1.0/(decay_depth_xy[i][j]*1000)
                rho_grain = rho_grain_xy[i][j]                
                maxb = maxb_xy[i][j]
                z_top = z_top_xy[i][j]
                thick = thick_xy[i][j]
                z1n = z_top - z_surf
                z2n = z1n + thick
                phi_bulk = bulk_props.bulk_porosity_layer(phi_o, c, z1n, z2n)
                rho_bulk = bulk_props.bulk_rho_layer(
                                                        phi_bulk, rho_grain, 
                                                        rho_water
                                                    )
                maxb_update_xy[i][j] = maxb
                rho_bulk_xy[i][j] = rho_bulk


#@jit(nopython=True, parallel=True)
def burial_history_depo_and_ero_events(
                                    jj, nx, ny, rho_water, iuse_compact,
                                    base_level, base_level_p1,
                                    index_TOS, itop_event,
                                    AOI_xy, pwd_xy, pwd_xy_p1, emag_p1_xy,
                                    salt_thick_xy, z_top_p1_xy, thick_p1_xy,
                                    maxb_event_p1_xy, maxb_update_xy,
                                    phi_o_xy, decay_depth_xy, rho_grain_xy,
                                    emag_top_p1_xy, z_prev_xy, t_prev_xy,
                                    z_top_n_xy, thick_n_xy, maxb_n_xy,
                                    rho_bulk_n_xy, iuse_salt
):
    for i in numba.prange(nx):
        for j in range(ny):
            AOI_flag = AOI_xy[i][j]
            if AOI_flag == 1:
                if iuse_salt == 1:
                    salt_thick = salt_thick_xy[i][j]
                else:
                    salt_thick = 0.0 
                #**********************
                # Compaction properties
                #**********************
                # input in %, function needs fraction
                phi_o = phi_o_xy[i][j]/100.0
                # entry is in km, function needs 1/m
                c = 1.0/(decay_depth_xy[i][j]*1000)
                rho_grain = rho_grain_xy[i][j]
                #****************************************
                # Information for top from current event
                #****************************************                  
                pwd = pwd_xy[i][j]
                z_surf = base_level + pwd
                #****************************************
                # Information for top from previous event
                #****************************************               
                z_prev = z_prev_xy[i][j]
                t_prev = t_prev_xy[i][j]
                #*************************************
                # Information for top from event m + 1
                #*************************************
                pwd_p1 = pwd_xy_p1[i][j]
                z_surf_p1 = base_level_p1 + pwd_p1
                emag_p1 = emag_p1_xy[i][j]
                z_top_p1 = z_top_p1_xy[i][j]
                z_top_p1_sm = z_top_p1 - z_surf_p1
                thick_p1 = thick_p1_xy[i][j]
                maxb_event_p1 = maxb_event_p1_xy[i][j]
                maxb_event = maxb_event_p1
                emag_top_p1 = emag_top_p1_xy[i][j]
                if emag_p1 > 0.0:
                    ( 
                     z_top_n, thick_n, 
                     maxb_n, rho_bulk
                    ) = reconstruct_from_erosion_event(
                                        jj, itop_event, z_surf, 
                                        thick_p1, emag_top_p1, maxb_event_p1,
                                        z_prev, t_prev, phi_o, c, rho_grain, 
                                        rho_water, iuse_salt, salt_thick
                                    )
                else: 
                    ( 
                     z_top_n, thick_n, 
                     maxb_n, rho_bulk
                    ) = reconstruct_from_deposition_event(
                                        iuse_compact, jj, itop_event, 
                                        z_surf, z_top_p1_sm, maxb_event,
                                        thick_p1, emag_top_p1, maxb_event_p1,
                                        z_prev, t_prev, phi_o, c, rho_grain, 
                                        rho_water, iuse_salt, salt_thick
                                    )
                maxb_update_xy[i][j] = maxb_event
                z_top_n_xy[i][j] = z_top_n
                thick_n_xy[i][j] = thick_n
                maxb_n_xy[i][j] = maxb_n
                rho_bulk_n_xy[i][j] = rho_bulk 

    
def calculate_burial_history(iuse_compact, ioutput, isalt_restore, model):
    """ Calculate burial history using a hydrostatic decompaction model
    
    """
    # Unpack model object for jitted functions
    rho_water = model.rho_water
    nx = model.nx
    ny = model.ny
    AOI_np = model.AOI_np
    salt_layer_index = model.salt_layer_index

    keys = list(model.event_dict_bs.keys())    
    nevents_bs = len(keys)
    event_ID_list_bs = keys[:]
    event_ID_last_bs = event_ID_list_bs[nevents_bs-1]
    ntops = len(model.tops_list_bs)
    # event_ID for pre-salt deposition event; event ID is zero for first even
    event_ID_salt = ntops - salt_layer_index    
    try:
        event_age_salt = model.event_dict_bs[event_ID_salt][0]
    except:
        event_age_salt = -99999
        print("!! WARNING : salt_layer_index does not have input : "
              "event_age_salt : ", event_age_salt)
    itype3D = model.itype3D
    (
        burial_history_final_event_active
    ) = manage_parallel.manage_parallel(burial_history_final_event, itype3D)
    (
        burial_history_depo_and_ero_events_active
    ) = manage_parallel.manage_parallel(
                                burial_history_depo_and_ero_events, itype3D)
    index_TOS = int(ntops - salt_layer_index)
    for kk, key in enumerate(keys):
        # Loop over events from young to old
        mm = nevents_bs - 1 - kk        
        event_ID = mm
        # Get the upper most top for event
        itop_event = model.event_dict_bs[event_ID][2]
        pwd_xy = np.copy(model.event_dict_bs[event_ID][5])
        # Assume delta base-level is 0 for now
        base_level = model.deltaSL_list[event_ID]
        salt_thick_xy = np.copy(model.event_dict_bs[event_ID][23])
        # No need to decompact for final event
        if event_ID == event_ID_last_bs:
            # Initialize last event maximum burial with 
            # final maximum burial and calculate bulk density for
            # tops_list_bs[jj][17][event_index]
            for ii in range(ntops):
                jj = ntops-1-ii
                event_index = model.tops_list_bs[jj][14][event_ID]
                # Unpack xy maps
                phi_o_xy = np.copy(model.tops_list_bs[jj][33])
                rho_grain_xy = np.copy(model.tops_list_bs[jj][32])
                decay_depth_xy = np.copy(model.tops_list_bs[jj][34])
                z_top_xy = np.copy(model.tops_list_bs[jj][1][event_index])
                thick_xy = np.copy(model.tops_list_bs[jj][2][event_index])
                maxb_xy = np.copy(model.tops_list_bs[jj][12])
                # Arrays that will be updated
                maxb_update_xy = np.copy(
                                    model.tops_list_bs[jj][13][event_index])
                rho_bulk_xy = np.copy(model.tops_list_bs[jj][17][event_index])  
                burial_history_final_event_active(
                                            nx, ny, rho_water, base_level, 
                                            AOI_np, pwd_xy, phi_o_xy,
                                            z_top_xy, thick_xy, rho_grain_xy,
                                            decay_depth_xy, maxb_xy,
                                            maxb_update_xy, rho_bulk_xy,
                                        )
                # Update main data structures
                model.tops_list_bs[jj][17][event_index] = np.copy(rho_bulk_xy)
                model.tops_list_bs[jj][13][event_index] = np.copy(
                                                                maxb_update_xy)
        else:
            pwd_xy_p1 = model.event_dict_bs[event_ID + 1][5]
            base_level_p1 = model.deltaSL_list[event_ID + 1]
            emag_p1_xy = model.event_dict_bs[event_ID + 1][3]
            for ii in range(ntops):
                # Looping from young to old
                jj = ntops - 1 - ii
                # We only consider tops that have been deposited
                if jj <= itop_event:
                    phi_o_xy = np.copy(model.tops_list_bs[jj][33])
                    decay_depth_xy = np.copy(model.tops_list_bs[jj][34])
                    rho_grain_xy = np.copy(model.tops_list_bs[jj][32])
                    if jj != itop_event:
                        # event index for previous event
                        (
                            event_index_prev
                        ) = model.tops_list_bs[jj + 1][14][event_ID]   
                        (
                            z_prev_xy
                        ) = np.copy(
                               model.tops_list_bs[jj + 1][1][event_index_prev])
                        (
                            t_prev_xy
                        ) = np.copy(
                               model.tops_list_bs[jj + 1][2][event_index_prev])
                    else:
                        z_prev_xy = np.zeros((nx,ny))
                        t_prev_xy = np.zeros((nx,ny))
                    # event index for current event
                    event_index = model.tops_list_bs[jj][14][event_ID]        
                    # event index for next event (plus 1)
                    event_index_p1 = event_index + 1
                    z_top_p1_xy = np.copy(
                                    model.tops_list_bs[jj][1][event_index_p1])
                    thick_p1_xy = np.copy(
                                     model.tops_list_bs[jj][2][event_index_p1])
                    (
                        maxb_event_p1_xy
                    ) = np.copy(model.tops_list_bs[jj][13][event_index_p1])
                    (
                        emag_top_p1_xy
                    ) = np.copy(model.tops_list_bs[jj][3][event_index_p1])
                    # Arrays that will be updated
                    maxb_update_xy = np.copy(
                                       model.tops_list_bs[jj][13][event_index])
                    z_top_n_xy = np.copy(
                                        model.tops_list_bs[jj][1][event_index])
                    thick_n_xy = np.copy(
                                        model.tops_list_bs[jj][2][event_index])
                    maxb_n_xy = np.copy(
                                       model.tops_list_bs[jj][13][event_index])
                    rho_bulk_n_xy = np.copy(
                                       model.tops_list_bs[jj][17][event_index])
                    if jj == index_TOS and isalt_restore == 1:
                        iuse_salt = 1
                    else:
                        iuse_salt = 0                 
                    burial_history_depo_and_ero_events_active(
                                    jj, nx, ny, rho_water, iuse_compact,
                                    base_level, base_level_p1,
                                    index_TOS, itop_event,
                                    AOI_np, pwd_xy, pwd_xy_p1, emag_p1_xy,
                                    salt_thick_xy, z_top_p1_xy, thick_p1_xy,
                                    maxb_event_p1_xy, maxb_update_xy,
                                    phi_o_xy, decay_depth_xy, rho_grain_xy,
                                    emag_top_p1_xy, z_prev_xy, t_prev_xy,
                                    z_top_n_xy, thick_n_xy, maxb_n_xy,
                                    rho_bulk_n_xy, iuse_salt
                                )
                    # Update main data structures
                    model.tops_list_bs[jj][13][event_index] = np.copy(
                                                                maxb_update_xy)
                    model.tops_list_bs[jj][1][event_index] = np.copy(
                                                                    z_top_n_xy)
                    model.tops_list_bs[jj][2][event_index] = np.copy(
                                                                    thick_n_xy)
                    model.tops_list_bs[jj][13][event_index] = np.copy(
                                                                     maxb_n_xy)
                    model.tops_list_bs[jj][17][event_index] = np.copy(
                                                                 rho_bulk_n_xy)
