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
from numba import jit
import numba
import finite_rifting
import lithospheric_stretching
import math_tools
import manage_parallel
import print_funcs


@jit(nopython = True, cache=True)
def build_interp_arrays(
                        iuse_flexure, ts_Myr, ttsub_t, hf_t, 
                        q_thermal_vec, q_crustal_vec, Te_vec
):
    nt = ts_Myr.size
    ttsub_t_xy = np.zeros((nt,2))
    hf_t_xy = np.zeros((nt,2))
    q_thermal_t_xy = np.zeros((nt,2))
    q_crustal_t_xy = np.zeros((nt,2))
    Te_t_xy = np.zeros((nt,2))
    for ii in range(nt):
        ttsub_t_xy[ii][0] = ts_Myr[ii]
        ttsub_t_xy[ii][1] = ttsub_t[ii]
        hf_t_xy[ii][0] = ts_Myr[ii]
        hf_t_xy[ii][1] = hf_t[ii]
        if iuse_flexure == 1:
            q_thermal_t_xy[ii][0] = ts_Myr[ii]
            q_thermal_t_xy[ii][1] = q_thermal_vec[ii]                        
            q_crustal_t_xy[ii][0] = ts_Myr[ii]
            q_crustal_t_xy[ii][1] = q_crustal_vec[ii]    
            Te_t_xy[ii][0] = ts_Myr[ii]
            Te_t_xy[ii][1] = Te_vec[ii]
    return nt, ttsub_t_xy, hf_t_xy, q_thermal_t_xy, q_crustal_t_xy, Te_t_xy


@jit(nopython = True, cache=True)
def interpolate_for_event_arrays(
                                    i, j, nt, k_bulk, T_base, 
                                    tc_initial, tm_initial,
                                    hf_conv, event_IDs, event_ages, start_age1,
                                    ttsub_t_xy, hf_t_xy, q_thermal_t_xy,
                                    q_crustal_t_xy, Te_t_xy, iuse_flexure,
                                    Telastic, ttsub_events_xy, 
                                    hf_tot_events_xy, hf_anom_events_xy,
                                    q_crustal_events_xy, q_thermal_events_xy,
                                    Te_events_xy, bghf_xy                                 
):
    hf_ini = k_bulk*T_base/(tc_initial + tm_initial)*1000.0
    for event_ID in event_IDs:
        age = event_ages[event_ID]
#        if age < 1:
#            age = 1
        tMyr = start_age1 - age
        if tMyr >= 0:
            ttsub = math_tools.linear_interp_v2(
                                            nt, tMyr, ttsub_t_xy
                                        )
            hf = hf_conv*math_tools.linear_interp_v2(
                                                nt, tMyr, hf_t_xy
                                            )              
            if iuse_flexure == 1:
                q_thermal = math_tools.linear_interp_v2(
                                        nt, tMyr, q_thermal_t_xy
                                    )                            
                q_crustal = math_tools.linear_interp_v2(
                                        nt, tMyr, q_crustal_t_xy
                                    )
                Te = math_tools.linear_interp_v2(
                                                nt, tMyr, Te_t_xy
                                            )
        else:
            ttsub = 0.0
            hf = hf_ini
            if iuse_flexure == 1:
                q_thermal = 0
                q_crustal = 0
                Te = (tc_initial + tm_initial)/T_base*Telastic
        ttsub_events_xy[event_ID][i][j] = ttsub
        hf_tot_events_xy[event_ID][i][j] = (
                                              hf 
                                            - hf_ini 
                                            + bghf_xy[i][j]
                                            )
        hf_anom_events_xy[event_ID][i][j] = hf - hf_ini
        if iuse_flexure == 1:
            q_crustal_events_xy[event_ID][i][j] = q_crustal
            q_thermal_events_xy[event_ID][i][j] = q_thermal
            Te_events_xy[event_ID][i][j] = Te


@jit(nopython = True, cache=True)
def update_event_arrays_loop(
                            iuse_flexure, event_ID, nx, ny, AOI_np, 
                            ttsub_xy, hf_tot_xy, hf_anom_xy,
                            q_crustal_xy, q_thermal_xy, Te_xy, 
                            ttsub_events_xy, hf_tot_events_xy,
                            hf_anom_events_xy, q_crustal_events_xy,
                            q_thermal_events_xy, Te_events_xy
):
    for i in range(nx): 
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:
                ttsub_xy[i][j] = ttsub_events_xy[event_ID][i][j]
                hf_tot_xy[i][j] = hf_tot_events_xy[event_ID][i][j] 
                hf_anom_xy[i][j] = hf_anom_events_xy[event_ID][i][j]
                if iuse_flexure == 1:
                    q_crustal_xy[i][j] = (
                                    q_crustal_events_xy[event_ID][i][j])
                    q_thermal_xy[i][j] = (
                                    q_thermal_events_xy[event_ID][i][j])
                    Te_xy[i][j] = Te_events_xy[event_ID][i][j]


@jit(nopython = True, cache=True)
def interpolate_for_TTS_FW_arrays(
                                    i, j, hf_conv, k_bulk, T_base, 
                                    tc_initial, tm_initial,
                                    interp_IDs, interp_ages, start_age1,
                                    nt, ttsub_t_xy, hf_t_xy,
                                    TTS_FW_events_xy, TTS_FW_hf_event_xy,
                                    TTS_FW_hf_anom_xy
                                    
):
    hf_ini = k_bulk*T_base/(tc_initial + tm_initial)*1000.0
    for interp_ID in interp_IDs:
        age = interp_ages[interp_ID]
        tMyr = start_age1 - age
        if tMyr >= 0:
            ttsub = math_tools.linear_interp_v2(
                                                nt,
                                                tMyr, 
                                                ttsub_t_xy
                                            )
            hf = hf_conv*math_tools.linear_interp_v2(
                                                        nt, 
                                                        tMyr, 
                                                        hf_t_xy
                                                    )
        else:
            ttsub = 0.0
            hf = hf_ini
        TTS_FW_events_xy[interp_ID][i][j] = ttsub
        TTS_FW_hf_event_xy[interp_ID][i][j] = hf
        TTS_FW_hf_anom_xy[interp_ID][i][j] = hf - hf_ini

                  
def update_event_arrays(
                        model, event_IDs, ttsub_events_xy, 
                        hf_tot_events_xy, hf_anom_events_xy,
                        q_crustal_events_xy, q_thermal_events_xy,
                        Te_events_xy
):
    iuse_flexure = model.iuse_flexure
    nx = model.nx
    ny = model.ny
    AOI_np = model.AOI_np
    for event_ID in event_IDs:
        ttsub_xy = np.copy(model.event_dict_bs[event_ID][14])
        hf_tot_xy = np.copy(model.event_dict_bs[event_ID][9])
        hf_anom_xy = np.copy(model.event_dict_bs[event_ID][15])
        if iuse_flexure == 1:
            q_crustal_xy = np.copy(model.event_dict_bs[event_ID][16])
            q_thermal_xy = np.copy(model.event_dict_bs[event_ID][17])
            Te_xy = np.copy(model.event_dict_bs[event_ID][19])
        else:
            q_crustal_xy = np.zeros((nx,ny))
            q_thermal_xy = np.zeros((nx,ny))
            Te_xy = np.zeros((nx,ny))
        update_event_arrays_loop(
                                    iuse_flexure, event_ID, nx, ny, AOI_np, 
                                    ttsub_xy, hf_tot_xy, hf_anom_xy,
                                    q_crustal_xy, q_thermal_xy, Te_xy, 
                                    ttsub_events_xy, hf_tot_events_xy,
                                    hf_anom_events_xy, q_crustal_events_xy,
                                    q_thermal_events_xy, Te_events_xy
                                )
        # Update main data strcutres
        model.event_dict_bs[event_ID][14] = np.copy(ttsub_xy)
        model.event_dict_bs[event_ID][9] = np.copy(hf_tot_xy)
        model.event_dict_bs[event_ID][15] = np.copy(hf_anom_xy)
        if iuse_flexure == 1:
            model.event_dict_bs[event_ID][16]= np.copy(q_crustal_xy)
            model.event_dict_bs[event_ID][17] = np.copy(q_thermal_xy)
            model.event_dict_bs[event_ID][19] = np.copy(Te_xy)


def update_TTS_FW_dict(
                        model, interp_IDs, 
                        TTS_FW_events_xy, TTS_FW_hf_event_xy,
                        TTS_FW_hf_anom_event_xy
):
    nx = model.nx
    ny = model.ny
    AOI_np = model.AOI_np
    for interp_ID in interp_IDs:
        for i in range(nx): 
            for j in range(ny):
                AOI_flag = AOI_np[i][j]
                if AOI_flag == 1:        
                    ttsub = TTS_FW_events_xy[interp_ID][i][j]
                    hf = TTS_FW_hf_event_xy[interp_ID][i][j]
                    hf_anom = TTS_FW_hf_anom_event_xy[interp_ID][i][j]
                    model.TTS_FW_dict[interp_ID][1][i][j] = ttsub
                    model.TTS_FW_dict[interp_ID][2][i][j] = hf
                    model.TTS_FW_dict[interp_ID][3][i][j] = hf_anom
                    

@jit(nopython=True, cache=True)
def clean_crustal_vec(ts_Myr, q_crustal_vec):
    ifind_last = 0
    ntimes = ts_Myr.size
    for mm in range(ntimes):
        t = ts_Myr[mm]
        if mm > 0 and t == 0.0 and ifind_last == 0:
            ifind_last = 1
            q_last = q_crustal_vec[mm-1]
        if mm > 0 and t == 0.0:
            q_crustal_vec[mm] = q_last           

      
#@jit(nopython=True, parallel=True)
def calculate_sub_and_hf_maps_loop(
                    nx, ny, AOI_np, xth_xy, nphases_xy,
                    iuse_numerical_rift, iuse_flexure, HP_itype, rift_itype, 
                    icon_kappa, irho_var, iup_in_air, itype_rho_a, 
                    start_age1_xy, end_age1_xy, rift_mag1_xy, mantle_fac1_xy,
                    start_age2_xy, end_age2_xy, rift_mag2_xy, mantle_fac2_xy,
                    start_age3_xy, end_age3_xy, rift_mag3_xy, mantle_fac3_xy,
                    dt_Myr, dt_rift_Myr, dt_out, 
                    k_crust, k_mantle, cp_crust, cp_mantle, alpha_crust, 
                    alpha_mantle, rho_crust, rho_mantle, Q_crust, Q_mantle, 
                    Ao, ar, rho_sea, k_bulk, cp_bulk, alpha_bulk,
                    Llith, z_moho, dz_lith, T_top, T_bottom, T_base,
                    L_crust_ref, L_lith_ref, rho_crust_ref, rho_mantle_ref, 
                    Telastic, tc_initial, tm_initial,
                    zs_np, Tn_np, nnodes, event_IDs, event_ages, 
                    ttsub_events_xy, hf_tot_events_xy, hf_anom_events_xy,
                    q_crustal_events_xy, q_thermal_events_xy, Te_events_xy, 
                    bghf_xy, interp_IDs, interp_ages, TTS_FW_events_xy, 
                    TTS_FW_hf_event_xy, TTS_FW_hf_anom_event_xy
                    
):
    #**************************************************************************
    # For each x-y coordinate calculate deformation, temperature and heat flow
    #**************************************************************************
    for i in numba.prange(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:
                tc_final = xth_xy[i][j]
                nphases = int(nphases_xy[i][j])
                start_age1 = start_age1_xy[i][j]            
                end_age1 = end_age1_xy[i][j]
                rift_mag1 = rift_mag1_xy[i][j] 
                mantle_fac1 = mantle_fac1_xy[i][j]
                start_age2 = start_age2_xy[i][j]
                end_age2 =end_age2_xy[i][j]
                rift_mag2 = rift_mag2_xy[i][j]
                mantle_fac2 = mantle_fac2_xy[i][j]
                start_age3 = start_age3_xy[i][j]
                end_age3 =end_age3_xy[i][j]
                rift_mag3 = rift_mag3_xy[i][j]
                mantle_fac3 = mantle_fac3_xy[i][j]
                (
                    start_age1, end_age1, rift_mag1, mantle_fac1,
                    start_age2, end_age2, rift_mag2, mantle_fac2,
                    start_age3, end_age3, rift_mag3, mantle_fac3,
                    delta_1, delta_2, delta_3, 
                    Beta_1, Beta_2, Beta_3, 
                    rift_time_1, rift_time_2, rift_time_3, 
                    t_pr1, t_pr2, t_pr3, tc_change, 
                    tc_change_1, tc_change_2, tc_change_3, 
                    tc_final_1, tc_final_2, tc_final_3
                ) = get_rift_parameters(
                        start_age1, end_age1, rift_mag1, mantle_fac1,
                        start_age2, end_age2, rift_mag2, mantle_fac2,
                        start_age3, end_age3, rift_mag3, mantle_fac3,
                        tc_initial, tc_final, nphases
                        )
                hf_conv = 1.0
                if iuse_numerical_rift == 1:
                    hf_conv = -1.0
                    (
                        number_of_timesteps,
                        ts_Myr, hf_t, 
                        ttsub_t, Te_vec, 
                        q_crustal_vec, q_thermal_vec
                    ) = finite_rifting.finite_rifting(
                                dt_Myr, dt_rift_Myr, dt_out, 
                                k_crust, k_mantle, cp_crust, cp_mantle, 
                                alpha_crust, alpha_mantle, 
                                rho_crust, rho_mantle, rho_sea, Llith, z_moho, 
                                dz_lith, T_top, T_bottom, 
                                HP_itype, Q_crust, Q_mantle, Ao, ar, 
                                rift_itype, nphases, delta_1, Beta_1, 
                                rift_time_1, t_pr1, delta_2, Beta_2, 
                                rift_time_2, t_pr2, delta_3, Beta_3, 
                                rift_time_3, t_pr3, icon_kappa, irho_var, 
                                iup_in_air, itype_rho_a, 
                                L_crust_ref, L_lith_ref, 
                                rho_crust_ref, rho_mantle_ref, Telastic
                                )                    
                else:
                    (
                        ts_Myr, hf_t, ttsub_t,
                        q_thermal_vec, q_crustal_vec, 
                        Te_vec
                    ) = lithospheric_stretching.analytical_rifting(
                            dt_Myr, delta_1, Beta_1, start_age1, 
                            end_age1, tc_initial, tm_initial, rho_sea, 
                            rho_crust, rho_mantle, k_bulk, cp_bulk, 
                            alpha_bulk, T_bottom, iup_in_air, itype_rho_a, 
                            Telastic, iuse_flexure, zs_np, Tn_np, 
                            nnodes
                            )
                # One non-critical clean up is to make sure q_crustal_vec 
                # values out of time range are set equal to last event
                # so clean up is not required
                clean_crustal_vec(ts_Myr, q_crustal_vec)
                (
                    nt, ttsub_t_xy, hf_t_xy, 
                    q_thermal_t_xy, q_crustal_t_xy, 
                    Te_t_xy
                ) = build_interp_arrays(
                                        iuse_flexure, ts_Myr, ttsub_t, hf_t, 
                                        q_thermal_vec, q_crustal_vec, Te_vec
                                        )
                interpolate_for_event_arrays(
                                    i, j, nt, k_bulk, T_base, 
                                    tc_initial, tm_initial,
                                    hf_conv, event_IDs, event_ages, start_age1,
                                    ttsub_t_xy, hf_t_xy, q_thermal_t_xy,
                                    q_crustal_t_xy, Te_t_xy, iuse_flexure,
                                    Telastic, ttsub_events_xy, 
                                    hf_tot_events_xy, hf_anom_events_xy,
                                    q_crustal_events_xy, q_thermal_events_xy,
                                    Te_events_xy, bghf_xy
                                    )
                interpolate_for_TTS_FW_arrays(
                                    i, j, hf_conv, k_bulk, T_base, 
                                    tc_initial, tm_initial,
                                    interp_IDs, interp_ages, start_age1,
                                    nt, ttsub_t_xy, hf_t_xy,
                                    TTS_FW_events_xy, TTS_FW_hf_event_xy,
                                    TTS_FW_hf_anom_event_xy
                                    )
                

def calculate_sub_and_hf_maps(ioutput_main, process, model, xth_xy):
    # Unpack model objects for jitted functions
    nx = model.nx
    ny = model.ny
    AOI_np = model.AOI_np
    tc_initial = model.tc_initial
    tm_initial = model.tm_initial
    rho_water = model.rho_water
    rho_crust_1 = model.rho_crust
    rho_mantle_1 = model.rho_mantle
    k_bulk = model.k_bulk
    cp_bulk = model.cp_bulk
    alpha_bulk = model.alpha_bulk
    T_base = model.T_base
    HP_itype = model.HP_itype
    Q_crust = model.Q_crust
    Q_mantle = model.Q_mantle
    Ao = model.Ao
    ar = model.ar
    L_crust_ref = model.L_crust_ref
    L_lith_ref = model.L_lith_ref
    rho_crust_ref = model.rho_crust_ref 
    rho_mantle_ref = model.rho_mantle_ref
    dt_Myr = model.dt_Myr
    dt_rift_Myr = model.dt_rift_Myr
    dz_lith = model.dz_lith 
    #TTS_FW_dict = model.TTS_FW_dict
    PWD_interp_ages = model.PWD_interp_ages 
    nphases_xy = model.nphases_xy
    start_age1_xy = model.start_age1_xy
    end_age1_xy = model.end_age1_xy
    start_age2_xy = model.start_age2_xy
    end_age3_xy = model.end_age3_xy 
    start_age3_xy = model.start_age3_xy
    end_age2_xy = model.end_age2_xy
    rift_mag1_xy = model.rift_mag1_xy
    rift_mag2_xy = model.rift_mag2_xy
    rift_mag3_xy = model.rift_mag3_xy 
    mantle_fac1_xy = model.mantle_fac1_xy
    mantle_fac2_xy = model.mantle_fac2_xy
    mantle_fac3_xy = model.mantle_fac3_xy
    itype_rho_a = model.itype_rho_a
    rift_itype = model.rift_itype
    iuse_numerical_rift = model.iuse_numerical_rift
    bghf_xy = model.bghf_xy
    Telastic = model.temp_elastic
    iuse_flexure = model.iuse_flexure
                
    tt1 = time.time()
    n_interp_ages = len(PWD_interp_ages)
    interp_ages = np.zeros((n_interp_ages))
    for k, PWD_age in enumerate(PWD_interp_ages): 
        # Looping over events from oldest to youngest
        dum_xy = np.zeros((nx,ny))
        dum2_xy = np.zeros((nx,ny))
        dum3_xy = np.zeros((nx,ny))
        model.TTS_FW_dict[k] = [PWD_age, dum_xy, dum2_xy, dum3_xy]
        interp_ages[k] = PWD_age
    interp_IDs = np.asarray(list(model.TTS_FW_dict.keys()))
    event_IDs = np.asarray(list(model.event_dict_bs.keys()))
    nevents = event_IDs.size
    event_ages = np.zeros((nevents))
    for event_ID in event_IDs:
        # Inititalize heat flow and subsidence maps
        hf_xy = np.zeros((nx,ny))
        ttsub_xy = np.zeros((nx,ny))
        model.event_dict_bs[event_ID][14] = np.copy(ttsub_xy)
        model.event_dict_bs[event_ID][9] = np.copy(hf_xy)
        age = model.event_dict_bs[event_ID][0]
        event_ages[event_ID] = age
    ttsub_events_xy = np.zeros((nevents, nx, ny))
    hf_tot_events_xy = np.zeros((nevents, nx, ny))
    hf_anom_events_xy = np.zeros((nevents, nx, ny))
    q_crustal_events_xy = np.zeros((nevents, nx, ny))
    q_thermal_events_xy = np.zeros((nevents, nx, ny))
    Te_events_xy = np.zeros((nevents, nx, ny))
    TTS_FW_events_xy = np.zeros((n_interp_ages, nx, ny))
    TTS_FW_hf_event_xy = np.zeros((n_interp_ages, nx, ny))
    TTS_FW_hf_anom_event_xy = np.zeros((n_interp_ages, nx, ny))
    # Output time step in Myr
    dt_out = dt_Myr*2.0
    k_crust = k_bulk
    k_mantle = k_bulk
    cp_crust = cp_bulk
    cp_mantle = cp_bulk
    alpha_crust = alpha_bulk
    alpha_mantle = alpha_bulk
    rho_crust = rho_crust_1
    rho_mantle = rho_mantle_1
    rho_sea = rho_water
    Llith = tc_initial+tm_initial
    z_moho = tc_initial
    T_top = 0.0
    # Force constant thermal diffusivity
    kappa_mantle = k_mantle/rho_mantle/cp_mantle    
    cp_crust = k_crust/rho_crust/kappa_mantle    
    T_bottom = T_base
    # Force constant thermal diffusivity
    icon_kappa = 1
    irho_var = 0
    iup_in_air = 1    
    # Initialize 1D grid for calculating thermal load and Te
    # Only do this once to speed up raw python
    dzz = 500.0
    L = tc_initial + tm_initial
    nnodes = int(L/dzz) + 1
    zs_np = np.zeros((nnodes))
    Tn_np = np.zeros((nnodes))
    for i in range(nnodes):
        zD = float(i)*dzz
        zs_np[i]=zD/1000.0
    itype3D = model.itype3D
    calculate_sub_and_hf_maps_loop_active = manage_parallel.manage_parallel(
                                       calculate_sub_and_hf_maps_loop, itype3D)       
    calculate_sub_and_hf_maps_loop_active(
                    nx, ny, AOI_np, xth_xy, nphases_xy,
                    iuse_numerical_rift, iuse_flexure, HP_itype, rift_itype, 
                    icon_kappa, irho_var, iup_in_air, itype_rho_a, 
                    start_age1_xy, end_age1_xy, rift_mag1_xy, mantle_fac1_xy,
                    start_age2_xy, end_age2_xy, rift_mag2_xy, mantle_fac2_xy,
                    start_age3_xy, end_age3_xy, rift_mag3_xy, mantle_fac3_xy,
                    dt_Myr, dt_rift_Myr, dt_out, 
                    k_crust, k_mantle, cp_crust, cp_mantle, alpha_crust, 
                    alpha_mantle, rho_crust, rho_mantle, Q_crust, Q_mantle, 
                    Ao, ar, rho_sea, k_bulk, cp_bulk, alpha_bulk,
                    Llith, z_moho, dz_lith, T_top, T_bottom, T_base,
                    L_crust_ref, L_lith_ref, rho_crust_ref, rho_mantle_ref, 
                    Telastic, tc_initial, tm_initial,
                    zs_np, Tn_np, nnodes, event_IDs, event_ages, 
                    ttsub_events_xy, hf_tot_events_xy, hf_anom_events_xy,
                    q_crustal_events_xy, q_thermal_events_xy, Te_events_xy, 
                    bghf_xy, interp_IDs, interp_ages, TTS_FW_events_xy, 
                    TTS_FW_hf_event_xy, TTS_FW_hf_anom_event_xy
                    )   
    update_event_arrays(
                            model, event_IDs, ttsub_events_xy, 
                            hf_tot_events_xy, hf_anom_events_xy,
                            q_crustal_events_xy, q_thermal_events_xy,
                            Te_events_xy
                            )
    update_TTS_FW_dict(
                        model, interp_IDs, 
                        TTS_FW_events_xy, TTS_FW_hf_event_xy,
                        TTS_FW_hf_anom_event_xy
                        )
    tt2 = time.time()
    print_funcs.print_finfo(
                    ioutput_main, process, 
                    "Calculated subsidence and heat flow maps", tt2-tt1)


@jit(nopython = True, cache=True)
def apply_rift_param_limits(
                            start_age1, end_age1, rift_mag1, mantle_fac1,
                            start_age2, end_age2, rift_mag2, mantle_fac2,
                            start_age3, end_age3, rift_mag3, mantle_fac3
):
    if start_age1 < 0.0:
        start_age1 = 200.0    
    if end_age1 < 0.0:
        end_age1 = 150.0    
    if rift_mag1 < 0.0:
        rift_mag1 = 1.0    
    if mantle_fac1 < 0.0:
        mantle_fac1 = 1.0
    if start_age2 < 0.0:
        start_age2 = 200.0
    if end_age2 < 0.0:
        end_age2 = 150.0
    if rift_mag2 < 0.0:
        rift_mag2 = 1.0
    if mantle_fac2 < 0.0:
        mantle_fac2 = 1.0
    if start_age3 < 0.0:
        start_age3 = 200.0
    if end_age3 < 0.0:
        end_age3 = 150.0                    
    if rift_mag3 < 0.0:
        rift_mag3 = 1.0                    
    if mantle_fac3 < 0.0:
        mantle_fac3 = 1.0
    return (
            start_age1, end_age1, rift_mag1, mantle_fac1,
            start_age2, end_age2, rift_mag2, mantle_fac2,
            start_age3, end_age3, rift_mag3, mantle_fac3
        )
    

@jit(nopython = True, cache=True)
def apply_delta_beta_limits(delta_1, Beta_1, delta_2, Beta_2, delta_3, Beta_3):
    # Limits need to be applied to avoid division by zero
    Beta_limit = 20
    delta_limit = 20
    if delta_1 > delta_limit:
        delta_1 = delta_limit
    if delta_2 > delta_limit:
        delta_2 = delta_limit
    if delta_3 > delta_limit:
        delta_3 = delta_limit
    if Beta_1 > Beta_limit:
        Beta_1 = Beta_limit
    if Beta_2 > Beta_limit:
        Beta_2 = Beta_limit
    if Beta_3 > Beta_limit:
        Beta_3 = Beta_limit
    return (delta_1, Beta_1, delta_2, Beta_2, delta_3, Beta_3)


@jit(nopython = True, cache=True)
def get_rift_parameters(
                        start_age1, end_age1, rift_mag1, mantle_fac1,
                        start_age2, end_age2, rift_mag2, mantle_fac2,
                        start_age3, end_age3, rift_mag3, mantle_fac3,
                        tc_initial, tc_final, nphases

):
    (
        start_age1, end_age1, rift_mag1, mantle_fac1,
        start_age2, end_age2, rift_mag2, mantle_fac2,
        start_age3, end_age3, rift_mag3, mantle_fac3
    ) = apply_rift_param_limits(
                start_age1, end_age1, rift_mag1, mantle_fac1,
                start_age2, end_age2, rift_mag2, mantle_fac2,
                start_age3, end_age3, rift_mag3, mantle_fac3
            )
    (
        delta_1, delta_2, delta_3, Beta_1, 
        Beta_2, Beta_3, rift_time_1, rift_time_2, 
        rift_time_3, t_pr1, t_pr2, t_pr3, 
        tc_change, tc_change_1, tc_change_2, tc_change_3, 
        tc_final_1, tc_final_2, tc_final_3
    ) = lithospheric_stretching.calc_rift_params(
                        nphases, rift_mag1, rift_mag2, rift_mag3, 
                        mantle_fac1, mantle_fac2, mantle_fac3, 
                        tc_initial, tc_final, 
                        start_age1, end_age1, start_age2, 
                        end_age2, start_age3, end_age3
                    )
    (
        delta_1, Beta_1, delta_2, Beta_2, 
        delta_3, Beta_3
    ) = apply_delta_beta_limits(
                                delta_1, Beta_1, delta_2, Beta_2, 
                                delta_3, Beta_3
                            )
    return(
            start_age1, end_age1, rift_mag1, mantle_fac1,
            start_age2, end_age2, rift_mag2, mantle_fac2,
            start_age3, end_age3, rift_mag3, mantle_fac3,
            delta_1, delta_2, delta_3, 
            Beta_1, Beta_2, Beta_3, 
            rift_time_1, rift_time_2, rift_time_3, 
            t_pr1, t_pr2, t_pr3, tc_change, 
            tc_change_1, tc_change_2, tc_change_3, 
            tc_final_1, tc_final_2, tc_final_3
        )