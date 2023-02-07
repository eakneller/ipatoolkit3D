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
import lithospheric_stretching
import finite_rifting
import print_funcs
import manage_parallel


def stretching_inversion(ioutput_main, process, model):
    nx = model.nx
    ny = model.ny
    inv_itype = model.inv_itype
    
    xth_xy = np.zeros((nx, ny))
    moho_xy_np = np.zeros((nx, ny))
    moho_twt_xy_np = np.zeros((nx, ny))
    crustal_thick_xy = np.zeros((nx, ny))
    delta_best_fit_xy = np.zeros((nx, ny))
    if inv_itype >= 0:
        iup_in_air = 1
        tt1 = time.time()
        ioutput = 1
        (
            delta_best_fit_xy, 
            crustal_thick_xy
        ) = match_present_day_thermo_tect(ioutput, iup_in_air, model)
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Calculated best-fit deltas", tt2-tt1)        
        xth_xy = np.copy(crustal_thick_xy)
        
        tt1 = time.time()
        moho_xy_np = make_moho_TVD(xth_xy, model)
        moho_twt_xy_np = make_moho_TWT(xth_xy, model)
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Calculated moho maps in TWT and TVD", tt2-tt1) 
    return (
            xth_xy, moho_xy_np, moho_twt_xy_np,
            delta_best_fit_xy
           )


def make_moho_TWT(xth_xy, model):
    if model.inv_itype >= 0:
        vcrust = 6800.0 # m/s
        moho_twt_xy_np = np.zeros((model.nx, model.ny))
        itopID = 0    
        twt_bsmt_xy = model.tops_list_bs[itopID][18]
        for i in range(model.nx):
            for j in range(model.ny):
                moho_twt_xy_np[i][j] = (
                                      twt_bsmt_xy[i][j] 
                                    + xth_xy[i][j]/vcrust*2.0*1000
                                )
        return moho_twt_xy_np


def make_moho_TVD(xth_xy, model):
    if model.inv_itype >= 0:
        xth_xy_np = np.asarray(xth_xy)
        moho_xy_np = np.zeros((model.nx, model.ny))    
        keys = list(model.event_dict_bs.keys())
        nevents = len(keys)
        event_ID_last = keys[nevents-1]    
        itopID = 0
        event_index = model.tops_list_bs[itopID][14][event_ID_last]    
        depth_bsmt_xy = model.tops_list_bs[itopID][1][event_index]    
        depth_bsmt_xy_np = np.asarray(depth_bsmt_xy)    
        for i in range(model.nx):        
            for j in range(model.ny):            
                moho_xy_np[i][j] = depth_bsmt_xy_np[i][j] + xth_xy_np[i][j]    
        return moho_xy_np
    

@jit(nopython=True)
def do_bisection_FiniteRifting(
                                nx, ny, ttsub_pd_xy_np, age_s_xy_np, 
                                age_e_xy_np, mantle_fac1_xy_np, 
                                tc_initial, tm_initial,
                                rho_water, rho_crust, rho_mantle, k_bulk, 
                                cp_bulk, alpha_bulk, dt_Myr, dt_rift_Myr, 
                                dz_lith, HP_itype, Q_crust, Q_mantle, Ao, ar,
                                rift_itype, T_base, iup_in_air, itype_rho_a,
                                delta_best_fit_xy_np, crustal_thick_xy_np, TOL,
                                L_crust_ref, L_lith_ref, rho_crust_ref, 
                                rho_mantle_ref, AOI_np, itype3D, inode, jnode
):
    # Force single phase
    nphases = 1
    # Force zero surface temperature
    T_top = 0.0
     # Force constant kappa
    icon_kappa = 1
    dt_out = dt_Myr*2.0
    # Dummy variables
    irho_var = 0
    delta_2 = 4.0
    Beta_2 = 4.0
    rift_time_2 = 100.0
    t_pr2 = 100.0
    delta_3 = 4.0
    Beta_3 = 4.0
    rift_time_3 = 10.0
    t_pr3 = 10.0
    Llith = tc_initial + tm_initial
    icount_max = 100
    for i in numba.prange(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if itype3D == 0:
                if i == inode and j == jnode:
                    AOI_flag = AOI_flag
                else:
                    AOI_flag = 0
            if AOI_flag == 1:
                tt_sub_obs = ttsub_pd_xy_np[i][j]
                age_start = age_s_xy_np[i][j]
                age_end = age_e_xy_np[i][j]
                mantle_fac1 = mantle_fac1_xy_np[i][j]
                # shifts the end age of rifting to older age (within syn-rift) 
                # to account for the effects of finite rifting not including in 
                # analytical model
                check = 1e32
                icount = 0
                delta_L = 1
                delta_R = 100.0
                rift_time_1 = age_start - age_end
                t_pr1 = age_end
                # Inverted parameters for finite_and_instan_rifting()
                delta_1 = delta_R
                # Allow depth depthdent stretching
                Beta_1 = delta_1*mantle_fac1
                if Beta_1 < 1:
                    Beta_1 = 1
                Telastic = 0
                (
                    number_of_timesteps,
                    ts_Myr, hf_t, 
                    ttsub_t, Te_t, 
                    q_uc_t, q_thermal_t
                ) = finite_rifting.finite_rifting(
                            dt_Myr, dt_rift_Myr, dt_out, 
                            k_bulk, k_bulk, cp_bulk, cp_bulk, 
                            alpha_bulk, alpha_bulk, 
                            rho_crust, rho_mantle, rho_water, 
                            Llith, tc_initial, 
                            dz_lith, T_top, T_base, 
                            HP_itype, Q_crust, Q_mantle, Ao, ar, 
                            rift_itype, nphases, delta_1, Beta_1, 
                            rift_time_1, t_pr1, delta_2, Beta_2, 
                            rift_time_2, t_pr2, delta_3, Beta_3, 
                            rift_time_3, t_pr3, icon_kappa, 
                            irho_var, iup_in_air, itype_rho_a, 
                            L_crust_ref, L_lith_ref, 
                            rho_crust_ref, rho_mantle_ref, Telastic
                        )
                tt_sub_R = ttsub_t[number_of_timesteps - 1]
                do_bisect = 1 
                if tt_sub_obs > tt_sub_R:
                    do_bisect = 0
                    delta_n = 100
                elif tt_sub_obs < 0.0:
                    do_bisect = 0
                    delta_n = 1
                if do_bisect == 1:
                    while check > TOL:
                        delta_n = (delta_L+delta_R)/2.0  
                        if check != 0.0:
                            # Inverted parameters
                            delta_1 = delta_n
                            # Allow depth depthdent stretching
                            Beta_1 = delta_1*mantle_fac1
                            if Beta_1 < 1:
                                Beta_1 = 1
                            Telastic = 0
                            #iuse_flexure = 0
                            (
                                number_of_timesteps,
                                ts_Myr, hf_t, 
                                ttsub_t, Te_t, 
                                q_uc_t, q_thermal_t
                            ) = finite_rifting.finite_rifting(
                                        dt_Myr, dt_rift_Myr, dt_out, 
                                        k_bulk, k_bulk, cp_bulk, cp_bulk, 
                                        alpha_bulk, alpha_bulk, 
                                        rho_crust, rho_mantle, rho_water, 
                                        Llith, tc_initial, 
                                        dz_lith, T_top, T_base, 
                                        HP_itype, Q_crust, Q_mantle, Ao, ar, 
                                        rift_itype, nphases, delta_1, Beta_1, 
                                        rift_time_1, t_pr1, delta_2, Beta_2, 
                                        rift_time_2, t_pr2, delta_3, Beta_3, 
                                        rift_time_3, t_pr3, icon_kappa, 
                                        irho_var, iup_in_air, itype_rho_a, 
                                        L_crust_ref, L_lith_ref, 
                                        rho_crust_ref, rho_mantle_ref, Telastic
                                    )
                            tt_sub_n = ttsub_t[number_of_timesteps - 1]
                        check = abs(tt_sub_n - tt_sub_obs)
                        if tt_sub_n < tt_sub_obs:
                            delta_L = delta_n
                        elif tt_sub_n > tt_sub_obs:
                            delta_R = delta_n
                        icount = icount + 1
                        if icount > icount_max:
                            check = 0.0
                else: 
                    delta_n = delta_R
                delta_best_fit_xy_np[i][j] = delta_n
                crustal_thick_xy_np[i][j] = tc_initial/delta_n
            else:
                delta_best_fit_xy_np[i][j] = -99999.0
                crustal_thick_xy_np[i][j] = -99999.0


@jit(nopython=True)
def do_bisection(
                    nx, ny, ttsub_pd_xy_np, age_s_xy_np, age_e_xy_np, 
                    shift_fac, event_age, 
                    tc_initial, tm_initial, rho_water, rho_crust, 
                    rho_mantle, k_bulk, cp_bulk, alpha_bulk, T_base, 
                    iup_in_air, itype_rho_a, delta_best_fit_xy_np, 
                    crustal_thick_xy_np, 
                    TOL, AOI_np, itype3D, inode, jnode, 
                    mantle_fac1_xy_np
):
    icount_max = 100    
    ncount = 100
    icount_nodes = 0
    total_nodes_comp = 0
    nnodes = 100000
    nages = 2
    for i in numba.prange(nx):
        for j in range(ny):
            if icount_nodes == ncount:
                total_nodes_comp = total_nodes_comp + ncount
                icount_nodes = 0
            icount_nodes = icount_nodes + 1            
            AOI_flag = AOI_np[i][j]            
            if itype3D == 0:
                if i == inode and j == jnode:
                    AOI_flag = AOI_flag
                else:
                    AOI_flag = 0                    
            if AOI_flag == 1:
                zs = np.zeros((nnodes))
                xc = np.zeros((nnodes))
                yc = np.zeros((nnodes))
                xm = np.zeros((nnodes))
                ym = np.zeros((nnodes))
                Ts = np.zeros((nnodes))
                ts_Myr = np.zeros((nages))
                tt_sub = np.zeros((nages))
                hf_vec = np.zeros((nages))  
                tt_sub_obs = ttsub_pd_xy_np[i][j]                
                age_start = age_s_xy_np[i][j]
                age_end = age_e_xy_np[i][j]
                mfac = mantle_fac1_xy_np[i][j]                
                # shifts the end age of rifting to older age (within syn-rift) 
                # to account for the effects of finite rifting not including in 
                # analytical model
                age_end = age_start - (age_start - age_end)*shift_fac
                check = 1e32
                icount = 0                
                delta_L = 1
                delta_R = 100.0                
                (
                    tt_sub_R, tt_sub_ini, 
                    tt_sub_final, hf, hf_ini
                 ) = lithospheric_stretching.calculate_subsidence_v3(
                                     nages, ts_Myr, tt_sub, hf_vec, event_age, 
                                     delta_R, delta_R*mfac, age_start,
                                     age_end, tc_initial, tm_initial, 
                                     rho_water, rho_crust, rho_mantle,
                                     k_bulk, cp_bulk, alpha_bulk, T_base, 
                                     iup_in_air, itype_rho_a,
                                     Ts, zs, xc, yc, xm, ym, nnodes
                                )
                do_bisect = 1
                if tt_sub_obs > tt_sub_R:
                    do_bisect = 0
                    delta_n = 100
                elif tt_sub_obs < 0.0:
                    do_bisect = 0
                    delta_n = 1
                if do_bisect == 1:  
                    while check > TOL:
                        delta_n = (delta_L+delta_R)/2.0
                        if check != 0.0:
                            (
                                tt_sub_n, tt_sub_ini, 
                                tt_sub_final, hf, hf_ini
                           ) = lithospheric_stretching.calculate_subsidence_v3(
                                   nages, ts_Myr, tt_sub, hf_vec, event_age, 
                                   delta_n, delta_n*mfac,
                                   age_start, age_end, tc_initial, tm_initial,
                                   rho_water, rho_crust, rho_mantle, k_bulk,
                                   cp_bulk, alpha_bulk, T_base, iup_in_air,
                                   itype_rho_a, Ts, zs, xc, yc, xm, ym, nnodes
                                )
                        check = abs(tt_sub_n - tt_sub_obs)      
                        if tt_sub_n < tt_sub_obs:
                            # tt_sub_R rmains constant
                            # delta_R remains unchanged
                            #tt_sub_L = tt_sub_n
                            delta_L = delta_n
                        elif tt_sub_n > tt_sub_obs:
                            #tt_sub_R = tt_sub_n
                            delta_R = delta_n
                            # tt_sub_L remains constant
                            # delta_L remains unchanged
                        icount = icount + 1
                        if icount > icount_max:
                            check = 0.0
                else:
                    delta_n = delta_R                
                delta_best_fit_xy_np[i][j] = delta_n
                crustal_thick_xy_np[i][j] = tc_initial/delta_n
            else:
                delta_best_fit_xy_np[i][j] = -99999.0
                crustal_thick_xy_np[i][j] = -99999.0


def match_present_day_thermo_tect(ioutput, iup_in_air, model):
    # Unpack model object for jitted functions
    #event_dict_bs = model.event_dict_bs
    age_s_xy = model.age_s_xy
    age_e_xy = model.age_e_xy
    tc_initial = model.tc_initial
    tm_initial = model.tm_initial
    rho_water = model.rho_water
    rho_crust = model.rho_crust
    rho_mantle = model.rho_mantle
    k_bulk = model.k_bulk
    cp_bulk = model.cp_bulk
    alpha_bulk = model.alpha_bulk
    T_base = model.T_base
    nx = model.nx
    ny = model.ny
    AOI_np = model.AOI_np
    shift_fac = model.shift_fac
    itype_rho_a = model.itype_rho_a
    TOL_delta_bisec = model.TOL_delta_bisec
    dt_Myr = model.dt_Myr
    dt_rift_Myr = model.dt_rift_Myr
    dz_lith = model.dz_lith
    HP_itype = model.HP_itype
    Q_crust = model.Q_crust
    Q_mantle = model.Q_mantle
    Ao = model.Ao
    ar = model.ar
    rift_itype = model.rift_itype
    mantle_fac1_xy = model.mantle_fac1_xy
    inv_itype = model.inv_itype
    L_crust_ref = model.L_crust_ref
    L_lith_ref = model.L_lith_ref
    rho_crust_ref = model.rho_crust_ref 
    rho_mantle_ref = model.rho_mantle_ref
    itype3D = model.itype3D
    inode = model.inode
    jnode = model.jnode                              
    keys = list(model.event_dict_bs.keys())
    nevents_bs = len(keys)
    event_ID_list_bs = keys[:]
    event_ID_last_bs = event_ID_list_bs[nevents_bs-1]
    event_age = model.event_dict_bs[event_ID_last_bs][0]
    ttsub_pd_xy = model.event_dict_bs[event_ID_last_bs][8]      
    ttsub_pd_xy_np = np.copy(ttsub_pd_xy)     
    age_s_xy_np = np.copy(age_s_xy)
    age_e_xy_np = np.copy(age_e_xy)
    mantle_fac1_xy_np = np.copy(mantle_fac1_xy)
    delta_best_fit_xy_np = np.zeros((nx,ny))
    crustal_thick_xy_np = np.zeros((nx,ny))
    if inv_itype == 0:
        do_bisection(
                        nx, ny, ttsub_pd_xy_np, age_s_xy_np, age_e_xy_np,
                        shift_fac, event_age, 
                        tc_initial, tm_initial, rho_water, rho_crust, 
                        rho_mantle, k_bulk, cp_bulk, alpha_bulk, T_base, 
                        iup_in_air, itype_rho_a, delta_best_fit_xy_np, 
                        crustal_thick_xy_np, 
                        TOL_delta_bisec, AOI_np, itype3D, inode, jnode, 
                        mantle_fac1_xy_np
                        )
    elif inv_itype == 1:
        #(
        #    do_bisection_FiniteRifting_active
        #) = manage_parallel.manage_parallel(
        #                                do_bisection_FiniteRifting, itype3D)
        do_bisection_FiniteRifting(
                                    nx, ny, ttsub_pd_xy_np, age_s_xy_np, 
                                    age_e_xy_np, mantle_fac1_xy_np, 
                                    tc_initial, tm_initial,
                                    rho_water, rho_crust, rho_mantle, k_bulk, 
                                    cp_bulk, alpha_bulk,
                                    dt_Myr, dt_rift_Myr, dz_lith, HP_itype, 
                                    Q_crust, Q_mantle, Ao, ar, rift_itype, 
                                    T_base, iup_in_air, itype_rho_a,
                                    delta_best_fit_xy_np, crustal_thick_xy_np, 
                                    TOL_delta_bisec, L_crust_ref, L_lith_ref, 
                                    rho_crust_ref, rho_mantle_ref, AOI_np,
                                    itype3D, inode, jnode
                                    )
    return delta_best_fit_xy_np, crustal_thick_xy_np

