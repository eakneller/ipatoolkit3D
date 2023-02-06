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
import map_tools
import analytical_sub_heatflow
import FDME_HeatConduction
import math_tools


@jit(nopython=True, cache=True)
def fill_interp_arrays(
                        nt1, ts_Myr1, hf_t1, ttsub_t1, 
                        q_thermal_vec1, q_crustal_vec1,
                        Te_vec1, ttsub_t1_xy, hf_t1_xy, 
                        q_thermal_t1_xy, q_crustal_t1_xy, 
                        Te_t1_xy
):
    for ii in range(nt1):
        ttsub_t1_xy[ii][0] = ts_Myr1[ii]
        ttsub_t1_xy[ii][1] = ttsub_t1[ii]
        hf_t1_xy[ii][0] = ts_Myr1[ii]
        hf_t1_xy[ii][1] = hf_t1[ii]
        q_thermal_t1_xy[ii][0] = ts_Myr1[ii]
        q_thermal_t1_xy[ii][1] = q_thermal_vec1[ii]
        q_crustal_t1_xy[ii][0] = ts_Myr1[ii]
        q_crustal_t1_xy[ii][1] = q_crustal_vec1[ii]
        Te_t1_xy[ii][0] = ts_Myr1[ii]
        Te_t1_xy[ii][1] = Te_vec1[ii]


@jit(nopython=True, cache=True)
def quick_interp(
                    hf_conv, nt1, nt2, start_age1, start_age2, 
                    end_age2, ttsub_t1_xy, hf_t1_xy,
                    q_thermal_t1_xy, q_crustal_t1_xy, Te_t1_xy, 
                    ttsub_t2_xy, hf_t2_xy, q_thermal_t2_xy, 
                    q_crustal_t2_xy, Te_t2_xy, ts_Myr1, hf_t1, 
                    ttsub_t1, q_thermal_vec1, q_crustal_vec1, Te_vec1
):
    # Get event 1 values at start of second event
    tMyr_o = start_age1 - start_age2
    ttsub_o = math_tools.linear_interp_v2(nt1, tMyr_o, ttsub_t1_xy)
    hf_o = hf_conv*math_tools.linear_interp_v2(nt1, tMyr_o, hf_t1_xy)
    q_thermal_o = math_tools.linear_interp_v2(nt1, tMyr_o, q_thermal_t1_xy)
    q_crustal_o = math_tools.linear_interp_v2(nt1, tMyr_o, q_crustal_t1_xy)
    Te_o = math_tools.linear_interp_v2(nt1, tMyr_o, Te_t1_xy)
    # Get values at end of syn-kinematic phase of second event
    tMyr_f = start_age1 - end_age2
    ttsub_f = math_tools.linear_interp_v2(nt2, tMyr_f, ttsub_t2_xy)
    hf_f = hf_conv*math_tools.linear_interp_v2(nt2, tMyr_f, hf_t2_xy)
    q_thermal_f = math_tools.linear_interp_v2(nt2, tMyr_f, q_thermal_t2_xy)
    q_crustal_f = math_tools.linear_interp_v2(nt2, tMyr_f, q_crustal_t2_xy)
    Te_f = math_tools.linear_interp_v2(nt2, tMyr_f, Te_t2_xy)
    # Interpolate to main array at each main array time
    for i, tMyr in enumerate(ts_Myr1):
        age = start_age1 - tMyr
        inrange = 0
        if age <= start_age2 and age >= end_age2:
            ddt = tMyr_f - tMyr_o
            ttsub = ttsub_o + (ttsub_f - ttsub_o)/ddt*(tMyr-tMyr_o)
            hf = hf_o + (hf_f - hf_o)/ddt*(tMyr-tMyr_o)
            q_thermal = (
                              q_thermal_o 
                            + (q_thermal_f - q_thermal_o)/ddt*(tMyr-tMyr_o)
                        )
            q_crustal = (
                              q_crustal_o 
                            + (q_crustal_f - q_crustal_o)/ddt*(tMyr-tMyr_o)
                        )
            Te = Te_o + (Te_f - Te_o)/ddt*(tMyr-tMyr_o)             
            inrange = 1
        elif age < end_age2:
            ttsub = math_tools.linear_interp_v2(nt2, tMyr, ttsub_t2_xy)
            hf = hf_conv*math_tools.linear_interp_v2(nt2, tMyr, hf_t2_xy)
            q_thermal = math_tools.linear_interp_v2(nt2, tMyr, q_thermal_t2_xy)
            q_crustal = math_tools.linear_interp_v2(nt2, tMyr, q_crustal_t2_xy)
            Te = math_tools.linear_interp_v2(nt2, tMyr, Te_t2_xy)
            inrange = 1
        if inrange == 1:
            hf_t1[i] = hf
            ttsub_t1[i] = ttsub
            q_thermal_vec1[i] = q_thermal
            q_crustal_vec1[i] = q_crustal
            Te_vec1[i] = Te


@jit(nopython=True, cache=True)
def quick_transfer(
                    nt2, start_age1, start_age2, ttsub_t2_xy, ttsub_t2,
                    hf_t2_xy, hf_t2, q_thermal_t2_xy, q_thermal_vec2,
                    q_crustal_t2_xy, q_crustal_vec2, Te_t2_xy, Te_vec2, 
                    ts_Myr2
):
    age_shift = start_age1 - start_age2
    for ii in range(nt2):
        ttsub_t2_xy[ii][0] = ts_Myr2[ii]+age_shift
        ttsub_t2_xy[ii][1] = ttsub_t2[ii]
        hf_t2_xy[ii][0] = ts_Myr2[ii]+age_shift
        hf_t2_xy[ii][1] = hf_t2[ii]
        q_thermal_t2_xy[ii][0] = ts_Myr2[ii]+age_shift
        q_thermal_t2_xy[ii][1] = q_thermal_vec2[ii]
        q_crustal_t2_xy[ii][0] = ts_Myr2[ii]+age_shift
        q_crustal_t2_xy[ii][1] = q_crustal_vec2[ii]
        Te_t2_xy[ii][0] = ts_Myr2[ii]+age_shift
        Te_t2_xy[ii][1] = Te_vec2[ii]


def analytical_rifting_multiple_events(
                    dt_Myr, delta_1, delta_2, delta_3, 
                    Beta_1, Beta_2, Beta_3, start_age1, 
                    start_age2, start_age3, end_age1, 
                    end_age2, end_age3, tc_initial, 
                    tm_initial, rho_sea, rho_crust, 
                    rho_mantle, k_bulk, cp_bulk, 
                    alpha_bulk, T_bottom, iup_in_air, 
                    itype_rho_a, Telastic, iuse_flexure, 
                    zs_np, Tn_np, hf_conv, nnodes, nphases
):    
    # Run the first event
    (
     ts_Myr1, hf_t1, ttsub_t1, 
     q_thermal_vec1, q_crustal_vec1, Te_vec1
     ) = analytical_rifting(
                             dt_Myr, delta_1, Beta_1, start_age1,
                             end_age1, tc_initial, tm_initial, 
                             rho_sea, rho_crust, rho_mantle,
                             k_bulk, cp_bulk, alpha_bulk, T_bottom, 
                             iup_in_air, itype_rho_a,
                             Telastic, iuse_flexure, zs_np, Tn_np, nnodes
                            ) 
    if nphases > 1:
        #**********************************************************************
        # Initialize interpolation arrays for firs event
        #**********************************************************************        
        nt1 = len(ts_Myr1)
        ttsub_t1_xy = np.zeros((nt1,2))
        hf_t1_xy = np.zeros((nt1,2))
        q_thermal_t1_xy = np.zeros((nt1,2))
        q_crustal_t1_xy = np.zeros((nt1,2))
        Te_t1_xy = np.zeros((nt1,2))
        
        fill_interp_arrays(
                            nt1, ts_Myr1, hf_t1, ttsub_t1, q_thermal_vec1, 
                            q_crustal_vec1, Te_vec1, ttsub_t1_xy, hf_t1_xy, 
                            q_thermal_t1_xy, q_crustal_t1_xy, Te_t1_xy
                        )
        #**********************************************************************
        # Run the second event
        #**********************************************************************
        # For the second event
        delta_2f = delta_1*delta_2 
        if delta_2f < 1:
            delta_2f = 1
        Beta_2f = delta_2f*Beta_2/delta_2
        
        (
            ts_Myr2, hf_t2, ttsub_t2, 
            q_thermal_vec2, 
            q_crustal_vec2, Te_vec2
        ) = analytical_rifting(
                            dt_Myr, delta_2f, Beta_2f, start_age2,
                            end_age2, tc_initial, tm_initial, 
                            rho_sea, rho_crust, rho_mantle,
                            k_bulk, cp_bulk, alpha_bulk, 
                            T_bottom, iup_in_air, itype_rho_a,
                            Telastic, iuse_flexure, zs_np, 
                            Tn_np, nnodes
                        )
        #**********************************************************************
        # Interpolate to master array and use linear transition between events
        #**********************************************************************
        nt2 = len(ts_Myr2)
        ttsub_t2_xy = np.zeros((nt2,2))
        hf_t2_xy = np.zeros((nt2,2))
        q_thermal_t2_xy = np.zeros((nt2,2))
        q_crustal_t2_xy = np.zeros((nt2,2))
        Te_t2_xy = np.zeros((nt2,2))
        quick_transfer(
                        nt2, start_age1, start_age2, ttsub_t2_xy, ttsub_t2,
                        hf_t2_xy, hf_t2, q_thermal_t2_xy, q_thermal_vec2,
                        q_crustal_t2_xy, q_crustal_vec2, Te_t2_xy, 
                        Te_vec2, ts_Myr2
                    )
        quick_interp(
                        hf_conv, nt1, nt2, start_age1, start_age2, end_age2,
                        ttsub_t1_xy, hf_t1_xy,
                        q_thermal_t1_xy, q_crustal_t1_xy, Te_t1_xy, 
                        ttsub_t2_xy, hf_t2_xy,
                        q_thermal_t2_xy, q_crustal_t2_xy, Te_t2_xy, 
                        ts_Myr1, hf_t1, ttsub_t1,
                        q_thermal_vec1, q_crustal_vec1, Te_vec1
                    )  
    if nphases == 3:
        #**********************************************************************
        # Run the third event
        #**********************************************************************    
        # For the third event
        delta_3f = delta_1*delta_2*delta_3 
        if delta_3f < 1:
            delta_3f = 1
        Beta_3f = delta_3f*Beta_3/delta_3
        # Run the third event
        (
            ts_Myr3, hf_t3, ttsub_t3, 
            q_thermal_vec3, q_crustal_vec3, 
            Te_vec3
        ) = analytical_rifting(
                                dt_Myr, delta_3f, Beta_3f, start_age3,
                                end_age3, tc_initial, tm_initial, 
                                rho_sea, rho_crust, rho_mantle,
                                k_bulk, cp_bulk, alpha_bulk, T_bottom, 
                                iup_in_air, itype_rho_a,
                                Telastic, iuse_flexure, zs_np, 
                                Tn_np, nnodes
                            )
        #**********************************************************************
        # Interpolate to master array and use linear transition between events
        #**********************************************************************
        nt3 = len(ts_Myr3)
        ttsub_t3_xy = np.zeros((nt3,2))
        hf_t3_xy = np.zeros((nt3,2))
        q_thermal_t3_xy = np.zeros((nt3,2))
        q_crustal_t3_xy = np.zeros((nt3,2))
        Te_t3_xy = np.zeros((nt3,2))
        quick_transfer(
                        nt3, start_age1, start_age3, 
                        ttsub_t3_xy, ttsub_t3,
                        hf_t3_xy, hf_t3, q_thermal_t3_xy, 
                        q_thermal_vec3,
                        q_crustal_t3_xy, q_crustal_vec3, 
                        Te_t3_xy, Te_vec3, ts_Myr3
                    )
        quick_interp(
                        hf_conv, nt2, nt3, start_age1, start_age3, 
                        end_age3, ttsub_t2_xy, hf_t2_xy,
                        q_thermal_t2_xy, q_crustal_t2_xy, 
                        Te_t2_xy, ttsub_t3_xy, hf_t3_xy,
                        q_thermal_t3_xy, q_crustal_t3_xy,
                        Te_t3_xy, ts_Myr1, hf_t1, ttsub_t1,
                        q_thermal_vec1, q_crustal_vec1, Te_vec1
                    )
    return ts_Myr1, hf_t1, ttsub_t1, q_thermal_vec1, q_crustal_vec1, Te_vec1


def Read_TTS_FW(
                event_dict, TTS_map_dict, input_path, 
                Lx, Ly, nx, ny, dx, dy, 
                xmin, xmax, ymin, ymax, irun_backward
):
    AOI_np = np.zeros((1,1))
    event_IDs = event_dict.keys()
    for event_ID in event_IDs:
        file_name = TTS_map_dict[event_ID]
        (
            ttsub_xy, nx, ny, dx, dy, 
            xmin, xmax, ymin, ymax, AOI_np_L
        ) = map_tools.read_ZMAP(input_path, file_name, AOI_np)
        if irun_backward == 0:
            event_dict[event_ID][8] = ttsub_xy[:]
        else:
            event_dict[event_ID][14] = ttsub_xy[:]


@jit(nopython=True, cache=True)
def calc_rift_params(
                        nphases, rift_mag1, rift_mag2, rift_mag3,
                        mantle_fac1, mantle_fac2, mantle_fac3, 
                        tc_initial, tc_final,
                        start_age1, end_age1, 
                        start_age2, end_age2, 
                        start_age3, end_age3
                    ):
    tc_change = tc_initial - tc_final
    # Initital variables
    delta_1 = 4.0
    Beta_1 = 4.0
    rift_time_1 = 10
    t_pr1 = 10.0
    tc_change_1 = 0.0
    tc_final_1 = 0.0
    delta_2 = 4.0
    Beta_2 = 4.0
    rift_time_2 = 10.0 
    t_pr2 = 10.0
    tc_change_2 = 0.0
    tc_final_2 = 0.0
    delta_3 = 4.0
    Beta_3 = 4.0
    rift_time_3 = 10.0 
    t_pr3 = 10
    tc_change_3 = 0.0
    tc_final_3 = 0.0
    if nphases == 1:
        rift_mag_tot = rift_mag1
        if rift_mag_tot == 0.0:
            rift_mag_tot = 1.0
            rift_mag1 = 1.0
        rift_mag1 = rift_mag1/rift_mag_tot
        tc_change_1 = tc_change*rift_mag1
        tc_final_1 = tc_initial - tc_change_1
        delta_1 = tc_initial/tc_final_1
        Beta_1 = delta_1*mantle_fac1
        if Beta_1 < 1:
            Beta_1 = 1
        rift_time_1 = start_age1-end_age1
        t_pr1 = end_age1
    elif nphases == 2:
        # normalize rift magnitudes
        rift_mag_tot = rift_mag1+rift_mag2
        if rift_mag_tot == 0.0:
            rift_mag_tot = 1.0
            rift_mag1 = 1.0
        rift_mag1 = rift_mag1/rift_mag_tot
        rift_mag2 = rift_mag2/rift_mag_tot 
        tc_change_1 = tc_change*rift_mag1
        tc_change_2 = tc_change*rift_mag2
        tc_final_1 = tc_initial - tc_change_1
        tc_final_2 = tc_final_1 - tc_change_2
        delta_1 = tc_initial/tc_final_1
        delta_2 = tc_final_1/tc_final_2
        Beta_1 = delta_1*mantle_fac1
        if Beta_1 < 1:
            Beta_1 = 1
        Beta_2 = delta_2*mantle_fac2
        if Beta_2 < 1:
            Beta_1 = 1
        rift_time_1 = start_age1-end_age1
        t_pr1 = end_age1-start_age2
        rift_time_2 = start_age2-end_age2
        t_pr2 = end_age2
    else:
        # normalize rift magnitudes
        rift_mag_tot = rift_mag1+rift_mag2+rift_mag3
        if rift_mag_tot == 0.0:
            rift_mag_tot = 1.0
            rift_mag1 = 1.0
        rift_mag1 = rift_mag1/rift_mag_tot
        rift_mag2 = rift_mag2/rift_mag_tot
        rift_mag3 = rift_mag3/rift_mag_tot
        tc_change_1 = tc_change*rift_mag1
        tc_change_2 = tc_change*rift_mag2
        tc_change_3 = tc_change*rift_mag3
        tc_final_1 = tc_initial - tc_change_1
        tc_final_2 = tc_final_1 - tc_change_2
        tc_final_3 = tc_final_2 - tc_change_3
        delta_1 = tc_initial/tc_final_1
        delta_2 = tc_final_1/tc_final_2
        delta_3 = tc_final_2/tc_final_3
        Beta_1 = delta_1*mantle_fac1
        if Beta_1 < 1:
            Beta_1 = 1
        Beta_2 = delta_2*mantle_fac2
        if Beta_2 < 1:
            Beta_2 = 2
        Beta_3 = delta_3*mantle_fac3
        if Beta_3 < 1:
            Beta_3 = 3
        rift_time_1 = start_age1-end_age1
        t_pr1 = end_age1-start_age2
        rift_time_2 = start_age2-end_age2
        t_pr2 = end_age2-start_age3
        rift_time_3 = start_age3-end_age3
        t_pr3 = end_age3
    
    return (
            delta_1, delta_2, delta_3, 
            Beta_1, Beta_2, Beta_3,
            rift_time_1, rift_time_2, rift_time_3, 
            t_pr1, t_pr2, t_pr3,
            tc_change, tc_change_1, tc_change_2, tc_change_3,
            tc_final_1, tc_final_2, tc_final_3
        )
            

@jit(nopython=True, cache=True)
def calculate_subsidence_v3(
                            nages, ts_Myr, tt_sub, hf_vec, event_age,
                            delta, beta, age_start, age_end, 
                            tc_initial, tm_initial,
                            rho_water, rho_crust, rho_mantle, 
                            k_bulk, cp_bulk, alpha_bulk, 
                            T_base, iup_in_air, itype_rho_a, 
                            Ts, zs, xc, yc, xm, ym, nnodes
):
    """
    output: 
        analytical subsidence and heat flow values at initital and some 
        given age. If the given age falls within the syn-rift, a linear 
        approximation is used.
    """
    L = tc_initial + tm_initial
    kappa = k_bulk/cp_bulk/rho_mantle
    hf_ini = k_bulk*T_base/L*1000.0
    # Define time relative to end age of rifting
    tMyr = age_end - event_age
    # A time array is defined because the functions below calcualte values for
    # multiple timesteps. Time is defined for the age of interest and at t = 0.
    ts_Myr[0] = 0.0
    ts_Myr[1] = tMyr
    sec_yr = 365.0*24.0*60.0*60.0
    tt_sub_ini = analytical_sub_heatflow.Analytical_Ini_Sub(
                                                        L, tc_initial, 
                                                        rho_crust, rho_mantle, 
                                                        rho_water, alpha_bulk,
                                                        T_base, delta, beta, 
                                                        iup_in_air, itype_rho_a
                                                    )
    tt_sub_final = analytical_sub_heatflow.Analytical_Final_Sub_v2(
                                    L, tc_initial, rho_crust, rho_mantle, 
                                    rho_water, alpha_bulk,
                                    T_base, delta, beta, tc_initial, Ts, 
                                    zs, xc, yc, xm, ym, nnodes, itype_rho_a
                                )
    if tMyr >= 0.0:
        analytical_sub_heatflow.Analytical_Sub_v2(
                                    nages, ts_Myr, tt_sub, delta, beta, L, 
                                    kappa, k_bulk, T_base,
                                    rho_mantle, rho_crust, rho_water, 
                                    alpha_bulk, sec_yr,
                                    tc_initial, iup_in_air, itype_rho_a
                                )
        tt_sub_v = tt_sub[1]
        analytical_sub_heatflow.Analytical_HeatFlow_v2(
                                    ts_Myr, hf_vec, delta, beta, L, kappa, 
                                    k_bulk, T_base, sec_yr, tc_initial
                                )
        hf = hf_vec[1]*-1.0
    else:
        ts_Myr[0]=0.0
        ts_Myr[1]=0.0
        analytical_sub_heatflow.Analytical_HeatFlow_v2(
                                    ts_Myr, hf_vec, delta, beta, L, kappa, 
                                    k_bulk, T_base, sec_yr, tc_initial
                                )
        tMyr = age_start - event_age
        # If we are in the syn-rift phase then use linear approximation        
        if tMyr >= 0.0:
            hf_peak = hf_vec[0]*-1.0
            tt_sub_v = tt_sub_ini/(age_start-age_end)*tMyr
            hf = hf_ini + (hf_peak-hf_ini)/(age_start-age_end)*tMyr
        else:
            tt_sub_v = 0.0
            hf = hf_ini
    return tt_sub_v, tt_sub_ini, tt_sub_final, hf, hf_ini


@jit(nopython=True, cache=True)
def calculate_hf_sub_analytical(
                                ts_Myr, tt_sub, hf_vec, event_age,
                                delta, beta, age_start, age_end, 
                                tc_initial, tm_initial,
                                rho_water, rho_crust, rho_mantle, 
                                k_bulk, cp_bulk, alpha_bulk,
                                T_base, iup_in_air, itype_rho_a, zs_np, 
                                Tn_np, nnodes, Telastic, iuse_flexure
):
    L = tc_initial+tm_initial
    kappa = k_bulk/cp_bulk/rho_mantle
    hf_ini = k_bulk*T_base/L*1000.0 
    TOLage = 0.25
    dage = abs(age_start - age_end)
    if age_start < age_end or dage < TOLage:
         age_start = age_end
    # Define time relative to end age of rifting
    tMyr = age_end - event_age
    if abs(tMyr) < TOLage:
        tMyr = 0.0
    # A time array is defined because the functions below calcualtes values for
    # multiple timesteps. Time is defined for the age of interest and at t = 0.
    ts_Myr[0] = 0.0
    ts_Myr[1] = tMyr
    sec_yr = 365.0*24.0*60.0*60.0
    if iuse_flexure == 1:
        g = 9.8
        (
            q_thermal, q_uc, Te
         ) = analytical_sub_heatflow.Analytical_Temp_v2(
                             nnodes, alpha_bulk, kappa, rho_crust, 
                             rho_mantle, rho_water, Telastic,
                             g, zs_np, tMyr, delta, beta, L, T_base, 
                             tc_initial, Tn_np
                            )
        Te_ini = (tc_initial + tm_initial)/T_base*Telastic
        if tMyr < 0.0:
            Te_peak = Te
            q_uc_peak = q_uc
            q_thermal_peak = q_thermal
        else:
            Te_peak = Te_ini
            q_uc_peak = 0.0
            q_thermal_peak = 0.0
    else:
        q_thermal = 0.0
        q_uc = 0.0
        Te = 0.0
        Te_ini = 0.0
    tt_sub_ini = analytical_sub_heatflow.Analytical_Ini_Sub(
                                L, tc_initial, rho_crust, rho_mantle, 
                                rho_water, alpha_bulk, T_base, delta, 
                                beta, iup_in_air, itype_rho_a
                            )
    if tMyr >= 0.0:
        nages = 1 # obsolete
        analytical_sub_heatflow.Analytical_Sub_v2(
                                            nages, ts_Myr, tt_sub, delta, beta, 
                                            L, kappa, k_bulk, T_base,
                                            rho_mantle, rho_crust, rho_water, 
                                            alpha_bulk, sec_yr, tc_initial, 
                                            iup_in_air, itype_rho_a
                                        )
        tt_sub_v = tt_sub[1]
        analytical_sub_heatflow.Analytical_HeatFlow_v2(
                                        ts_Myr, hf_vec, delta, beta, L, 
                                        kappa, k_bulk, T_base, sec_yr, 
                                        tc_initial
                                    )
        hf = hf_vec[1]*-1.0
    else:
        ts_Myr[0]=0.0
        ts_Myr[1]=0.0
        analytical_sub_heatflow.Analytical_HeatFlow_v2(
                                            ts_Myr, hf_vec, delta, beta, L, 
                                            kappa, k_bulk, T_base, sec_yr, 
                                            tc_initial
                                        )
        tMyr = age_start - event_age
        # If we are in the syn-rift phase then use linear approximation        
        if tMyr >= 0.0:
            hf_peak = hf_vec[0]*-1.0
            delta_age = abs(age_start - age_end)
            if abs(delta_age) > 0:
                tt_sub_v = tt_sub_ini/(age_start - age_end)*tMyr
            else:
                 tt_sub_v = tt_sub_ini # this is instantaneous subsidence
            if abs(delta_age) > 0:
                hf = hf_ini + (hf_peak - hf_ini)/(age_start - age_end)*tMyr
            else:
                hf = hf_peak
            if abs(delta_age) > 0:
                Te = Te_ini + (Te_peak - Te_ini)/(age_start - age_end)*tMyr
                q_uc = 0.0 + (q_uc_peak)/(age_start - age_end)*tMyr
                q_thermal = (q_thermal_peak)/(age_start - age_end)*tMyr
            else:
                Te = Te_peak
                q_uc = q_uc_peak
                q_thermal = q_thermal_peak
        else:
            tt_sub_v = 0.0
            hf = hf_ini
    return tt_sub_v, tt_sub_ini, hf, hf_ini, q_uc, q_thermal, Te


@jit(nopython=True, cache=True)
def analytical_rifting_loop(
                            delta, beta, start_age, end_age, nages, event_ages,
                            tc_initial, tm_initial, rho_water, rho_crust, 
                            rho_mantle, k_bulk, cp_bulk, alpha_bulk, 
                            T_bottom, iup_in_air, itype_rho_a,
                            ts_Myr_tmp, tt_sub_tmp, hf_vec_tmp, shf_t_L, 
                            sub_t_L, q_thermal_vec_tmp,
                            q_uc_vec_tmp, Te_vec_tmp, zs_np, Tn_np, nnodes, 
                            Telastic, iuse_flexure
):
    for i in range(nages):
        event_age = event_ages[i]
        (
            tt_sub_v, tt_sub_ini, 
            hf_v, hf_ini, q_uc,
            q_thermal, Te
        ) = calculate_hf_sub_analytical(
                                        ts_Myr_tmp, tt_sub_tmp, hf_vec_tmp,
                                        event_age, delta, beta, start_age, 
                                        end_age, tc_initial, tm_initial,
                                        rho_water, rho_crust, rho_mantle, 
                                        k_bulk, cp_bulk, alpha_bulk,
                                        T_bottom, iup_in_air, itype_rho_a, 
                                        zs_np, Tn_np, nnodes, Telastic,
                                        iuse_flexure
                                    )
        shf_t_L[i] = hf_v
        sub_t_L[i] = tt_sub_v
        q_thermal_vec_tmp[i]=q_thermal
        q_uc_vec_tmp[i]=q_uc
        Te_vec_tmp[i]=Te


@jit(nopython=True, cache=True)
def analytical_rifting(
                        dt, delta, beta, start_age, end_age, 
                        tc_initial, tm_initial, rho_water, 
                        rho_crust, rho_mantle, k_bulk, 
                        cp_bulk, alpha_bulk,
                        T_bottom, iup_in_air, itype_rho_a, 
                        Telastic, iuse_flexure, zs_np, 
                        Tn_np, nnodes
):
    nages = int(start_age/dt) + 1
    event_ages = np.zeros((nages))
    ts_Myr_L = np.zeros((nages))
    for i in range(nages):        
        if i < nages-1:
            age = start_age - float(i)*dt
        else: # Force last age to be present day
            age = 0.0            
        event_ages[i] = age        
        time_Myr = start_age - age
        ts_Myr_L[i] = time_Myr        
    shf_t_L = np.zeros((nages))
    sub_t_L = np.zeros((nages))
    q_thermal_vec_tmp = np.zeros((nages))
    q_uc_vec_tmp = np.zeros((nages))
    Te_vec_tmp = np.zeros((nages))
    nages_tmp = 2
    ts_Myr_tmp = np.zeros((nages_tmp))
    tt_sub_tmp = np.zeros((nages_tmp))
    hf_vec_tmp = np.zeros((nages_tmp))
    analytical_rifting_loop(
                            delta, beta, start_age, end_age, 
                            nages, event_ages,
                            tc_initial, tm_initial, rho_water, 
                            rho_crust, rho_mantle, k_bulk, cp_bulk, 
                            alpha_bulk,T_bottom, iup_in_air, 
                            itype_rho_a, ts_Myr_tmp, tt_sub_tmp,
                            hf_vec_tmp, shf_t_L, sub_t_L, 
                            q_thermal_vec_tmp, q_uc_vec_tmp, Te_vec_tmp, 
                            zs_np, Tn_np, nnodes, Telastic, iuse_flexure
                        )
    shf_t = np.zeros((nages))
    sub_t = np.zeros((nages))
    ts_Myr = np.zeros((nages))
    q_thermal_vec = np.zeros((nages))
    q_uc_vec = np.zeros((nages))
    Te_vec = np.zeros((nages))
    for i in range(nages):
        shf_t[i] = shf_t_L[i]
        sub_t[i] = sub_t_L[i]
        ts_Myr[i] = ts_Myr_L[i]
        q_thermal_vec[i] = q_thermal_vec_tmp[i]
        q_uc_vec[i] = q_uc_vec_tmp[i]
        Te_vec[i] = Te_vec_tmp[i]
    return ts_Myr, shf_t, sub_t, q_thermal_vec, q_uc_vec, Te_vec


def FD_RUNIT():
    sec_yr = 365.0*24.0*60.0*60.0
    L = 125000.0
    dz = 1000.0
    ##2.5      ### k_crust : Thermal conductivity of crust (W/m/K)            
    ##3.14      ### k_mantle : Thermal conductivity of mantle (W/m/K)
    ##1046.0   ### cp_crust : Specific heat of crust (J/kg/K)                 
    ##1129.68   ### cp_mantle : Specific heat of mantle (J/kg/K)                
    ##3.3e-5  ### alpha_crust : Isobaric thermal expansivity of crust (K^-1)   
    ##3.3e-5  ### alpha_mantle : Isobaric thermal expansivity of mantle (K^-1) 
    ##1.0e-11  ### B_crust : Isothermal compressibility of crust (Pa^-1)        
    ##1.0e-11  ### B_mantle : Isothermal compressibility of mantle (Pa^-1)   
    ##2800.0   ### rho_crust : 0 degree C density of the crust (kg/m^3)     
    ##3300.0   ### rho_mantle : 0 degree C density of the mantle (kg/m^3)    
    ##1040.0   ### rho_sea : Density of sea water (kg/m^3)                 
    ##100000.0 ### Llith : Depth of model & initial thickness of litho. (m)    
    ##30000.0  ### z_moho : Initial crustal thickness (m)                     
    ##1000.0   ### dz_lith: Finite difference node spacing for lithosphere (m) 
    ##7.5      ### gamma_MPa_K : Clapeyron slope for solidus (MPa/C)           
    ##200.0   ### Tmelt_o: Zero pressure melting temperature for solidus (C) 
    z1 = 10000.0 # Moho
    z2 = 100000.0 # Mantle tracking depth
    rho_crust = 2800.0
    rho_mantle = 3330.0
    cp = 1200.0
    alpha = 3.3e-5
    Q_crust = 0.0
    Q_mantle = 0.0
    k_mantle = 3.0
    k_crust = 3.0
    Tini = 1330.0
    Ts_ini = []
    nnodes = int(L/dz)+1
    zs = []
    rho_zs = []
    cp_zs = []
    alpha_zs = []
    Q_zs = []
    k_zs = []
    for i in range(nnodes):
        z = dz*float(i)/1000.0
        zs.append(z)
        if z <= z1:
            rho = rho_crust
            Q = Q_crust
            k = k_crust
        else:
            rho = rho_mantle
            Q = Q_mantle
            k = k_mantle
        rho_zs.append(rho)
        cp_zs.append(cp)
        alpha_zs.append(alpha)
        Q_zs.append(Q)
        k_zs.append(k)
        Ts_ini.append(Tini)
    T_t = [[0.0, Ts_ini]]
    bc_stype = "TEMP"
    HP_stype = "NA"
    T_top = 0.0
    T_bottom = 1330.0
    q_bottom = -50e3
    shf_t = []
    ts_Myr = []
    avg_Tc_t = []
    avg_Tm_t = []
    z_moho_t = []
    z_mantle_t = []
    irho_var = 0
    istart = 0
    DT_FAC_PRIMARY = 4.0
    kappa = k_mantle/rho_crust/cp
    dt_limit =(dz*dz)/kappa/DT_FAC_PRIMARY
    itype = 0
    if itype == 0:
        dt_Myr = 1.0
        dt = dt_Myr*sec_yr*1e6
    else:
        dt = dt_limit
    t_start_Ma = 0.0
    t_end_Myr = 300.0
    t_end = t_end_Myr*1e6*sec_yr # seconds
    dt_out = 50.0 # Myr
    print ("Solution type = ", itype, ": 0 = IMP, 1 = EXP")
    if itype == 0:
        tsolve1 = time.time()
        # This needs to be updated!
        FDME_HeatConduction.FD_conduction_kvar_IMP(
                                            istart, t_start_Ma, t_end, dt, 
                                            dz, zs, k_zs, cp_zs,
                                            alpha_zs, rho_zs, Q_zs, bc_stype, 
                                            HP_stype, T_top, T_bottom, 
                                            q_bottom, z1, z2,
                                            L,sec_yr, T_t, shf_t, ts_Myr, 
                                            avg_Tc_t, avg_Tm_t, z_moho_t, 
                                            z_mantle_t,irho_var, dt_out
                                        )
        tsolve2 = time.time()
        print (" FD_conduction_kvar_IMP cpu(s) : ", tsolve2-tsolve1)
    nt = len(T_t)
    for j in range(nt):
        if itype == 0:
            sflag = "IMP"
        else:
            sflag = "EXP"
        file_name = "TEMP_itype_"+sflag+"_nt_"+str(j)+".csv"
        fout = open(file_name, 'w')
        str_out = "dt_Myr,"+str(dt_Myr)+"\n"
        fout.write(str_out)
        str_out = "dz_km,"+str(dz)+"\n"
        fout.write(str_out)
        str_out = "t_Ma,"+str(T_t[j][0])+"\n"
        fout.write(str_out)
        
        T = T_t[j][1]
        for i, z, in enumerate(zs):
            str_out = str(z)+","+str(T[i])+"\n"
            fout.write(str_out)
        fout.close()