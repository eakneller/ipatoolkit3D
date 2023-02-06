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
import math
import numpy as np
import numba
from numba import jit
import math_tools
import map_tools
import steady_state_thermal
import manage_parallel
import print_funcs


def thermal_history(model, process, ioutput_main):
    if model.icalc_temp == 1:
        tt1 = time.time()
        calculate_thermal_history_para(model, process, ioutput_main)            
        tt2 = time.time()
        print_funcs.print_finfo(
                    ioutput_main, process, 
                    "Calculated thermal history", tt2-tt1)
        tt1 = time.time()       
        iapply_thermal_smooth = 1
        if iapply_thermal_smooth == 1:
            # search radius in meters
            rad_search_therm = 2500.0
            apply_thermal_smoothing(model, rad_search_therm)            
        tt2 = time.time()
        print_funcs.print_finfo(
                                ioutput_main, process, 
                                "Applied thermal smoothing", tt2-tt1)    
    
    
    
def calc_SWIT(APWP_dict, surf_temp_xy, lat, lon, splate, age_event, PWD):
    # Define APWP data for this plate
    APWP_lons = APWP_dict[splate][0]
    APWP_lats = APWP_dict[splate][1]
    APWP_ages = APWP_dict[splate][2]
    # Interpolate to get virtual pole
    vlon = math_tools.linear_interp_python(age_event, APWP_ages, APWP_lons)
    vlat = math_tools.linear_interp_python(age_event, APWP_ages, APWP_lats)
    # get lons from 0 to 360
    if vlon < 0:
        vlon = 360 + vlon
    # Convert to radians
    vlon = vlon*math.pi/180.0
    vlat = vlat*math.pi/180.0
    lat = lat*math.pi/180.0
    lon = lon*math.pi/180.0
    # Paleo-lat (rad) using Haile (1975)
    plat = (
              math.cos(lat)*math.cos(vlat)*math.cos(vlon-lon)
            + math.sin(lat)*math.sin(vlat)
            )
    # Convert to degrees    
    plat = plat*180.0/math.pi
    if age_event == 0.0:
        plat = lat*180.0/math.pi
    # Determine average surface temperature and average arctic temperature
    i = int(498 - round(age_event,0))
    # This may need to be adjusted by 1 unit 
    # (given the uncertainty this is not critical)
    j = int(90 - round(plat,0))
    #print("i, j : ", i, j)
    T_surf = surf_temp_xy[i][j]
    T_arctic = surf_temp_xy[i][0]
    #print("T_surf, T_arctic, plat : ", T_surf, T_arctic, plat)
    # SWIT Model (Beardsmore and Cull, 2001, pg 74) 
    # and by Hantschel and Kauerauf (2010). 
    T_200m = T_surf - 3.0
    if PWD <= 0.1:
        T_swit = T_surf
    elif PWD <= 200.0:
        T_swit = T_surf + (T_200m-T_surf)/200.0*PWD
    elif PWD <= 600.0:
        T_swit = T_200m + (T_arctic - T_200m)/400.0*(PWD-200.0)
    else:
        T_swit = T_arctic+(-2.5-T_arctic)/7400.0*(PWD-600.0)
    return T_swit


@jit(nopython=True, cache=True)
def calc_temp(
                age_event, age_event_prev, 
                iuse_temp_dep_k, nrelax, nsublayers, nnodes, inode_max, 
                event_type_ID, kappa_lith, sec_per_myr, T_top, q_bottom, 
                tot_sed_thick_prev, tot_sed_thick, 
                zs, kg_zs, Qg_zs, phi_o_zs, Q_zs, c_zs, maxfb_zs, Tprev_zs, 
                Tini_zs, Lthick_prev, layer_thick, elem_thick, elem_tops, 
                elem_bottoms, layer_thick_pd, erothick, k_layer_zs, k_elem_zs, 
                kg_up_zs, dzi, Hxdzi, Qbi, T_base, Tc_ss, T_trans, k_water, 
                Q_water
):
    #*************************
    # Define initial condition
    #*************************
    for mm in range(nnodes):
        if event_type_ID == 0:
            # Deposition
            if mm > 0:
                T = Tprev_zs[mm-1]
                Tini_zs[mm] = T
            else:
                T = T_top
                Tini_zs[mm] = T                
        else:
            # Deposition and erosion
            # Note that the unconformity is tracked with an additional node
            if mm > 1:
                T = Tprev_zs[mm-1]
                Tini_zs[mm] = T
            else:
                #T = T_top
                T = (
                          Tprev_zs[0]
                        + (Tprev_zs[1]-Tprev_zs[0])/Lthick_prev[0]*erothick
                    )
                Tini_zs[mm] = T
    #**********************************
    # Define layer thicknesses (meters)
    #**********************************
    for mm in range(nnodes):
        if mm < inode_max:
            zc = zs[mm]
            zl = zs[mm+1]
            Lthick = (zl-zc)
        else:
            Lthick = 1.0   
        layer_thick[mm] = Lthick*1000.0
    #*************************************************
    # Define top centered element thicknesses (meters)
    #*************************************************
    for mm in range(nnodes):
        if mm == 0:
            zc = zs[mm]
            zl = zs[mm+1]
            ethick = (zl-zc)/2.0
            z_top = 0.0
            z_bottom = (zl+zc)/2
        elif mm > 0 and mm < inode_max:
            zu = zs[mm-1]
            zc = zs[mm]
            zl = zs[mm+1]
            ethick = (zc-zu)/2.0 + (zl-zc)/2.0
            z_top = (zc+zu)/2.0
            z_bottom = (zl+zc)/2
        elif mm == inode_max:
            zu = zs[mm-1]
            zc = zs[mm]
            ethick = (zc-zu)/2.0
            z_top = (zc+zu)/2.0
            z_bottom = zc
        elem_thick[mm] = ethick*1000.0
        elem_tops[mm] = z_top*1000.0
        elem_bottoms[mm] = z_bottom*1000.0
    #*********************************************
    # Calculate average radiogenic heat production 
    # using geometric average of sub-layers
    #*********************************************
    for mm in range(nnodes):
        phi_o = phi_o_zs[mm]
        c = c_zs[mm]
        #z = zs[mm]*1000.0
        maxfb = maxfb_zs[mm]
        Q_grain = Qg_zs[mm]        
        phi_node = phi_o*math.exp(-maxfb*c)
        Qnode = Q_grain*(1 - phi_node) + Q_water*(phi_node)        
        Q_zs[mm] = Qnode
    #**************************************************
    # Calculate average thermal conductivity for layers 
    # using harmonic average of sub-layers
    #**************************************************
    for irelax in range(nrelax):
        for mm in range(nnodes):
            # Get rock properties of layer                        
            k_grain = kg_zs[mm] # Grain k: W/m/K
            if irelax > 0:
                if iuse_temp_dep_k == 1:
                    Tprev = Tc_ss[mm]
                    k_grain_up = (
                                     358.0*(1.0227*k_grain - 1.882)*
                                                            (
                                                               1/(Tprev+273.15)
                                                             - 0.00068
                                                            )
                                    + 1.84
                                )
                else:
                    k_grain_up = k_grain
            else:
                k_grain_up = k_grain
            kg_up_zs[mm] = k_grain_up
            # surface Porosity % converted to fraction
            phi_o = phi_o_zs[mm]
            # Porosity decay depth 1/m
            c = c_zs[mm]
            # maximum forward burial depth in meters
            maxfb = maxfb_zs[mm]
            # layer thickness in meters
            Lthick = layer_thick[mm]
            if Lthick > 0.0:
                dthick = Lthick/nsublayers
                sum_ksub_inv = 0.0
                for nn in range(nsublayers):
                    # depth relative to maximum burial of top (meters)
                    z = maxfb + dthick*float(nn)
                    phi_sublayer = phi_o*math.exp(-z*c)                                        
                    # Geometric average
                    k_geom_avg = (
                                     k_grain_up**(1.0-phi_sublayer)
                                    *k_water**(phi_sublayer)
                                )                    
                    ksub_inv = 1.0 / k_geom_avg                    
                    sum_ksub_inv = sum_ksub_inv + ksub_inv                
                kavg = Lthick / dthick / sum_ksub_inv                
                k_layer_zs[mm] = kavg            
            else:                
                z = maxfb                                    
                phi_sublayer = phi_o*math.exp(-z*c)                                    
                k_geom_avg = (
                                 k_grain_up**(1.0-phi_sublayer)
                                *k_water**(phi_sublayer)
                            )                    
                kavg = k_geom_avg                
                k_layer_zs[mm] = kavg
        #****************************************************
        # Calculate average thermal conductivity for elements 
        # using harmonic averages
        #****************************************************
        for mm in range(nnodes):
            if mm < inode_max:
                # get top centered element thickness
                E = elem_thick[mm]
                # center of element (meters)
                zc = zs[mm]*1000.0
                # top of element (meters)
                z_top = elem_tops[mm]
                # base of element (meters)
                z_bottom = elem_bottoms[mm]
                d1 = zc-z_top
                d2 = z_bottom-zc
                # Get average layer conductivities from layers 
                # above and below top
                if mm > 0:
                    kL1 = k_layer_zs[mm-1]
                    kL2 = k_layer_zs[mm]
                    check = (d2*kL1 + d1*kL2)
                    if check > 0.0:
                        kavg = E*kL1*kL2 / (d2*kL1 + d1*kL2)
                    else:
                        kavg = kL1
                else:
                    kavg = k_layer_zs[mm]
            else:
                kavg = k_layer_zs[mm-1]
            k_elem_zs[mm] = kavg
        steady_state_thermal.steady_state_nlayer_v3(
                                                    0, nnodes, T_top,
                                                    q_bottom, zs, k_elem_zs, 
                                                    Q_zs, dzi, Hxdzi, Qbi, 
                                                    T_base, Tc_ss
                                                )           
    #*********************
    # Calculate transients
    #*********************
    # initial bottom temperature (C)
    Tbottom_ini = Tini_zs[inode_max]
    # submud bottom depth of current model (meters)
    z_bottom = zs[inode_max]*1000.0
    # current average steady-state gradient (C/meters)
    A = (np.amax(Tc_ss) - Tc_ss[0])/tot_sed_thick
    if A == 0.0:
        A = 20.0
    # Characteristic initial depth (meters)
    delta_z = (Tbottom_ini - T_top)/A
    # Effective deposition/erosion thickness for analytical transient equation
    Heff = z_bottom - delta_z
    # get the decompacted thickness of youngest layer (meters) 
    Hid =  layer_thick[0]
    if event_type_ID == 0:
        # Deposition
        if Hid > Heff:
            H = Hid
        else:
            H = Heff
    else:
        # Erosion and Deposition
        H = -Heff
    # Approximate transient
    for mm in range(nnodes):
        T_ss_actual = Tc_ss[mm]
        z = zs[mm]*1000.0
        if event_type_ID == 0:
            T_ana_ini = T_top + A * (z - H)
        else:
            T_ana_ini = T_top + A * (z + H)
        
        if T_ana_ini < T_top:
            T_ana_ini = T_top
        T_ana_ss = T_top + A*z
        dtime_Myr = age_event_prev - age_event
        dtime_sec = dtime_Myr*sec_per_myr
        if event_type_ID == 0:
            # Deposition (Wangen, 2010)
            arg = z/2.0/math.sqrt(kappa_lith*dtime_sec)
            T_ana_trans = T_top + A*(z-H) + A*H*(1.0 - math.erf(arg))
        else:
            # Erosion and deposition (Wangen, 2010)
            arg = z/2.0/math.sqrt(kappa_lith*dtime_sec)
            T_ana_trans = T_top + A*z + A*H*(math.erf(arg))        
        if (T_ss_actual + (T_ana_trans - T_ana_ss)) > 0: 
            # (T_ss_actual - (T_ana_ss - T_ana_trans)) > 0:            
            #T = T_ss_actual - (T_ana_ss - T_ana_trans)
            T = T_ss_actual + (T_ana_trans - T_ana_ss)
        else:
            T = T_top
        T_trans[mm] = T

def initialize_decompacted_1Dmesh(ntops):
    # depth of tops (kilometers)
    zs = np.zeros((ntops))
    # grain conductivity at top locations (W/m/K)
    kg_zs = np.zeros((ntops))
    # updated grain conductivity at top locations (W/m/K)
    kg_up_zs = np.zeros((ntops))
    # grain heat production at top location (W/m/m/m)
    Qg_zs = np.zeros((ntops))
    # surface porosity fraction at top location
    phi_o_zs = np.zeros((ntops))
    # porosity decay depth 1/m
    c_zs = np.zeros((ntops))
    # maximum forward burial depth at top 
    # locations (meters)
    maxfb_zs = np.zeros((ntops))
    # thickness of layers (present-day) in meters
    layer_thick_pd = np.zeros((ntops))
    # thickness of layers in meters
    layer_thick = np.zeros((ntops))
    # thickness of elements centered on layer tops  (meters)
    elem_thick = np.zeros((ntops))
    # depth of top centerd element tops (meters)
    elem_tops = np.zeros((ntops))
     # depth of top centerd element tops (meters)
    elem_bottoms = np.zeros((ntops))
    # depth of tops of previous event (kilometers)
    Lthick_prev = np.zeros((ntops))
    # Previous temperature in previous mesh configuration (C)
    Tprev_zs = np.zeros((ntops))
    # Initial temperature in current mesh configuration (C)
    Tini_zs = np.zeros((ntops))                        
    # harmonic average conductivity layer using 
    # sublayering (W/m/K)
    k_layer_zs = np.zeros((ntops))
    # harmonic average conductivity for elem centered 
    # at top location (W/m/K)
    k_elem_zs = np.zeros((ntops))
    # average heat production of layer centered at top 
    # location (W/m/m/m)
    Q_zs = np.zeros((ntops))
    # Initialize arrays used in steady state functions
    # depth of tops (meters) array used in 
    # steady state function
    dzi = np.zeros((ntops))
    # array of products use in steady state function
    Hxdzi = np.zeros((ntops))
     # array of products used in steady-state function
    Qbi = np.zeros((ntops))
    # Initialize solution arrays
    # Steady-state solution at the base of layers 
    # centered on tops (C)
    T_base = np.zeros((ntops))
    # steady-state solution interpolated to top 
    # locations (C)
    Tc_ss = np.zeros((ntops))
    # approximate transient (C)
    T_trans = np.zeros((ntops))
    
    return (
            zs, kg_zs, kg_up_zs, Qg_zs, phi_o_zs, c_zs, maxfb_zs, 
            layer_thick_pd, layer_thick, elem_thick, elem_tops, elem_bottoms,
            Lthick_prev, Tprev_zs, Tini_zs, k_layer_zs, k_elem_zs, Q_zs, dzi, 
            Qbi, Hxdzi, T_base, Tc_ss, T_trans
        ) 

@jit(nopython=True, cache=True)
def zero_decompacted_mesh(
                            ntops, zs, kg_zs, kg_up_zs, Qg_zs, phi_o_zs, c_zs, 
                            maxfb_zs, layer_thick_pd, layer_thick, elem_thick, 
                            elem_tops, elem_bottoms, Lthick_prev, Tprev_zs, 
                            Tini_zs, k_layer_zs, k_elem_zs, Q_zs, dzi, 
                            Qbi, Hxdzi, T_base, Tc_ss, T_trans
):
    for i in range(ntops):
        zs[i] = 0.0
        kg_zs[i] = 0.0
        kg_up_zs[i] = 0.0
        Qg_zs[i] = 0.0
        phi_o_zs[i] = 0.0
        c_zs[i] = 0.0 
        maxfb_zs[i] = 0.0
        layer_thick_pd[i] = 0.0
        layer_thick[i] = 0.0
        elem_thick[i] = 0.0
        elem_tops[i] = 0.0 
        elem_bottoms[i] = 0.0
        Lthick_prev[i] = 0.0
        Tprev_zs[i] = 0.0
        Tini_zs[i] = 0.0
        k_layer_zs[i] = 0.0
        k_elem_zs[i] = 0.0
        Q_zs[i] = 0.0
        dzi[i] = 0.0
        Qbi[i] = 0.0
        Hxdzi[i] = 0.0
        T_base[i] = 0.0
        Tc_ss[i] = 0.0
        T_trans[i] = 0.0

                                
def apply_thermal_smoothing(model, rad_search_m):
    tops_list_bs = model.tops_list_bs
    event_dict_bs = model.event_dict_bs
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    AOI_np = model.AOI_np
    keys = list(event_dict_bs.keys())    
    ntops = len(tops_list_bs)
    for kk, event_ID in enumerate(keys):
        itop_event = event_dict_bs[event_ID][2]
        for jj in range(ntops):
            if jj <= itop_event:
                event_index = tops_list_bs[jj][14][event_ID]
                temp_xy = tops_list_bs[jj][38][event_index]
                temp_xy_np_tmp = np.asarray(temp_xy)                           
                temp_xy_np = np.zeros((nx, ny))
                # apply smoothing
                map_tools.low_pass_filter_zmap(
                                                dx, dy, nx, ny, AOI_np,
                                                rad_search_m, temp_xy_np_tmp, 
                                                temp_xy_np
                                            )
                # update main array with smoothed result
                tops_list_bs[jj][38][event_index] = np.copy(temp_xy_np)


#@jit(nopython=True, parallel=True)
def prev_temp_strcutures_loop(
                                event_ID, icount_tops, tot_sed_thick_prev, 
                                nx, ny, itype3D, inode, jnode, AOI_np, 
                                Lthick_xy, Tprev_xy, Lthick_prev, 
                                Tprev_zs
):
    for i in numba.prange(nx):
        for j in range(ny): # Rows
            AOI_flag = AOI_np[i][j]
            if itype3D == 0:
                if i == inode and j == jnode:
                    AOI_flag = AOI_flag
                else:
                    AOI_flag = 0
            if AOI_flag == 1:
                # get the event_index for the current event so 
                # the current layer geometry can be accessed
                #************************
                # Define layer parameters
                #************************
                Lthick = Lthick_xy[i][j]
                Tprev = Tprev_xy[i][j]
                #******************************************
                # Load layer parameters into 1D meshes
                #******************************************                                
                Lthick_prev[event_ID][i][j][icount_tops] = Lthick
                Tprev_zs[event_ID][i][j][icount_tops] = Tprev
                (
                    tot_sed_thick_prev[event_ID][i][j]
                ) = (
                        tot_sed_thick_prev[event_ID][i][j] 
                        + Lthick
                    )
                
    return tot_sed_thick_prev


def update_prev_temp_structures_and_thickness(
                                                event_ID, event_ID_prev, 
                                                ntops, itop_event_prev, 
                                                model, 
                                                itype3D, nx, ny, inode, jnode, 
                                                AOI_np, Lthick_prev, Tprev_zs,
                                                tot_sed_thick_prev
):
    itype3D = model.itype3D
    prev_temp_strcutures_loop_active = manage_parallel.manage_parallel(
                                            prev_temp_strcutures_loop, itype3D)
    #******************************************************
    # Determine previous temperature structure Tprev_zs 
    # and total sediment thickness
    #******************************************************
    icount_tops = 0                     
    for mm in range(ntops):                            
        jj = ntops - 1 - mm
        # We only consider tops that have been deposited
        # We also include unconformity surfaces as nodes
        if jj <= itop_event_prev:
            #*************************
            # Define event information
            #*************************
            event_index = model.tops_list_bs[jj][14][event_ID_prev]
            #*****************
            # Get numpy arrays
            #*****************                    
            Lthick_xy = np.copy(model.tops_list_bs[jj][2][event_index])
            Tprev_xy = np.copy(model.tops_list_bs[jj][38][event_index])
            #*******************************************************
            # Define decompacted previous temperature structure and 
            # total sediment thickness for each x-y grid node
            #******************************************************* 
            prev_temp_strcutures_loop_active(
                                        event_ID, icount_tops, 
                                        tot_sed_thick_prev, nx, ny, 
                                        itype3D, inode, jnode, AOI_np, 
                                        Lthick_xy, Tprev_xy, 
                                        Lthick_prev, Tprev_zs
                                    )
            model.tops_list_bs[jj][2][event_index] = np.copy(Lthick_xy)
            model.tops_list_bs[jj][38][event_index] = np.copy(Tprev_xy)
            icount_tops = icount_tops + 1               


#@jit(nopython=True, parallel=True)
def define_decompacted_1D_meshes_loop(
                                        event_ID, icount_tops, itype3D, 
                                        nx, ny, inode, jnode, 
                                        AOI_np, pwd_xy, base_level,
                                        k_grain_xy, Q_grain_xy, phi_o_xy,
                                        decay_km_xy, maxfb_xy, Lthick_xy,
                                        Lthick_final_xy, z_top_subsea_xy,
                                        tot_sed_thick, kg_zs, Qg_zs, phi_o_zs,
                                        c_zs, maxfb_zs, layer_thick_pd, zs
):
    for i in numba.prange(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if itype3D == 0:
                if i == inode and j == jnode:
                    AOI_flag = AOI_flag
                else:
                    AOI_flag = 0
            if AOI_flag == 1:
                pwd = pwd_xy[i][j]
                z_surf = base_level + pwd
                #************************
                # Define layer parameters
                #************************
                # W/m/K
                k_grain = k_grain_xy[i][j]
                # Matrix HP: microW/m^3 converted to W/m/m/m
                Q_grain = Q_grain_xy[i][j]/1e6
                # Surface Porosity % converted to fraction
                phi_o = phi_o_xy[i][j]/100.0
                # Porosity Decay Depth (km) converted to 1/m
                c = 1.0/(decay_km_xy[i][j]*1000)
                # maximum forward burial in meters
                maxfb = maxfb_xy[i][j]
                # layer thickness in meters
                Lthick = Lthick_xy[i][j]                  
                Lthick_final = Lthick_final_xy[i][j]
                # subsea depth of top in meters
                z_top_subsea = z_top_subsea_xy[i][j]
                # submud depth of layer top in meters
                z_top = z_top_subsea - z_surf
                #******************************************
                # Load layer parameters into 1D meshes
                #******************************************
                kg_zs[event_ID][i][j][icount_tops] = k_grain
                Qg_zs[event_ID][i][j][icount_tops] = Q_grain
                phi_o_zs[event_ID][i][j][icount_tops] = phi_o
                c_zs[event_ID][i][j][icount_tops] = c
                maxfb_zs[event_ID][i][j][icount_tops] = maxfb
                layer_thick_pd[event_ID][i][j][icount_tops] = Lthick_final
                zs[event_ID][i][j][icount_tops] = z_top/1000.0
                (
                    tot_sed_thick[event_ID][i][j]
                ) = tot_sed_thick[event_ID][i][j] + Lthick

    
def define_decompacted_1D_meshes(
                                    event_ID, event_ID_last_bs, 
                                    itype3D, inode, jnode,
                                    itop_event, ntops, model,
                                    nx, ny, base_level, pwd_xy, AOI_np,
                                    kg_zs, Qg_zs, phi_o_zs, c_zs,
                                    maxfb_zs, layer_thick_pd, zs, 
                                    tot_sed_thick
):
    itype3D = model.itype3D
    define_decompacted_1D_meshes_loop_active = manage_parallel.manage_parallel(
                                    define_decompacted_1D_meshes_loop, itype3D)
    #********************************************
    # Define decompacted mesh and rock properties 
    # at top locations
    #********************************************
    icount_tops = 0
    nnodes = 0
    for mm in range(ntops):    
        jj = ntops - 1 - mm
        if jj <= itop_event:
            #*************************
            # Define event information
            #*************************
            # We only consider tops that have been 
            # deposited and erosional unconformities that
            # are present
            nnodes = nnodes + 1
            # get the event_list index for the current event 
            # so the current layer geometry can be accessed
            event_index = model.tops_list_bs[jj][14][event_ID]
            # Get the event list index for the final event
            event_index_final = model.tops_list_bs[jj][14][event_ID_last_bs]
            #*****************
            # Get numpy arrays
            #*****************
            k_grain_xy = model.tops_list_bs[jj][30]
            Q_grain_xy = model.tops_list_bs[jj][29]
            phi_o_xy = model.tops_list_bs[jj][33]
            decay_km_xy = model.tops_list_bs[jj][34]
            maxfb_xy = model.tops_list_bs[jj][46][event_index]
            Lthick_xy = model.tops_list_bs[jj][2][event_index]
            Lthick_final_xy = model.tops_list_bs[jj][2][event_index_final]
            z_top_subsea_xy = model.tops_list_bs[jj][1][event_index]
            #*******************************************************
            # Define decompacted meshes and rock properties for each
            # x-y grid node
            #*******************************************************
            define_decompacted_1D_meshes_loop_active(
                                        event_ID, icount_tops, itype3D, 
                                        nx, ny, inode, jnode, 
                                        AOI_np, pwd_xy, base_level,
                                        k_grain_xy, Q_grain_xy, phi_o_xy,
                                        decay_km_xy, maxfb_xy, Lthick_xy,
                                        Lthick_final_xy, z_top_subsea_xy,
                                        tot_sed_thick, kg_zs, Qg_zs, phi_o_zs,
                                        c_zs, maxfb_zs, layer_thick_pd, zs
                                    )
            icount_tops = icount_tops + 1
    inode_max = nnodes - 1
    return inode_max, nnodes


#@jit(nopython=True, parallel=True)
def calc_temp_grid(
                    event_ID, nx, ny, itype3D, AOI_np, inode, jnode, 
                    age_event, age_event_prev, iuse_temp_dep_k, 
                    nrelax, nsublayers, nnodes, sec_per_myr,
                    tot_sed_thick_prev, tot_sed_thick,
                    inode_max, event_type_ID, kappa_lith, 
                    iuse_anomalous_heatflow, ahf_max, 
                    bghf_xy, erothick_xy, ahf_xy, 
                    hf_reduc_fac_xy, T_top_xy, 
                    zs, kg_zs, Qg_zs, phi_o_zs, Q_zs, c_zs, 
                    maxfb_zs, Tprev_zs, Tini_zs, Lthick_prev, 
                    layer_thick, elem_thick, elem_tops, 
                    elem_bottoms, layer_thick_pd, 
                    k_layer_zs, k_elem_zs, kg_up_zs, dzi, Hxdzi, 
                    Qbi, T_base, Tc_ss, T_trans, k_water, Q_water
):
    for i in numba.prange(nx):
        for j in range(ny): # Rows
            AOI_flag = AOI_np[i][j]
            if itype3D == 0:
                if i == inode and j == jnode:
                    AOI_flag = AOI_flag
                else:
                    AOI_flag = 0
            if AOI_flag == 1:    
                # background heat flow mW/m/m
                bghf = bghf_xy[i][j]
                # erosion thickness in meters
                erothick = erothick_xy[i][j]
                # anomalous heat flow mW/m/m
                ahf = ahf_xy[i][j]
                # heat flow reduction factor
                hfred_fac = hf_reduc_fac_xy[i][j]
                # Surface temperature C
                T_top = T_top_xy[i][j]
                        
                Tini_zs[event_ID][i][j][0] = T_top
                
                ahf = ahf/hfred_fac
                ahf = min(ahf, ahf_max)               
                if iuse_anomalous_heatflow == 1:
                    # background heat flow + anomalous heat flow W/m/m
                    q_bottom = - (ahf + bghf)/1000.0
                else:
                    q_bottom = -bghf/1000.0
                calc_temp(
                            age_event, age_event_prev, iuse_temp_dep_k, 
                            nrelax, nsublayers, nnodes, 
                            inode_max, event_type_ID, kappa_lith, 
                            sec_per_myr, T_top, q_bottom, 
                            tot_sed_thick_prev[event_ID][i][j], 
                            tot_sed_thick[event_ID][i][j],
                            zs[event_ID][i][j], 
                            kg_zs[event_ID][i][j], 
                            Qg_zs[event_ID][i][j], 
                            phi_o_zs[event_ID][i][j], 
                            Q_zs[event_ID][i][j], 
                            c_zs[event_ID][i][j], 
                            maxfb_zs[event_ID][i][j], 
                            Tprev_zs[event_ID][i][j], 
                            Tini_zs[event_ID][i][j], 
                            Lthick_prev[event_ID][i][j], 
                            layer_thick[event_ID][i][j], 
                            elem_thick[event_ID][i][j], 
                            elem_tops[event_ID][i][j], 
                            elem_bottoms[event_ID][i][j], 
                            layer_thick_pd[event_ID][i][j], 
                            erothick, 
                            k_layer_zs[event_ID][i][j], 
                            k_elem_zs[event_ID][i][j], 
                            kg_up_zs[event_ID][i][j], 
                            dzi[event_ID][i][j], 
                            Hxdzi[event_ID][i][j], 
                            Qbi[event_ID][i][j], 
                            T_base[event_ID][i][j], 
                            Tc_ss[event_ID][i][j], 
                            T_trans[event_ID][i][j], 
                            k_water, Q_water
                        )    


#@jit(nopython=True, parallel=True)
def update_main_temp_arrays_loop(
                                    event_ID, icount_tops, AOI_np, 
                                    nx, ny, inode, jnode, itype3D,
                                    Tc_ss, T_trans, 
                                    iuse_trans, Tss_xy, Ttrans_xy
):
    for i in numba.prange(nx):
        for j in range(ny): # Rows
            AOI_flag = AOI_np[i][j]
            if itype3D == 0:
                if i == inode and j == jnode:
                    AOI_flag = AOI_flag
                else:
                    AOI_flag = 0
            if AOI_flag == 1:
                #*****************************************
                # Get temperature solutions from 1D meshes
                #*****************************************
                T = Tc_ss[event_ID][i][j][icount_tops]
                T_t = T_trans[event_ID][i][j][icount_tops]
                #******************************
                # Update master data structures
                #******************************
                Tss_xy[i][j] = T
                if iuse_trans == 1:
                    Ttrans_xy[i][j] = T_t
                else:
                    Ttrans_xy[i][j] = T


def update_main_temp_arrays(
                            event_ID, itop_event, itype3D, 
                            ntops, model, iuse_trans,
                            nx, ny, inode, jnode, AOI_np,
                            Tc_ss, T_trans
):
    update_main_temp_arrays_loop_active = manage_parallel.manage_parallel(
                                         update_main_temp_arrays_loop, itype3D)   
    icount_tops = 0
    for mm in range(ntops):
        jj = ntops-1-mm                            
        if jj <= itop_event:
            # We only consider tops that have 
            # been deposited                                
            event_index = model.tops_list_bs[jj][14][event_ID]
            Tss_xy = np.copy(model.tops_list_bs[jj][37][event_index])
            Ttrans_xy = np.copy(model.tops_list_bs[jj][38][event_index])
            itype3D = model.itype3D         
            update_main_temp_arrays_loop_active(
                                    event_ID, icount_tops, AOI_np, 
                                    nx, ny, inode, jnode, itype3D,
                                    Tc_ss, T_trans, 
                                    iuse_trans, Tss_xy, Ttrans_xy
                                )
            model.tops_list_bs[jj][37][event_index] = np.copy(Tss_xy)
            model.tops_list_bs[jj][38][event_index] = np.copy(Ttrans_xy)            
            icount_tops = icount_tops + 1 
    
    
def initialize_decompacted_1Dmesh_for_grid(nevents, ntops, nx, ny):
    # depth of tops (kilometers)
    zs = np.zeros((nevents, nx, ny, ntops))
    # grain conductivity at top locations (W/m/K)
    kg_zs = np.zeros((nevents, nx, ny, ntops))
    # updated grain conductivity at top locations (W/m/K)
    kg_up_zs = np.zeros((nevents, nx, ny, ntops))
    # grain heat production at top location (W/m/m/m)
    Qg_zs = np.zeros((nevents, nx, ny, ntops))
    # surface porosity fraction at top location
    phi_o_zs = np.zeros((nevents, nx, ny, ntops))
    # porosity decay depth 1/m
    c_zs = np.zeros((nevents, nx, ny, ntops))
    # maximum forward burial depth at top 
    # locations (meters)
    maxfb_zs = np.zeros((nevents, nx, ny, ntops))
    # thickness of layers (present-day) in meters
    layer_thick_pd = np.zeros((nevents, nx, ny, ntops))
    # thickness of layers in meters
    layer_thick = np.zeros((nevents, nx, ny, ntops))
    # thickness of elements centered on layer tops  (meters)
    elem_thick = np.zeros((nevents, nx, ny, ntops))
    # depth of top centerd element tops (meters)
    elem_tops = np.zeros((nevents, nx, ny, ntops))
     # depth of top centerd element tops (meters)
    elem_bottoms = np.zeros((nevents, nx, ny, ntops))
    # depth of tops of previous event (kilometers)
    Lthick_prev = np.zeros((nevents, nx, ny, ntops))
    # Previous temperature in previous mesh configuration (C)
    Tprev_zs = np.zeros((nevents, nx, ny, ntops))
    # Initial temperature in current mesh configuration (C)
    Tini_zs = np.zeros((nevents, nx, ny, ntops))                        
    # harmonic average conductivity layer using 
    # sublayering (W/m/K)
    k_layer_zs = np.zeros((nevents, nx, ny, ntops))
    # harmonic average conductivity for elem defined bu mid-points between
    # stratigraphic tops (W/m/K)
    k_elem_zs = np.zeros((nevents, nx, ny, ntops))
    # average heat production of layer centered at top 
    # location (W/m^3)
    Q_zs = np.zeros((nevents, nx, ny, ntops))
    # Initialize arrays used in steady state functions
    # depth of tops (meters) array used in 
    # steady state function
    dzi = np.zeros((nevents, nx, ny, ntops))
    # array of products use in steady state function
    Hxdzi = np.zeros((nevents, nx, ny, ntops))
    # array of products used in steady-state function
    Qbi = np.zeros((nevents, nx, ny, ntops))
    # Initialize solution arrays
    # Steady-state solution at the base of layers 
    # centered on tops (C)
    T_base = np.zeros((nevents, nx, ny, ntops))
    # steady-state solution interpolated to top 
    # locations (C)
    Tc_ss = np.zeros((nevents, nx, ny, ntops))
    # approximate transient (C)
    T_trans = np.zeros((nevents, nx, ny, ntops))
    
    tot_sed_thick = np.zeros((nevents, nx, ny))
    tot_sed_thick_prev = np.zeros((nevents, nx, ny))
    
    return (
            zs, kg_zs, kg_up_zs, Qg_zs, phi_o_zs, c_zs, maxfb_zs, 
            layer_thick_pd, layer_thick, elem_thick, elem_tops, elem_bottoms,
            Lthick_prev, Tprev_zs, Tini_zs, k_layer_zs, k_elem_zs, Q_zs, dzi, 
            Qbi, Hxdzi, T_base, Tc_ss, T_trans, 
            tot_sed_thick, tot_sed_thick_prev
        ) 

def surface_temp_grid_event(
                            event_ID, age_event, itype3D, icalc_SWIT, 
                            APWP_dict, surf_temp_xy, splate, nx, ny, dx, dy, 
                            lat_LL, lon_LL, T_top_input, swit_list, pwd_xy
):
    
    ddy = float(ny)*dy
    # 111 km per deg of latitude
    lat = lat_LL + ddy/111000.0
    # This is just an estimate
    ddx = float(nx)*dx
    meters_per_dlon = math.cos(lat*math.pi/180.0)*111000.0
    lon = lon_LL - ddx/meters_per_dlon 
    if icalc_SWIT == 1:
        T_top_xy = np.zeros((nx,ny))
        for i in range(nx):
            for j in range(ny):
                pwd = pwd_xy[i][j]
                T_top = calc_SWIT(
                                    APWP_dict, surf_temp_xy, 
                                    lat, lon, splate, 
                                    age_event, pwd
                                )
                T_top_xy[i][j] = T_top
    else:
        T_top_xy = np.ones((nx,ny))
        if itype3D == 1:
            T_top = T_top_input
        else:
            T_top = swit_list[event_ID]
        T_top_xy = T_top_xy*T_top    
    return T_top_xy


def calculate_thermal_history_para(model, process, ioutput_main):
    # Unpack objects for jitted functions
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    AOI_np = model.AOI_np  
    APWP_dict = model.APWP_dict
    surf_temp_xy = model.surf_temp_xy
    hf_reduc_fac_xy = model.hf_reduc_fac_xy 
    icalc_SWIT = model.icalc_SWIT
    lat_LL = model.lat_LL
    lon_LL = model.lon_LL
    splate = model.splate
    T_top_input = model.T_top_input
    iuse_trans = model.iuse_trans 
    iuse_anomalous_heatflow = model.iuse_anomalous_heatflow
    iuse_temp_dep_k = model.iuse_temp_dep_k
    nrelax = model.nrelax
    nsublayers = model.nsublayers
    kappa_lith = model.kappa_lith
    k_water = model.k_water
    Q_water = model.Q_water
    bghf_xy = model.bghf_xy
    itype3D = model.itype3D
    inode = model.inode
    jnode = model.jnode
    swit_list = model.swit_list
    ahf_max = model.ahf_max
    
    sec_per_myr = 365*24*60*60*1000000
    keys = list(model.event_dict_bs.keys())
    nevents_bs = len(keys)
    event_ID_list_bs = keys[:]
    event_ID_last_bs = event_ID_list_bs[nevents_bs-1]
    ntops = len(model.tops_list_bs)
    
    if iuse_temp_dep_k == 0:
        nrelax = 1
    
    tt1 = time.time()
    (
    zs, kg_zs, kg_up_zs, Qg_zs, phi_o_zs, c_zs, maxfb_zs, 
    layer_thick_pd, layer_thick, elem_thick, elem_tops, elem_bottoms,
    Lthick_prev, Tprev_zs, Tini_zs, k_layer_zs, k_elem_zs, Q_zs, dzi, 
    Qbi, Hxdzi, T_base, Tc_ss, T_trans, 
    tot_sed_thick, tot_sed_thick_prev
    ) = initialize_decompacted_1Dmesh_for_grid(nevents_bs, ntops, nx, ny)
    tt2 = time.time()
    print_funcs.print_finfo(
                            ioutput_main, process, 
                            "Created 1D meshes for thermal history", tt2-tt1)
    itype3D = model.itype3D
    calc_temp_grid_active = manage_parallel.manage_parallel(
                                                       calc_temp_grid, itype3D)
    # Loop over each event          
    for event_ID, key in enumerate(keys):
        # Loop over events from old to young
        if event_ID > 0: 
            # Skip the oldest node since this is "basement rock"
            # event_type
            #--------------------------------------
            # Deposition : simple deposition event
            # Erosion_and_Deposition : deposition followed by 
            #                          erosion
            # Erosion : just erosion (obsolete feature) 
            event_type = model.event_dict_bs[event_ID][1]
            if event_type == "Deposition":
                event_type_ID = 0
            else:
                event_type_ID = 1
            # Get event information
            itop_event = model.event_dict_bs[event_ID][2]
            age_event = model.event_dict_bs[event_ID][0]                        
            event_ID_prev = event_ID - 1
            itop_event_prev = model.event_dict_bs[event_ID_prev][2]
            age_event_prev = model.event_dict_bs[event_ID_prev][0]
            base_level = model.deltaSL_list[event_ID]
            # Get event maps
            pwd_xy = model.event_dict_bs[event_ID][5]
            ahf_xy = model.event_dict_bs[event_ID][15]
            erothick_xy = model.event_dict_bs[event_ID][3]
            T_top_xy = surface_temp_grid_event(
                            event_ID, age_event, itype3D, icalc_SWIT, 
                            APWP_dict, surf_temp_xy, splate, nx, ny, dx, dy, 
                            lat_LL, lon_LL, T_top_input, swit_list, pwd_xy
                            )
            #******************************************************
            # Determine previous temperature structure Tprev_zs 
            # and total sediment thickness for this event
            #******************************************************
            update_prev_temp_structures_and_thickness(
                                                event_ID, event_ID_prev, 
                                                ntops, itop_event_prev, 
                                                model, 
                                                itype3D, nx, ny, inode, jnode, 
                                                AOI_np, Lthick_prev, Tprev_zs,
                                                tot_sed_thick_prev
                                                )
            #********************************************
            # Define decompacted mesh and rock properties 
            # at top locations for this event
            #********************************************
            (
                 inode_max, 
                 nnodes,
            ) = define_decompacted_1D_meshes(
                                            event_ID, event_ID_last_bs, 
                                            itype3D, inode, jnode,
                                            itop_event, ntops, model,
                                            nx, ny, base_level, pwd_xy, AOI_np,
                                            kg_zs, Qg_zs, phi_o_zs, c_zs,
                                            maxfb_zs, layer_thick_pd, zs,
                                            tot_sed_thick
                                            )
            #********************************
            # Calculate temperature for event
            #********************************
            calc_temp_grid_active(
                                event_ID, nx, ny, itype3D, AOI_np, 
                                inode, jnode, 
                                age_event, age_event_prev, iuse_temp_dep_k, 
                                nrelax, nsublayers, nnodes, sec_per_myr,
                                tot_sed_thick_prev, tot_sed_thick,
                                inode_max, event_type_ID, kappa_lith, 
                                iuse_anomalous_heatflow, ahf_max, 
                                bghf_xy, erothick_xy, ahf_xy, 
                                hf_reduc_fac_xy, T_top_xy, 
                                zs, kg_zs, Qg_zs, phi_o_zs, Q_zs, c_zs, 
                                maxfb_zs, Tprev_zs, Tini_zs, Lthick_prev, 
                                layer_thick, elem_thick, elem_tops, 
                                elem_bottoms, layer_thick_pd, 
                                k_layer_zs, k_elem_zs, kg_up_zs, dzi, Hxdzi, 
                                Qbi, T_base, Tc_ss, T_trans, k_water, Q_water
                                )
            #*********************************************
            # Save steady-state and transient solutions to 
            # main arrays
            #*********************************************
            update_main_temp_arrays(
                                    event_ID, itop_event, itype3D, 
                                    ntops, model, iuse_trans,
                                    nx, ny, inode, jnode, AOI_np,
                                    Tc_ss, T_trans
                                    )
            

if __name__ == "__main__":
    import fileIO
    lat = 53.47
    lon = -55.79
    pwd = 161.54
    age_event = 0.0
    splate = "NAM"
    main_path = "C:\\Users\\eaknell\\Desktop\\IPAtoolkit3D_1.0\\IPA_Python"
    input_file_path = main_path + "\\ipa_data\\surfaceTCvsMa.csv"
    (
        surf_temp_age, 
        surf_temp_lat, 
        surf_temp_xy
    ) = fileIO.read_surf_temp_file_csv(input_file_path)
            
    input_file_path = main_path + "\\ipa_data\\APWP.csv"
    APWP_dict = fileIO.read_APWP_file_csv(input_file_path)
    T_top = calc_SWIT(
                    APWP_dict, surf_temp_xy, 
                    lat, lon, splate, 
                    age_event, pwd)
    print("SWIT = ", T_top)