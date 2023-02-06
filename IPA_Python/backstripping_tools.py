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
import numpy as np
from numba import jit
import map_tools
import data_exporter


def make_constant_rifting_timing_maps(age_start, age_end, Lx, Ly, nx, ny, 
                                      dx, dy, xmin, xmax, ymin, ymax):
    age_s_xy = np.zeros((nx,ny))
    age_e_xy = np.zeros((nx,ny))
    # Loop over all x-y locations
    for i in range(nx):
        for j in range(ny): # Rows
            age_s_xy[i,j] = age_start
            age_e_xy[i,j] = age_end
    return age_s_xy, age_e_xy


@jit(nopython=True, cache=True)
def sed_thickness_and_bulk_density_sums(
                                        nx, ny, AOI_np, 
                                        thick_xy, rho_layer_xy, 
                                        sed_thick_xy, rhob_xy
):
   for i in range(nx):
        for j in range(ny):                            
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:
                thick = thick_xy[i][j]
                rho_layer = rho_layer_xy[i][j]
                sed_thick_xy[i,j] = sed_thick_xy[i,j] + thick
                rhob_xy[i,j] = rhob_xy[i,j] + rho_layer*thick


@jit(nopython=True, cache=True)
def sed_thick_rho_load_tts(
                            nx, ny, AOI_np,
                            rho_water, rho_mantle, dSL, pwd_xy,
                            thick_xy, rho_layer_xy, sed_thick_xy, 
                            rhob_xy, q_sed_xy, ttsub_xy
):
    for i in range(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:
                pwd = pwd_xy[i,j]
                sumit = rhob_xy[i,j]
                t_tot = sed_thick_xy[i,j]
                rhob_final = sumit/t_tot
                q_sed = (rhob_final - rho_water)*9.8*t_tot/1e6
                
                ttsub = (
                           pwd 
                         + t_tot*(rho_mantle-rhob_final)/(rho_mantle-rho_water)
                         + dSL*rho_mantle/(rho_mantle-rho_water)
                        )
                
                rhob_xy[i,j] = rhob_final
                sed_thick_xy[i,j] = t_tot
                q_sed_xy[i,j] = q_sed
                ttsub_xy[i,j] = ttsub
            else:
                rhob_xy[i,j] = -99999.0
                sed_thick_xy[i,j] = -99999.0
                q_sed_xy[i,j] = -99999.0


def update_sed_thick_rho_load_tts(imake_output, ioutput, model):
    # Unpack model object for jitted functions
    rho_water = model.rho_water
    rho_mantle = model.rho_mantle
    nx = model.nx
    ny = model.ny
    AOI_np = model.AOI_np
    
    keys = model.event_dict_bs.keys()
    ntops = len(model.tops_list_bs)
    # Looping over events from oldest to youngest
    for k, key in enumerate(keys):
        event_ID = key
        event_age = model.event_dict_bs[event_ID][0]
        pwd_xy = np.copy(model.event_dict_bs[event_ID][5])
        dSL = model.deltaSL_list[event_ID]
        itop_event = model.event_dict_bs[event_ID][2]
        sed_thick_xy = np.zeros((nx,ny))
        rhob_xy = np.zeros((nx,ny))
        q_sed_xy = np.zeros((nx,ny))
        ttsub_xy = np.zeros((nx,ny))
        for ii in range(ntops):      
            # We only consider tops that have been deposited   
            if ii <= itop_event:
                event_index = model.tops_list_bs[ii][14][event_ID]
                thick_xy = np.copy(model.tops_list_bs[ii][2][event_index])
                rho_layer_xy = np.copy(model.tops_list_bs[ii][17][event_index])
            
                sed_thickness_and_bulk_density_sums(
                                        nx, ny, AOI_np, 
                                        thick_xy, rho_layer_xy, 
                                        sed_thick_xy, rhob_xy)
        
        sed_thick_rho_load_tts(
                                nx, ny, AOI_np,
                                rho_water, rho_mantle, dSL, pwd_xy,
                                thick_xy, rho_layer_xy, sed_thick_xy, 
                                rhob_xy, q_sed_xy, ttsub_xy
                               )
        model.event_dict_bs[key][12] = np.copy(rhob_xy)
        model.event_dict_bs[key][20] = np.copy(q_sed_xy)
        model.event_dict_bs[key][21] = np.copy(sed_thick_xy)
        model.event_dict_bs[key][8] = np.copy(ttsub_xy)
        data_exporter.export_sed_load_and_tts(imake_output, event_age, key, 
                                              model, q_sed_xy, ttsub_xy)

                
#def interp_PWD(
#                PWD_interp_ages, tops_list_bs, event_dict_bs, 
#                ioutput, output_path, Lx, Ly, nx, ny, dx, dy, 
#                xmin, xmax, ymin, ymax, AOI_np
#):
#    keys = list(event_dict_bs.keys())
#    nevents_bs = len(keys)
#    for kk, PWD_age in enumerate(PWD_interp_ages):
#        pwd_new_xy = np.zeros((nx,ny))   
#        for i in range(nx):
#            for j in range(ny):
#                AOI_flag = AOI_np[i][j]
#                if AOI_flag == 1:
#                    PWD_ages_events = []
#                    PWD_events = []
#                    for k, key in enumerate(keys):
#                        event_ID = nevents_bs - 1 - k
#                        event_age = event_dict_bs[event_ID][0]
#                        pwd_xy = np.copy(event_dict_bs[event_ID][5])                 
#                        PWD_ages_events.append(event_age)
#                        PWD_events.append(pwd_xy[i][j])                 
#                    pwd_interp = math_tools.linear_interp_v2(
#                                        PWD_age, PWD_ages_events, PWD_events)
#                else:
#                    pwd_interp = -99999.0
#                pwd_new_xy[i,j] = pwd_interp
#        data_exporter.export_interpolated_PWD(
#                                            PWD_age, output_path, pwd_new_xy,
#                                            nx, ny, dx, dy, xmin, xmax, 
#                                            ymin, ymax, AOI_np
#                                        )


@jit(nopython=True, cache=True)
def res_sub_and_PWD_correction(
                                nx, ny, tt_sub_xy_np, pwd_xy_np, 
                                tt_sub_fw_xy_np, pwd_new_xy_np, 
                                res_sub_np, AOI_np
):
    for i in range(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:
                tt_sub_o = tt_sub_xy_np[i][j]
                pwd_o = pwd_xy_np[i][j]
                tt_sub_n = tt_sub_fw_xy_np[i][j]
                dsub = tt_sub_n-tt_sub_o
                pwd_n = pwd_o + dsub
                res = dsub
                limit_pwd = 0
                if limit_pwd == 1:
                    if pwd_n < 0.0:
                        pwd_n = 0.0
                        res = 0.0
            else:
                pwd_n = -99999.0
                res = -99999.0
            pwd_new_xy_np[i][j] = pwd_n
            res_sub_np[i][j] = res

            
def update_depth_maps_flex(model):
    #tops_list_bs = model.tops_list_bs
    #event_dict_bs = model.event_dict_bs
    #deltaSL_list = model.deltaSL_list
    nx = model.nx
    ny = model.ny
    
    # define all event keys (integers from 0 to N) from oldest to youngest
    keys = list(model.event_dict_bs.keys())
    nevents = len(keys)
    # Looping over events from oldest to youngest
    for k, key in enumerate(keys):
        # index of the top associated with this event
        itop = model.event_dict_bs[key][2]
        event_ID = key
        if k < nevents-1: 
            # Skip presentday
            pwd_xy = np.copy(model.event_dict_bs[event_ID][5])
            pwd_flex_xy = np.copy(model.event_dict_bs[event_ID][22])
            base_level = model.deltaSL_list[event_ID]
            ntops = len(model.tops_list_bs)
            for jj in range(ntops): 
                # Looping over all tops (i.e. events again)                
                # Only consider tops that have an ID less than top
                if jj <= itop:                    
                    event_index = model.tops_list_bs[jj][14][event_ID]                    
                    for i in range(nx):                       
                        for j in range(ny):                          
                            pwd_old = pwd_xy[i][j]
                            pwd_new = pwd_flex_xy[i][j]                            
                            z_surf_old = base_level + pwd_old
                            z_surf_new = base_level + pwd_new
                            zini = (
                                model.tops_list_bs[jj][1][event_index][i][j] 
                              - z_surf_old
                              )
                            zfinal = zini + z_surf_new                                             
                            model.tops_list_bs[jj][1][event_index][i][j] = (
                                                                        zfinal)

                            
def update_depth_maps(model):
    #event_dict_bs = model.event_dict_bs
    #tops_list_bs = model.tops_list_bs
    nx = model.nx
    ny = model.ny
    # define all event keys (integers from 0 to N) from oldest to youngest
    keys = list(model.event_dict_bs.keys())
    nevents = len(keys)
    for k, key in enumerate(keys): 
        # Looping over events from oldest to youngest
        # index of the top associated with this event
        itop = model.event_dict_bs[key][2]
        event_ID = key
        if k < nevents-1: 
            # Skip presentday
            ntops = len(model.tops_list_bs)
            for jj in range(ntops): 
                # Looping over all tops (i.e. events again)
                # Only consider tops that have an ID less than top
                if jj <= itop:
                    event_index = model.tops_list_bs[jj][14][event_ID]
                    for i in range(nx): # Columns
                        for j in range(ny): # Rows
                            rs = model.event_dict_bs[event_ID][13][i][j]
                            zini = model.tops_list_bs[jj][1][event_index][i][j]
                            zfinal = zini + rs
                            model.tops_list_bs[jj][1][event_index][i][j] = zfinal
                            
                            
def calculate_res_sub_and_corrected_PWD(ioutput, model):
    # Unpack model model for jitted functions
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    AOI_np = model.AOI_np
    rad_search_m = model.rad_search_m
    iupdate_PWD = model.iupdate_PWD
    iuse_flexure = model.iuse_flexure
    
    keys = list(model.event_dict_bs.keys())
    nevents_bs = len(keys)
    event_ID_list_bs = keys[:]
    event_ID_last_bs = event_ID_list_bs[nevents_bs-1]
    # Initialize numpy arrays
    tt_sub_xy_np = np.zeros((nx, ny))
    tt_sub_fw_xy_np = np.zeros((nx, ny))
    pwd_xy_np = np.zeros((nx, ny))
    res_sub_xy_np = np.zeros((nx, ny))
    # Looping over events from oldest to youngest
    for k, key in enumerate(keys):    
        if key != event_ID_last_bs:                   
            event_ID = key
            event_age = model.event_dict_bs[event_ID][0]        
            tt_sub_xy = np.copy(model.event_dict_bs[event_ID][8])
            tt_sub_fw_xy = np.copy(model.event_dict_bs[event_ID][14])
            pwd_xy = np.copy(model.event_dict_bs[event_ID][5])
            pwd_flex_xy = np.copy(model.event_dict_bs[event_ID][22])        
            # Create numpy arrays
            tt_sub_xy_np = np.copy(tt_sub_xy)
            tt_sub_fw_xy_np = np.copy(tt_sub_fw_xy)
            pwd_xy_np = np.copy(pwd_xy)        
            pwd_new_xy_np = np.zeros((nx, ny))        
            res_sub_and_PWD_correction(
                                        nx, ny, tt_sub_xy_np, pwd_xy_np, 
                                        tt_sub_fw_xy_np, pwd_new_xy_np, 
                                        res_sub_xy_np, AOI_np
                                    )
            # Apply low pass filter            
            pwd_new_lpf_xy_np = np.zeros((nx, ny))            
            map_tools.low_pass_filter_zmap(
                                            dx, dy, nx, ny, AOI_np, 
                                            rad_search_m, pwd_new_xy_np, 
                                            pwd_new_lpf_xy_np
                                        )            
            pwd_new_xy = np.copy(pwd_new_lpf_xy_np)            
            # Only update paleo-water depth if iupdate_PWD = 1
            if iupdate_PWD == 1:
                if iuse_flexure != 1:
                    model.event_dict_bs[event_ID][5] = np.copy(pwd_new_xy)            
            res_sub_xy = np.copy(res_sub_xy_np)
            model.event_dict_bs[event_ID][13] = np.copy(res_sub_xy)
            data_exporter.export_inverted_pwd_and_residual_sub(
                                                        model, event_age, 
                                                        pwd_new_xy, res_sub_xy
                                                        )
    if iupdate_PWD == 1:
        if iuse_flexure == 1:
            update_depth_maps_flex(model)
        else:
            # Update depth maps with new paleo-water depths
            update_depth_maps(model)    
    # Update main pwd only after updating depth maps if flexure is used
    if iuse_flexure == 1 and iupdate_PWD == 1:
        for k, key in enumerate(keys): 
            # Looping over events from oldest to youngest
            if key != event_ID_last_bs:
                event_ID = key
                pwd_flex_xy = np.copy(model.event_dict_bs[event_ID][22])
                model.event_dict_bs[event_ID][5] = np.copy(pwd_flex_xy)
    
    
