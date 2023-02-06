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
import math
import map_tools


def calc_depth_from_TWT_and_vint(
                                    nx, ny, tops_list_bs, AOI_np, 
                                    rhow, iuse_comp_law, niter
                                ):
    # porosity cuttoff for velocity model
    phi_cut = 0.4
    if iuse_comp_law == 0:
        niter = 1   
    ntops = len(tops_list_bs)
    itop_max = ntops - 1
    for i in range(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:
                for nn in range(niter):
                    # Calculate depths for all tops
                    for jj in range(ntops):
                        # calc water depth
                        twt_wb = tops_list_bs[itop_max][18][i][j]
                        vint = 1500.0
                        wd = twt_wb/2.0/1000.0*vint
                        # initialize with depth of water bottom
                        dtot = wd
                        for kk in range(ntops):
                            # Move from youngest to oldest
                            mm = ntops-1-kk                                    
                            if mm >= jj and mm < itop_max:
                                # Only use tops at or above current top 
                                # (i.e. younger)
                                
                                # (shallower top)
                                twt1 = tops_list_bs[mm + 1][18][i][j]
                                # (deeper top)
                                twt2 = tops_list_bs[mm][18][i][j]
                                dt_int_sec = (twt2 - twt1)/2.0/1000.0
                                vavg = tops_list_bs[mm + 1][24][i][j]
                                dz = dt_int_sec*vavg #vint # meters
                                dtot = dtot + dz
                        # integrated depth in meters
                        tops_list_bs[jj][23][i][j] = dtot                    
                    # Update bulk densities and interval velocity
                    if iuse_comp_law == 1:
                        # Update bulk density and interval 
                        # velocity for all tops
                        for jj in range(ntops):
                            # calc water depth
                            twt_wb = tops_list_bs[itop_max][18][i][j]
                            vint = 1500.0
                            wd_m = twt_wb/2.0/1000.0*vint
                            # top of layer
                            z_subsea_m = tops_list_bs[jj][23][i][j]
                            z_km = (z_subsea_m-wd_m)/1000
                            # bottom of layer
                            # If we are at the bottom of the sediment 
                            # just use node
                            if jj > 0:
                                z_subsea_m_bottom = (
                                                tops_list_bs[jj-1][23][i][j]
                                            )
                            else:
                                z_subsea_m_bottom = tops_list_bs[jj][23][i][j]                            
                            z_km_bottom = (z_subsea_m_bottom-wd_m)/1000
                            # Update interval velocity using compaction law
                            phi_o =  tops_list_bs[jj][33][i][j]/100
                            c =  tops_list_bs[jj][34][i][j]
                            rhog =  tops_list_bs[jj][32][i][j]
                            phi_bulk = phi_o*math.exp(-z_km/c)
                            rhob = (1.0-phi_bulk)*rhog + phi_bulk*rhow
                            phi_bulk_bottom = phi_o*math.exp(-z_km_bottom/c)
                            rhob_bottom = (
                                            (1.0 - phi_bulk_bottom)*rhog 
                                            + phi_bulk_bottom*rhow
                                        )
                            vmat = tops_list_bs[jj][28][i][j]
                            GL_d = tops_list_bs[jj][35][i][j]
                            GL_f = tops_list_bs[jj][36][i][j]
                            if GL_d > 0.0 and GL_f > 0.0: 
                                if phi_bulk >= phi_cut:
                                    vnode_comp = 1500.0
                                else:
                                    vnode_comp = (
                                        1000.0*(rhob/1000.0/GL_d)**(1.0/GL_f)
                                    )
                                if phi_bulk_bottom >= phi_cut:
                                    vnode_comp_bottom = 1500.0
                                else:
                                    vnode_comp_bottom = (
                                            1000.0*
                                            (rhob_bottom/1000.0/GL_d)
                                                                **(1.0/GL_f)
                                    ) 
                            else:
                                if phi_bulk >= phi_cut:
                                    vnode_comp = 1500.0
                                else:
                                    vnode_comp = (
                                                    1.0/((1 - phi_bulk)/vmat
                                                   + phi_bulk/1500.0)
                                                )
                                if phi_bulk_bottom >= phi_cut:
                                    vnode_comp_bottom = 1500.0
                                else:                                
                                    vnode_comp_bottom = (
                                            1.0/((1-phi_bulk_bottom)/vmat
                                           + phi_bulk_bottom/1500.0)
                                        )
                            vint = (vnode_comp + vnode_comp_bottom)/2.0
                            if vint < 1500.0:
                                vint = 1500.0
                            tops_list_bs[jj][24][i][j] = vint

           
def load_TWT_maps_and_calc_depth_map(
                                        input_path, output_path, 
                                        tops_list_bs, event_dict_bs, 
                                        rhow, iuse_comp_law, niter, 
                                        AOI_np, master_xy, nx, ny, 
                                        dx, dy, xmin, xmax, ymin, ymax
):
    ntops = len(tops_list_bs)
    dum = np.zeros((nx,ny))
    for jj in range(ntops):
        map_file_name = tops_list_bs[jj][19]
        try:
            depth_xy = map_tools.read_ZMAP_interp_to_master(
                                            -99999.0, master_xy, input_path, 
                                            map_file_name, nx, ny, xmin, ymin, 
                                            dx, dy, AOI_np
                                        )
        except:
            depth_xy = dum.tolist()
        tops_list_bs[jj][18] = np.copy(depth_xy)    
    # Initialize TVD maps and interval velocity maps
    depth_xy = np.zeros((nx,ny))
    int_v_xy = np.zeros((nx,ny))
    for jj in range(ntops):
        v = tops_list_bs[jj][20]
        for i in range(nx):
            for j in range(ny):
                int_v_xy[i][j] = v
        tops_list_bs[jj][23] = np.copy(depth_xy)
        tops_list_bs[jj][24] = np.copy(int_v_xy)
    # Calculate depth from TWT and vint
    calc_depth_from_TWT_and_vint(
                                    nx, ny, tops_list_bs, AOI_np, rhow, 
                                    iuse_comp_law, niter
                                )
    for jj in range(ntops):
        depth_xy = tops_list_bs[jj][23]
        age = tops_list_bs[jj][4]
        file_name = "Top_TVDcalc_"+str(age)
        # update depth map name
        tops_list_bs[jj][15] = file_name + ".dat"
        # Update WD file name
        if jj == ntops-1:
            tops_list_bs[jj][16] = file_name + ".dat"
            # define all event keys (integers from 0 to N) 
            # from oldest to youngest
            keys = list(event_dict_bs.keys())
            nevents_bs = len(keys)
            event_ID_list_bs = keys[:]
            # event ID of last event. This event ended at present day.
            event_ID_last_bs = event_ID_list_bs[nevents_bs-1]
            event_dict_bs[event_ID_last_bs][11] = file_name + ".dat"
        map_tools.make_output_file_ZMAP_v4(
                                            input_path, file_name, depth_xy, 
                                            nx, ny, dx, dy, 
                                            xmin, xmax, ymin, ymax, AOI_np
                                        )        

    
def calc_VELOC_TWT_maps(
                        input_path, output_path, 
                        tops_list_bs, event_dict_bs, rhow, iuse_comp_law, 
                        AOI_np, master_xy, nx, ny, dx, dy, 
                        xmin, xmax, ymin, ymax
): 
    phi_cut = 0.4
    ntops = len(tops_list_bs)
    # Initialize TWT maps, final depth and interval velocity maps
    for jj in range(ntops):
        v = tops_list_bs[jj][20]
        depth_xy = np.zeros((nx,ny))
        int_v_xy = np.zeros((nx,ny))
        for i in range(nx):
            for j in range(ny):
                int_v_xy[i,j] = v
        tops_list_bs[jj][18] = np.copy(depth_xy)
        tops_list_bs[jj][23] = np.copy(depth_xy)
        tops_list_bs[jj][24] = np.copy(int_v_xy)
    # Fill final depth list
    keys = list(event_dict_bs.keys())
    nevents_bs = len(keys)
    event_ID_list_bs = keys[:]
    event_ID_last_bs = event_ID_list_bs[nevents_bs-1]
    WD = 0.0 # water bottom in meters
    twt = 0.0
    dtwt = 0.0
    for inode in range(nx):
        for jnode in range(ny):
            AOI_flag = AOI_np[inode][jnode]
            if AOI_flag == 1:
                for kk in range(ntops): 
                    # Loop over tops from young to old 
                    mm = ntops - 1 - kk
                    event_index2 = tops_list_bs[mm][14][event_ID_last_bs]
                    # Depth of base of layer
                    z_top_ss2 = tops_list_bs[mm][1][event_index2][inode][jnode]
                    tops_list_bs[mm][23][inode][jnode] = z_top_ss2
                    vmat = tops_list_bs[mm][28][inode][jnode]
                    GL_d = tops_list_bs[mm][35][inode][jnode]
                    GL_f = tops_list_bs[mm][36][inode][jnode]
                    phi_o =  tops_list_bs[mm][33][inode][jnode]/100
                    c =  tops_list_bs[mm][34][inode][jnode]
                    rhog =  tops_list_bs[mm][32][inode][jnode]
                    if mm == ntops-1: 
                        # first layer is water
                        WD = z_top_ss2
                        if WD <= 0.0:
                            WD = 0.0
                            z_top_ss1 = z_top_ss2 
                        else:
                            z_top_ss1 = 0.0                    
                        vint = 1500.0
                        twt = 2*WD/vint*1000
                    elif mm < ntops-1:
                        phi_bulk_bottom = phi_o*math.exp(-(z_top_ss2-WD)/c)
                        rhob_bottom = (
                                         (1.0-phi_bulk_bottom)*rhog 
                                       + phi_bulk_bottom*rhow
                                      )
                        if GL_d > 0.0 and GL_f > 0.0: 
                            if phi_bulk_bottom >= phi_cut:
                                vnode_comp_bottom = 1500.0
                            else:
                                vnode_comp_bottom = (
                                        1000.0*(rhob_bottom/1000.0/GL_d)
                                                                **(1.0/GL_f)
                                    )
                        else: 
                            if phi_bulk_bottom >= phi_cut:
                                vnode_comp_bottom = 1500.0
                            else:                                
                                vnode_comp_bottom = (
                                          1.0/((1-phi_bulk_bottom)/vmat 
                                        + phi_bulk_bottom/1500.0)
                                    )
                        event_index1 = tops_list_bs[mm+1][14][event_ID_last_bs]
                        # Depth for top of layer
                        z_top_ss1 = tops_list_bs[mm+1][1][event_index1]\
                                                                [inode][jnode]
                        phi_bulk = phi_o*math.exp(-(z_top_ss1-WD)/c)
                        rhob = (1.0-phi_bulk)*rhog + phi_bulk*rhow
                        dthick = z_top_ss2 - z_top_ss1
                        if GL_d > 0.0 and GL_f > 0.0: 
                            if phi_bulk >= phi_cut:
                                vnode_comp = 1500.0
                            else:
                                vnode_comp = (
                                                1000.0*(rhob/1000.0/GL_d)
                                                                **(1.0/GL_f)
                                            )
                        else:
                            if phi_bulk >= phi_cut:
                                vnode_comp = 1500.0
                            else:
                                vnode_comp = (
                                        1.0/(
                                               (1-phi_bulk)/vmat 
                                             + phi_bulk/1500.0
                                             )
                                        )
                        vint = (vnode_comp_bottom + vnode_comp)/2.0
                        dtwt = 2*dthick/vint*1000
                        twt = twt + dtwt
                    tops_list_bs[mm][18][inode][jnode] = twt