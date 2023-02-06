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
import compaction
import map_tools


def calc_total_erosion(event_dict_bs, tops_list_bs, nx, ny):
    """ Calculate total erosion for each event and each top
    """
    old_shift_xy = np.copy(event_dict_bs[0][4])
    keys = event_dict_bs.keys()
    for k, key in enumerate(keys): 
        # Looping over events from oldest to youngest
        # index of the top associated with this event
        itop = event_dict_bs[key][2]
        #event_type = tops_list_bs[itop][1]
        if k > 0:
            this_shift_xy = event_dict_bs[key][4]
            e_tot_xy = np.zeros((nx,ny))
            event_dict_bs[key][3] = e_tot_xy
            for i in range(nx):
                for j in range(ny):
                    s_new = this_shift_xy[i][j]
                    s_old = old_shift_xy[i][j]
                    s_diff = s_new - s_old
                    if s_diff < 0.0: # erosion
                        e_tot = -1.0*s_diff
                    else:
                        # Since s_diff > 0 this is a depo event and
                        # the total eroded thickness of the deposited layer
                        # is equal to the difference in shift map
                        e_tot = 0.0
                        # Total eroded thickness for top
                        tops_list_bs[itop][8][0][i][j] = s_diff   
                    event_dict_bs[key][3][i][j] = e_tot
            old_shift_xy = np.copy(this_shift_xy)
        else: 
            # First event always has zero erosion
            e_tot_xy = np.zeros((nx,ny))
            event_dict_bs[key][3] = e_tot_xy


def calc_erosion_for_tops(event_dict_bs, tops_list_bs, nx, ny):
    """
    For each event calculate the erosion for each top
    """
    keys = event_dict_bs.keys()
    for k, key in enumerate(keys): 
        # Looping over events from oldest to youngest
        # index of the top associated with this event
        itop = event_dict_bs[key][2]        
        for i in range(nx): 
            for j in range(ny): 
                emag_event = event_dict_bs[key][3][i][j]
                for jj in range(itop+1): 
                    # We need to loop from itop to older tops
                    kk = itop - jj
                    event_index = tops_list_bs[kk][14][key]
                    emag_top = tops_list_bs[kk][8][0][i][j]
                    emag_left = emag_event - emag_top
                    # 0 = old method that was bechmarked??
                    # 1 = new method for IPA toolkit that needs some fixing
                    itype_local = 1
                    if itype_local == 0:
                        if emag_left == 0.0: # was <= 0.0
                            tops_list_bs[kk][3][event_index][i][j] = emag_event
                            emag_event = 0
                        #else:
                        #    tops_list_bs[kk][3][event_index][i][j] = emag_top
                        #    emag_event = emag_event - emag_top
                    else:
                        if emag_left <= 0.0:
                            tops_list_bs[kk][3][event_index][i][j] = emag_event
                        else:
                            tops_list_bs[kk][3][event_index][i][j] = emag_top
                            emag_event = emag_event - emag_top


def max_burial_initialize(event_dict_bs, tops_list_bs, nx, ny):
    """ For each top initialize final maximum burial
    """
    # define all event keys (integers from 0 to N) from oldest to youngest
    keys = list(event_dict_bs.keys())
    nevents_bs = len(keys)
    event_ID_list_bs = keys[:]
    # event ID of last event. This event ended at present day.
    # Only working on last event (i.e. present-day)
    event_ID = event_ID_list_bs[nevents_bs-1]
    
    pwd_pd_xy = event_dict_bs[event_ID][5]
    
    ntops = len(tops_list_bs)
    for kk in range(ntops):
        event_index = tops_list_bs[kk][14][event_ID]        
        depth_xy_pd = tops_list_bs[kk][1][event_index]
        maxf_xy = np.zeros((nx,ny))
        for i in range(nx):
            for j in range(ny): # Rows
                z_depth = depth_xy_pd[i][j]
                pwd = pwd_pd_xy[i][j]
                # baselevel is assumed to be zero at present-day
                z_depth_sm = z_depth - pwd
#                # This following code was removed now that a forward 
#                # maximum burial step is used               
#                s_list_tmp = []               
#                s_old = 0.0
#                for event_ID in event_IDs: 
#                    # Loop over all events and collect shifts or delta_shifts           
#                    shift_xy = event_dict_bs[key][4]
#                    s_new = shift_xy[i][j]
#                    delta_s = s_new - s_old
#                    if delta_s < 0:
#                        s_list_tmp.append(abs(delta_s))
#                    else:
#                        s_list_tmp.append(abs(s_new))
#                s_max = max(s_list_tmp) # take the maximum value               
#                maxf = z_depth_sm - s_pd + s_max
                maxf = z_depth_sm                
                maxf_xy[i][j] = maxf            
        tops_list_bs[kk][12] = maxf_xy


def total_erosion_in_max_compaction_state(
                                        iuse_max_state, tops_list, event_dict, 
                                        nx, ny, Lx, Ly, dx, dy, 
                                        xmin, xmax, ymin, ymax, output_path
):
    keys = event_dict.keys()
    # Initialize total eroded thickness 
    # for this event in maximum compaction state 
    for ii, key in enumerate(keys):
        event_ID = key
        # Inititalize total erosion list for this event
        tot_ero_xy = []
        for i in range(nx): # Columns     
            ini_row = []
            for j in range(ny): # Rows
                ini_row.append(0.0)
            tot_ero_xy.append(ini_row)
        # Total eroded thickness for this event in maximum compaction state
        event_dict[event_ID][3] = tot_ero_xy[:]
    # Need to get erosion thicknesses into present-dat maxumum compaction state
    #*******************************************************************
    # Calculate total erosion for each layer in maximum compaction state
    # and reconstruct the layer before erosion and in its maximum
    # compaction state.
    #
    # Begin with present-day thickness, then add eroded thicknesses
    # corrected for the maximum compaction state for each event begining
    # with the most recent.
    #
    # Calculations are performed at specific x, y nodes. So for each node
    # we loop over all events.
    #*******************************************************************
    ntops = len(tops_list)
    for kk in range(ntops): # Looping from youngest to oldest
        etot_xy = []
        for i in range(nx):
            etot_row = []
            for j in range(ny):
                event_list = tops_list[kk][0] 
                erosion_list = tops_list[kk][3]
                max_burial_list = tops_list[kk][13]
                max_pd_burial_list = tops_list[kk][12]
                nerosion = len(erosion_list)
                name = tops_list[kk][6]                   
                e_thick = 0.0
                icount = 0
                # Loop over all erosion lists and 
                # calculate total erosion at this coordinate
                for nn in range(nerosion):
                    mm = nerosion - 1 - nn
                    event_ID = event_list[mm]
                    if event_ID != 0:                                                
                        e_thick_o = erosion_list[mm][i][j]
                        if e_thick_o > 0.0:
                            if iuse_max_state == 1:
                                if icount == 0:
                                    # Get the present-day maximum burial depth
                                    # this will have to be adjusted as 
                                    # we sum eroded thicknesses
                                    z2n = max_pd_burial_list[i][j]
                                # Get the maximum burial depth of eroded layer
                                z1o = max_burial_list[mm-1][i][j]
                                z2o = z1o+e_thick_o
                                # Re-compact erosion thickness 
                                # using Zn2, Z1n and Z2n
                                phi_o = tops_list[kk][9]/100.0
                                c = 1.0/(tops_list[kk][10]*1000)
                                e_thick_f = compaction.\
                                    compact_or_decompact_from_bottom(
                                                    phi_o, c, z1o, z2o, z2n)
                            else:
                                e_thick_f = e_thick_o
                            # add this eroded thickness in 
                            # maximum compaction state to the
                            # total event erosion list
                            event_dict[event_ID][3][i][j] = \
                                    event_dict[event_ID][3][i][j] + e_thick_f
                            if iuse_max_state == 1:
                                    z1n = z2n - e_thick_f
                                    # adjust maximum burial depth to 
                                    # account for additional material
                                    z2n = z1n
                            e_thick = e_thick + e_thick_f
                            icount = icount + 1    
                etot_row.append(e_thick)
            etot_xy.append(etot_row)
        tops_list[kk][8].append(etot_xy)
        name = tops_list[kk][6]
        ioutput = 0
        if ioutput == 1:
            my_file_name = "EROTOT_top_"+str(kk)+"_"+name+".dat"
            map_tools.make_output_file_ZMAP_v3(
                                        output_path, my_file_name, etot_xy, 
                                        nx, ny, dx, dy, xmin, xmax, ymin, ymax
                                    )


def make_shift_maps(
                        event_dict, tops_list, Lx, Ly, nx, ny, dx, dy, 
                        xmin, xmax, ymin, ymax, output_path
):   
    keys = event_dict.keys()
    for ii, key in enumerate(keys):        
        itop = event_dict[key][2]
        age = event_dict[key][0]        
        stype = event_dict[key][1]
        tot_event_ero = event_dict[key][3]
        len_tot_event_ero = len(tot_event_ero)        
        tot_top_ero = tops_list[itop][8]        
        shift_xy_new = []
        shift_xy_old = []
        for i in range(nx):
            shift_row = []
            for j in range(ny):
                if ii == 0:
                    s_old = 0.0
                else:
                    s_old = shift_xy_old[i][j]
                # Total erosion for particular top at a given event age
                f1 = tot_top_ero[0][i][j]
                # Why does this include the [0], why not tot_top_ero[i][j]
                # OK, it is because a single list is append to [] is where top_list is initialized in _MAIN
                # Total erosion for event (all layers)
                if len_tot_event_ero > 0:
                    f2 = tot_event_ero[i][j]
                else:
                    f2 = 0.0
                if stype == "Deposition":
                    s_new = s_old + f1
                elif stype == "Erosion":
                    s_new = s_old - f2
                elif stype == "Erosion_and_Deposition":
                    s_new = s_old + f1 - f2 # I don't understand this!
                shift_row.append(s_new)
            shift_xy_new.append(shift_row)
        shift_xy_old = shift_xy_new[:]
        my_file_name = "ShiftMap_event_"+str(key)+"_"+str(age)+"Ma"
        map_tools.make_output_file_ZMAP_v3(
                                output_path, my_file_name, shift_xy_new, 
                                nx, ny, dx, dy, xmin, xmax, ymin, ymax
                            )
