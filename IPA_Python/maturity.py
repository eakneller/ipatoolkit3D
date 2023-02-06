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
import math_tools
import numpy as np
import numba
from numba import jit
import map_tools
import manage_parallel
import print_funcs


def maturity_history(model, process, ioutput_main):
    if model.icalc_temp == 1:
        tt1 = time.time()
        ioutput = 1
        calculate_maturity_history(model, ioutput)    
        tt2 = time.time()
        print_funcs.print_finfo(ioutput_main, process, 
                                "Calculated maturity", tt2-tt1)

        
def calc_all_LOM_maps(model, process, ioutput_main):
    if model.icalc_LOM  == 1:
        tt1 = time.time()
        make_LOM_maps(model)
        tt2 = time.time()
        print_funcs.print_finfo(ioutput_main, process, 
                                "Calculated LOM", tt2-tt1)
        if model.itype3D == 1:
            tt1 = time.time()
            calc_time_to_LOM(model)
            tt2 = time.time()
            print_funcs.print_finfo(ioutput_main, process, 
                                    "Calculated age to LOM", tt2-tt1)
            tt1 = time.time()
            calc_depth_to_LOM(model)
            tt2 = time.time()
            print_funcs.print_finfo(ioutput_main, process, 
                                    "Calculated depth to LOM", tt2-tt1)      


def calc_depth_to_LOM(model):
    tops_list_bs = model.tops_list_bs
    event_dict_bs = model.event_dict_bs
    output_path = model.output_path
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np
    
    keys = list(event_dict_bs.keys())
    nevents_bs = len(keys)
    event_ID_list_bs = keys[:]
    event_ID_last_bs = event_ID_list_bs[nevents_bs-1]
    ntops = len(tops_list_bs)
    data_xy_np = np.zeros((ntops,2))
    depthLOM8_map = np.ones((nx,ny))*-99999.0
    depthLOM9_map = np.ones((nx,ny))*-99999.0
    depthLOM10_map = np.ones((nx,ny))*-99999.0
    depthLOM11_map = np.ones((nx,ny))*-99999.0
    depthLOM12_map = np.ones((nx,ny))*-99999.0
    # Loop over each grid node    
    for i in range(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:                
                data_xy_np = data_xy_np*0                
                LOM_max = 0.0                
                # Loop over each top
                for mm in range(ntops):                    
                    jj = ntops-1-mm                    
                    event_index = tops_list_bs[jj][14][event_ID_last_bs]                    
                    zdepth = tops_list_bs[jj][1][event_index][i][j]                    
                    LOM = tops_list_bs[jj][40][event_index][i][j]
                    if LOM > LOM_max:
                        LOM_max = LOM                    
                    data_xy_np[mm][0] = LOM
                    data_xy_np[mm][1] = zdepth                
                LOM = 8
                if LOM_max >=8:
                    depth_LOM = math_tools.linear_interp_numba(LOM, data_xy_np)
                else:
                    depth_LOM = -99999.0                            
                depthLOM8_map[i][j] = depth_LOM                    
                LOM = 9
                if LOM_max >=9:
                    depth_LOM = math_tools.linear_interp_numba(LOM, data_xy_np)
                else:
                    depth_LOM = -99999.0                            
                depthLOM9_map[i][j] = depth_LOM  
                LOM = 10
                if LOM_max >=10:
                    depth_LOM = math_tools.linear_interp_numba(LOM, data_xy_np)
                else:
                    depth_LOM = -99999.0                            
                depthLOM10_map[i][j] = depth_LOM 
                LOM = 11
                if LOM_max >=11:
                    depth_LOM = math_tools.linear_interp_numba(LOM, data_xy_np)
                else:
                    depth_LOM = -99999.0                            
                depthLOM11_map[i][j] = depth_LOM 
                LOM = 12
                if LOM_max >=12:
                    depth_LOM = math_tools.linear_interp_numba(LOM, data_xy_np)
                else:
                    depth_LOM = -99999.0                            
                depthLOM12_map[i][j] = depth_LOM 
    stype= "D2LOM_"
    sflag = "8_m"
    file_name = stype+sflag
    map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, depthLOM8_map,
                                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                        AOI_np
                                        )                   
    stype="D2LOM_"
    sflag = "9_m"
    file_name = stype+sflag
    map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, depthLOM9_map,
                                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                        AOI_np
                                        )
    stype="D2LOM_"
    sflag = "10_m"
    file_name = stype+sflag
    map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, depthLOM10_map,
                                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                        AOI_np
                                        )
    stype="D2LOM_"
    sflag = "11_m"
    file_name = stype+sflag
    map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, depthLOM11_map,
                                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                        AOI_np
                                        )   
    stype="D2LOM_"
    sflag = "12_m"
    file_name = stype+sflag
    map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, depthLOM12_map,
                                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                        AOI_np
                                        )
 

       
def calc_time_to_LOM(model):
    tops_list_bs = model.tops_list_bs
    event_dict_bs = model.event_dict_bs
    output_path = model.output_path
    nx = model.nx
    ny = model.ny
    dx = model.dx
    dy = model.dy
    xmin = model.xmin
    xmax = model.xmax
    ymin = model.ymin
    ymax = model.ymax
    AOI_np = model.AOI_np 
    keys = list(event_dict_bs.keys())
    nkeys = len(keys)
    ntops = len(tops_list_bs)    
    data_xy_np = np.zeros((nkeys,2))
    # Loop over all tops from young to old
    for mm in range(ntops):
        jj = ntops-1-mm
        # Name
        name = tops_list_bs[jj][6]        
        timeLOM8_map = np.ones((nx,ny))*-99999.0
        timeLOM9_map = np.ones((nx,ny))*-99999.0
        timeLOM10_map = np.ones((nx,ny))*-99999.0
        timeLOM11_map = np.ones((nx,ny))*-99999.0
        timeLOM12_map = np.ones((nx,ny))*-99999.0        
        # Loop over each grid node    
        for i in range(nx):            
            for j in range(ny):                
                AOI_flag = AOI_np[i][j]               
                if AOI_flag == 1:                        
                    data_xy_np = data_xy_np*0
                    LOM_max = 0.0
                    for event_ID, key in enumerate(keys):  
                        # Loop over events from old to young
                        itop_event = event_dict_bs[event_ID][2]
                        age_event = event_dict_bs[event_ID][0] 
                        if jj <= itop_event:
                            event_index = tops_list_bs[jj][14][event_ID]
                            LOM = tops_list_bs[jj][40][event_index][i][j]
                        else:
                            LOM = 0.0
                        if LOM > LOM_max:
                            LOM_max = LOM
                        data_xy_np[event_ID][0] = LOM
                        data_xy_np[event_ID][1] = age_event
                    # calculate time to LOM's
                    LOM = 8
                    if LOM_max >=8:
                        age_LOM = math_tools.linear_interp_numba(
                                                            LOM, data_xy_np)
                    else:
                        age_LOM = -99999.0                        
                    timeLOM8_map[i][j] = age_LOM
                    LOM = 9
                    if LOM_max >=9:
                        age_LOM = math_tools.linear_interp_numba(
                                                            LOM, data_xy_np)
                    else:
                        age_LOM = -99999.0
                    timeLOM9_map[i][j] = age_LOM
                    LOM = 10
                    if LOM_max >=10:
                        age_LOM = math_tools.linear_interp_numba(
                                                            LOM, data_xy_np)
                    else:
                        age_LOM = -99999.0
                    timeLOM10_map[i][j] = age_LOM
                    LOM = 11
                    if LOM_max >=11:
                        age_LOM = math_tools.linear_interp_numba(
                                                            LOM, data_xy_np)
                    else:
                        age_LOM = -99999.0
                    timeLOM11_map[i][j] = age_LOM
                    LOM = 12
                    if LOM_max >=12:
                        age_LOM = math_tools.linear_interp_numba(
                                                            LOM, data_xy_np)
                    else:
                        age_LOM = -99999.0
                    timeLOM12_map[i][j] = age_LOM
        stype="ageLOM_"
        sflag = "8_Ma"
        file_name = stype+sflag+"_TOP_"+name
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, timeLOM8_map,
                                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                        AOI_np
                                        )                   
        stype="ageLOM_"
        sflag = "9_Ma"
        file_name = stype+sflag+"_TOP_"+name
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, timeLOM9_map,
                                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                        AOI_np
                                        )
        stype="ageLOM_"
        sflag = "10_Ma"
        file_name = stype+sflag+"_TOP_"+name
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, timeLOM10_map,
                                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                        AOI_np
                                        )
        stype="ageLOM_"
        sflag = "11_Ma"
        file_name = stype+sflag+"_TOP_"+name
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, timeLOM11_map,
                                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                        AOI_np
                                        )        
        stype="ageLOM_"
        sflag = "12_Ma"
        file_name = stype+sflag+"_TOP_"+name
        map_tools.make_output_file_ZMAP_v4(
                                        output_path, file_name, timeLOM12_map,
                                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                                        AOI_np
                                        )


@jit(nopython=True, cache=True)                    
def ro_to_LOM(Ro):   
    LOM = (
            - 0.039334094215*(Ro**6)
            + 0.68017990719*(Ro**5)
            - 4.6288766166*(Ro**4)
            + 15.911525349*(Ro**3)
            - 29.386310224*(Ro**2)
            + 30.159096332*Ro
            - 2.7946967927
            )
    return LOM


def make_LOM_maps(model):
    #tops_list_bs = model.tops_list_bs
    #event_dict_bs = model.event_dict_bs
    nx = model.nx
    ny = model.ny
    AOI_np = model.AOI_np
    itype3D = model.itype3D
    inode = model.inode
    jnode = model.jnode    
    keys = list(model.event_dict_bs.keys())
    ntops = len(model.tops_list_bs)
    # Loop over all x-y locations
    for i in range(nx):
        for j in range(ny): # Rows 
            AOI_flag = AOI_np[i][j]
            if itype3D == 0:
                if i == inode and j == jnode:
                    AOI_flag = AOI_flag
                else:
                    AOI_flag = 0
            if AOI_flag == 1:
                # Loop over tops at this location
                for mm in range(ntops):
                    jj = ntops-1-mm
                    icount_steps = 0
                    for event_ID, key in enumerate(keys): 
                        # Loop over events from old to young
                        if event_ID > 0: 
                            # Skip the oldest node since 
                            # this is "basement rock"
                            itop_event = model.event_dict_bs[event_ID][2]
                            if jj <= itop_event: 
                                # We only consider tops that have been 
                                # deposited
                                (
                                    event_index
                                ) = model.tops_list_bs[jj][14][event_ID]
                                Ro = model.tops_list_bs[jj][39]\
                                                            [event_index][i][j]
                                LOM = ro_to_LOM(Ro)
                                model.tops_list_bs[jj][40]\
                                                    [event_index][i][j] = LOM                               
                                icount_steps = icount_steps + 1


def initialize_EasyRo_parameters():
    A = 1e13
    a1 = 2.334733
    a2 = 0.250621
    b1 = 3.330657
    b2 = 1.681534
    nEa = 20
    Ea_array = np.zeros((nEa))
    f_array = np.zeros((nEa))
    Ea_array[0] = 142.0
    Ea_array[1] = 151.0
    Ea_array[2] = 159.0
    Ea_array[3] = 167.0
    Ea_array[4] = 176.0
    Ea_array[5] = 184.0
    Ea_array[6] = 192.0
    Ea_array[7] = 201.0
    Ea_array[8] = 209.0
    Ea_array[9] = 218.0
    Ea_array[10] = 226.0
    Ea_array[11] = 234.0
    Ea_array[12] = 243.0
    Ea_array[13] = 251.0
    Ea_array[14] = 259.0
    Ea_array[15] = 268.0
    Ea_array[16] = 276.0
    Ea_array[17] = 285.0
    Ea_array[18] = 293.0
    Ea_array[19] = 301.0
    f_array[0] = 0.03
    f_array[1] = 0.03
    f_array[2] = 0.04
    f_array[3] = 0.04
    f_array[4] = 0.05
    f_array[5] = 0.05
    f_array[6] = 0.06
    f_array[7] = 0.04
    f_array[8] = 0.04
    f_array[9] = 0.07
    f_array[10] = 0.06
    f_array[11] = 0.06
    f_array[12] = 0.06
    f_array[13] = 0.05
    f_array[14] = 0.05
    f_array[15] = 0.04
    f_array[16] = 0.03
    f_array[17] = 0.02
    f_array[18] = 0.02
    f_array[19] = 0.01 
    return nEa, f_array, Ea_array, A, a1, a2, b1, b2


@jit(nopython=True, cache=True)
def calc_EasyRo_time_loop(
                To, event_duration, ntimes, hr_Myr, hr, 
                nEa, Ea_array, f_array, A, a1, a2, b1, b2,
                delta_l_ij_initial
):
    # Intialize arrays used for the Ro calculation for this time step
    ro_all = np.zeros(ntimes)
    F = np.zeros((ntimes))
    l_ij_p1 = np.zeros((ntimes, nEa), dtype=np.float64)
    l_ij_p2 = np.zeros((ntimes, nEa), dtype=np.float64)
    l_ij = np.zeros((ntimes, nEa), dtype=np.float64)
    delta_l_ij = np.zeros((ntimes, nEa), dtype=np.float64)
    ff = np.zeros((ntimes, nEa), dtype=np.float64)
    delta_l_ij_final = np.zeros((nEa))
    Kat0C = 273.15
    Rgas = 0.008314472
    start_age = event_duration # Ma
    end_age = 0.0 # Ma
    dt_Myr = (start_age - end_age)/float(ntimes-1)
    for i in range(ntimes):
        tMyr = float(i)*dt_Myr
        TC = To + tMyr*hr_Myr
        TK = TC + Kat0C
        for j in range(nEa):
            Ea = Ea_array[j]
            l_ij_p1[i][j] = (Ea/Rgas/TK)**2 + a1*(Ea/Rgas/TK) + a2
            l_ij_p2[i][j] = (Ea/Rgas/TK)**2 + b1*(Ea/Rgas/TK) + b2
            (
                l_ij[i][j]
            ) = TK*A*(1 - l_ij_p1[i][j]/l_ij_p2[i][j])*math.exp(-Ea/Rgas/TK)
        if i > 0:
            for j in range(nEa):
                if abs(hr) > 0.0:
                    delta_l_ij[i][j] = (
                                         delta_l_ij[i-1][j] 
                                       + (l_ij[i][j] - l_ij[i-1][j])/hr
                                       )
                else:
                    delta_l_ij[i][j] = delta_l_ij[i-1][j]
        else:
            for j in range(nEa):
                delta_l_ij[i][j] = delta_l_ij_initial[j]
        for j in range(nEa):
            delta = delta_l_ij[i][j]
            ff[i][j] = f_array[j]*(1 - math.exp(-delta))
        sumit = 0.0
        for j in range(nEa):
            sumit = sumit + ff[i][j]
        F[i] = sumit
        ro_all[i] = math.exp(-1.6 + 3.7*F[i])
    for j in range(nEa):
        delta_l_ij_final[j] =  delta_l_ij[ntimes-1][j]
    Ro_final = ro_all[ntimes - 1]
    tfinal = float(ntimes - 1)*dt_Myr
    return tfinal, Ro_final, delta_l_ij_final


@jit(nopython=True, cache=True)    
def calc_easyRo_event(
                        ntimes, To, Tf, event_duration,
                        delta_l_ij_initial, 
                        nEa, A, a1, a2, b1, b2, Ea_array, f_array
):
    sec_per_myr = 365*24*60*60*1000000
    # linear heating rate C/Myr
    hr_Myr = (Tf-To)/event_duration
    # C/sec 
    hr = (Tf-To)/(event_duration*sec_per_myr)
    (
        tfinal, Ro, delta_l_ij_final
    ) = calc_EasyRo_time_loop(
                    To, event_duration, ntimes, hr_Myr, hr, 
                    nEa, Ea_array, f_array, A, a1, a2, b1, b2,
                    delta_l_ij_initial
                    )
    return Ro, delta_l_ij_final


#@jit(nopython=True, parallel=True)
def calculate_maturity_loop_xy(
                            nx, ny, AOI_np, itype3D, inode, jnode, 
                            icount_steps, event_duration, event_ID, 
                            To_xy, Tf_xy, 
                            ntimes, delta_l_ij_final_xy_events,
                            nEa, A, a1, a2, b1, b2, Ea_array, f_array
):
    Ro_xy = np.zeros((nx,ny))
    for i in numba.prange(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if itype3D == 0:
                if i == inode and j == jnode:
                    AOI_flag = AOI_flag
                else:
                    AOI_flag = 0
            if AOI_flag == 1:
                if icount_steps > 0:
                    To = To_xy[i][j]
                else:
                    To = 0.0
                Tf = Tf_xy[i][j]
                if icount_steps == 0:
                    delta_l_ij_initial = np.zeros((nEa))
                else:
                    (
                        delta_l_ij_initial
                    ) = np.copy(delta_l_ij_final_xy_events[event_ID - 1][i][j])
                Ro, delta_l_ij_final = calc_easyRo_event(
                                    ntimes, To, Tf, event_duration,
                                    delta_l_ij_initial, 
                                    nEa, A, a1, a2, b1, b2, Ea_array, f_array
                                    )
                delta_l_ij_final_xy_events[event_ID][i][j] = (
                                                    np.copy(delta_l_ij_final)
                                                    )
                Ro_xy[i][j] = Ro
    return Ro_xy
    
def calculate_maturity_history(model, ioutput):
    nx = model.nx
    ny = model.ny
    AOI_np = model.AOI_np
    itype3D = model.itype3D
    inode = model.inode
    jnode = model.jnode
    ntimes = 41 # Number of time steps per event used to calculate Ro
    (
         nEa, f_array, Ea_array, 
         A, a1, a2, b1, b2
    ) = initialize_EasyRo_parameters()
    keys = list(model.event_dict_bs.keys())
    ntops = len(model.tops_list_bs)
    nevents = ntops
    itype3D = model.itype3D
    calculate_maturity_loop_xy_active = manage_parallel.manage_parallel(
                                           calculate_maturity_loop_xy, itype3D)
    # Calculate maturity history for each top
    for mm in range(ntops):
        jj = ntops - 1 - mm
        # Initialize array for delta_l_ij for all events and x-y locations
        # associated with this top
        delta_l_ij_final_xy_events = np.zeros((nevents, nx, ny, nEa))
        icount_steps = 0
        for event_ID, key in enumerate(keys): 
            # Loop over events from old to young
            if event_ID > 0: 
                # Skip the oldest node since 
                # this is "basement rock"
                itop_event = model.event_dict_bs[event_ID][2]
                age_event = model.event_dict_bs[event_ID][0]
                event_ID_prev = event_ID - 1
                age_event_prev = model.event_dict_bs[event_ID_prev][0]
                if jj <= itop_event:
                    event_duration = age_event_prev - age_event
                    # We only consider tops that 
                    # have been deposited
                    event_index = model.tops_list_bs[jj][14][event_ID]                    
                    if icount_steps > 0:
                        (
                            event_index_prev
                        ) = model.tops_list_bs[jj][14][event_ID_prev]
                        (
                            To_xy
                        ) = np.copy(
                                model.tops_list_bs[jj][38][event_index_prev]
                                )
                    else:
                        To_xy = np.zeros((nx, ny))
                    Tf_xy = np.copy(model.tops_list_bs[jj][38][event_index])
                    Ro_xy = calculate_maturity_loop_xy_active(
                                    nx, ny, AOI_np, itype3D, inode, jnode, 
                                    icount_steps, event_duration, event_ID, 
                                    To_xy, Tf_xy, 
                                    ntimes, delta_l_ij_final_xy_events,
                                    nEa, A, a1, a2, b1, b2, Ea_array, f_array
                                    )
                    model.tops_list_bs[jj][39][event_index] = np.copy(Ro_xy)
                    icount_steps = icount_steps + 1                            
                                
                                