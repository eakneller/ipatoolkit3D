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
import fileIO
import numpy as np
import scipy.stats
import map_tools
from numba import jit


@jit(nopython=True)
def get_results_vec_at_ij(i,j, nruns, vec_np, scalars_nxy_np):
    for m in range(nruns):
        vec_np[m] = scalars_nxy_np[m][i][j]


def calc_maps_stats(
                    dir_path_list, target_output_dir, 
                    target_map_name, threshold,
                    output_path
):    
    nruns_list = []
    root_list = []
    sub_dir_list_all = []    
    nruns_tot = 0
    for dir_path in dir_path_list:        
        # Get information on all risk runs
        (
            nruns, output_path_root, 
            sub_dir_list, input_path
        ) = fileIO.read_riskrun_file(dir_path)        
        nruns_tot = nruns_tot + nruns
        nruns_list.append(nruns)
        root_list.append(output_path_root)
        sub_dir_list_all.append(sub_dir_list)
    ndir = len(dir_path_list)
    icount_run = 0
    for i in range(ndir):
        nruns = nruns_list[i]
        root_dir = root_list[i]
        sub_dir_list = sub_dir_list_all[i]
        nsub = len(sub_dir_list)
        for j in range(nsub):    
            input_path = (
                            root_dir 
                            + sub_dir_list[j] 
                            + "\\" + target_output_dir + "\\"
                            )
            print("i, j, input_path, target_map_name : ", 
                  input_path, target_map_name)
            AOI_np_ini = np.zeros((1,1))
            # Read Map
            (
                scalars_xy_np, nx, ny, 
                dx, dy, xmin, xmax, ymin, ymax, 
                AOI_np
            ) = map_tools.read_ZMAP(input_path, target_map_name, AOI_np_ini)
            if i == 0 and j == 0:
                # Initialize master numpy array
                scalars_nxy_np = np.zeros((nruns_tot, nx, ny))
            # Transfer to master numpy array
            scalars_nxy_np[icount_run] = scalars_xy_np
            icount_run = icount_run + 1
    # Calculate statistics
    nruns = scalars_nxy_np.shape[0]
    nx = scalars_nxy_np.shape[1]
    ny = scalars_nxy_np.shape[2]
    p10map_xy_np = np.ones((nx,ny))*-99999
    p50map_xy_np = np.ones((nx,ny))*-99999
    p90map_xy_np = np.ones((nx,ny))*-99999
    pgtmap_xy_np = np.ones((nx,ny))*-99999
    stdevmap_xy_np = np.ones((nx,ny))*-99999
    varmap_xy_np = np.ones((nx,ny))*-99999
    for i in range(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:
                vec_np = np.zeros((nruns))
                get_results_vec_at_ij(i,j, nruns, vec_np, scalars_nxy_np)
                percentile10 = np.percentile(vec_np, 10)
                percentile50 = np.percentile(vec_np, 50)
                percentile90 = np.percentile(vec_np, 90)
                p10 = percentile90
                p50 = percentile50
                p90 = percentile10
                p10map_xy_np[i][j]=p10
                p50map_xy_np[i][j]=p50
                p90map_xy_np[i][j]=p90
                pgt = np.sum(vec_np > threshold) / vec_np.size
                pgtmap_xy_np[i][j]=pgt
                stdev = np.std(vec_np)
                stdevmap_xy_np[i][j]=stdev
                var = np.var(vec_np)
                varmap_xy_np[i][j]=var               
                
    file_name = "p10_"+target_map_name
    map_tools.make_output_file_ZMAP_v4(
                            output_path, file_name, p10map_xy_np,
                            nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)
    file_name = "p50_"+target_map_name
    map_tools.make_output_file_ZMAP_v4(
                            output_path, file_name, p50map_xy_np,
                            nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)
    file_name = "p90_"+target_map_name
    map_tools.make_output_file_ZMAP_v4(
                            output_path, file_name, p90map_xy_np,
                            nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)
    file_name = "prob_gt_threshold_"+str(threshold)+"_"+target_map_name
    map_tools.make_output_file_ZMAP_v4(
                            output_path, file_name, pgtmap_xy_np,
                            nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)
    file_name = "stdev_"+target_map_name
    map_tools.make_output_file_ZMAP_v4(
                            output_path, file_name, stdevmap_xy_np,
                            nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)
    file_name = "var_"+target_map_name
    map_tools.make_output_file_ZMAP_v4(
                            output_path, file_name, varmap_xy_np,
                            nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np)
                
def random_values_from_discrete(p1, p2, p3, nruns):
    xk = np.zeros((3))
    xk[0] = -1
    xk[1] = 0
    xk[2] = 1
    pk = np.zeros((3))
    pk[0] = p1
    pk[1] = p2
    pk[2] = p3    
    custm = scipy.stats.rv_discrete(name='custm', values=(xk, pk))
    vec = custm.rvs(size=nruns)
    return vec


def random_values_from_pdf(pdf_stype, a, b, c, vmin, vmax, nruns):
    # Run# vs variance or std
    # Triangular: got it
    # Uniform
    # Guaisian
    # Beta: got it
    # Discrete: use other function
    scale_val = vmax - vmin
    loc_val = vmin
    if pdf_stype == "beta":
        vec = scipy.stats.beta.rvs(a, b, scale=scale_val, 
                                   loc=loc_val, size=nruns)
    elif pdf_stype == "triangular":
        vec = scipy.stats.triang.rvs(c, scale=scale_val, 
                                     loc=loc_val)
    elif pdf_stype == "uniform":
        vec = scipy.stats.uniform.rvs(scale=scale_val, 
                                      loc=loc_val)
    else:
        vec = scipy.stats.beta.rvs(a, b, scale=scale_val, 
                                   loc=loc_val, size=nruns)
    return vec


def build_high_low_delta_dict(
                                zeros_xy_np, risk_var_names, 
                                HL_dict, pdf_dict, discrete_dict
):    
    HL_delta_dict = {}    
    pdf_keys = list(pdf_dict.keys())
    active_var_names = list(HL_dict.keys())
    for var_name in risk_var_names:
        if var_name not in ["kinetics", "gas_frac", "polar_frac", 
                            "adsth", "inert", "temp_elastic"]:
            HL_delta_dict[var_name] = [[zeros_xy_np, "zero_map"]]
        elif var_name in ["kinetics", "gas_frac", "polar_frac"]:
            HL_delta_dict[var_name] = [[2,"2-base_model"]]
        else:
            HL_delta_dict[var_name] = [[0,"0"]]
    for active_var_name in active_var_names:
        # Get high and low case values
        delta_L = HL_dict[active_var_name][0]
        delta_H = HL_dict[active_var_name][1]
        if active_var_name in pdf_keys:
            Lelem = pdf_dict[active_var_name][4]
        elif active_var_name == "kinetics":
            Lelem = "Early"
        elif active_var_name in ["gas_frac", "polar_frac"]:
            Lelem = "minimum"
        HL_delta_dict[active_var_name].append([delta_L,str(Lelem)])
        # Loop over all keys
        for var_name in risk_var_names:
            if var_name != active_var_name:
                if var_name not in ["kinetics", "gas_frac", "polar_frac", 
                                    "adsth", "inert", "temp_elastic"]:
                    HL_delta_dict[var_name].append([zeros_xy_np,"zero_map"])
                elif var_name in ["kinetics", "gas_frac", "polar_frac"]:
                    HL_delta_dict[var_name].append([2, "2-base_model"])
                else:
                    HL_delta_dict[var_name].append([0,"0"])        
        if active_var_name in pdf_keys:
            Helem = pdf_dict[active_var_name][5]
        elif active_var_name == "kinetics":
            Helem = "Late"
        elif active_var_name in ["gas_frac", "polar_frac"]:
            Helem = "maximum"            
        HL_delta_dict[active_var_name].append([delta_H,str(Helem)])        
        # Loop over all keys
        for var_name in risk_var_names:            
            # Build high side scenarios
            if var_name != active_var_name:
                if var_name not in ["kinetics", "gas_frac", "polar_frac", 
                                    "adsth", "inert", "temp_elastic"]:
                    HL_delta_dict[var_name].append([zeros_xy_np,"zero_map"])
                elif var_name in ["kinetics", "gas_frac", "polar_frac"]:
                    HL_delta_dict[var_name].append([2,"2-base_model"])
                else:
                    HL_delta_dict[var_name].append([0,"0"])    
    nruns = len(HL_delta_dict[active_var_names[0]])    
    return HL_delta_dict, nruns


@jit(nopython=True)
def reset_AOI(delta_nxy_np, nruns, nx, ny, AOI_np):    
    for ir in range(nruns):        
        for i in range(nx):          
            for j in range(ny):                
                AOI_flag = AOI_np[i][j]
                if AOI_flag != 1:
                    delta_nxy_np[ir][i][j] = -99999


def assign_random_numbers_to_map(
                                    nx, ny, smin_xy_np, smax_xy_np, 
                                    delta_nxy_np, nruns, sname, 
                                    pdf_dict, AOI_np
):
    pdf_stype = pdf_dict[sname][0]
    a = pdf_dict[sname][1]
    b = pdf_dict[sname][2]
    c = pdf_dict[sname][3]
    trange_xy_np = smax_xy_np - smin_xy_np
    pdf_dict[sname].append([])    
    for ir in range(nruns):        
        if pdf_stype == "beta":
            rv = scipy.stats.beta.rvs(a, b, scale=1, loc=0)
        elif pdf_stype == "triangular":
            rv = scipy.stats.triang.rvs(c, scale=1, loc=0)
        elif pdf_stype == "uniform":
            rv = scipy.stats.uniform.rvs(scale=1, loc=0)
        else:
            rv = scipy.stats.beta.rvs(a, b, scale=1, loc=0)        
        delta_nxy_np[ir]= smin_xy_np + trange_xy_np*rv
        pdf_dict[sname][6].append(rv)        
    reset_AOI(delta_nxy_np, nruns, nx, ny, AOI_np)


@jit(nopython=True)
def update_scalars(nx, ny, delta_xy_tmp_np, val, AOI_np):
    for i in range(nx):    
        for j in range(ny):      
            AOI_flag = AOI_np[i][j]        
            if AOI_flag != 1:            
                delta_xy_tmp_np[i][j] = -99999        
            else:            
                delta_xy_tmp_np[i][j] = val