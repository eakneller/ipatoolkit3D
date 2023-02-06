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
import os
import math_tools
import numpy as np
from numba import jit
import fileIO
import map_tools


def get_yield_map_info(yield_dir_path, top_names):    
    topIDs = []
    event_starts = []
    age_strs_all = []   
    for src_name in top_names:    
        age_strs_tmp = []
        event_start = 99999             
        for file in os.listdir(yield_dir_path):
            file = file.replace('.zip','')
            if file.endswith(".dat"):
                file_check = file.replace('.dat','')
                check_name_list = file_check.split("_")
                # reassemble name if it includes "_"
                nlist = len(check_name_list)
                if nlist > 10:
                    nparts = nlist - 9 
                    for mm in range(nparts):
                        if mm == 0:
                            updated_name = (
                                       check_name_list[nlist - 1 - nparts + 1])
                        else:
                            updated_name = (
                                updated_name + "_" 
                                + check_name_list[nlist -1 - nparts + 1 + mm]
                                )
                    #print("nparts, updated_name : ", nparts, updated_name)
                    check_name_list.append(updated_name)
                #print("src_name: check_name_list ", src_name, check_name_list)
                if src_name in check_name_list:
                    #print("Found it ", src_name)
                    data = file.split("_")
                    ev = int(data[4])
                    tname = data[5]
                    nc = len(tname)
                    tid = int(tname[1:nc])
                    age_str = data[7]
                    if age_str not in age_strs_tmp:
                        age_strs_tmp.append(age_str)
                    if tid not in topIDs:
                        topIDs.append(tid)
                    age = float(age_str)
                    if ev < event_start:
                        event_start = ev
        
        age_strs = []
        iorder = []
        nstrs = len(age_strs_tmp)
        # Search for the biggest index    
        for j in range(nstrs):
            age_max = -99999
            imax = -99999
            for i in range(nstrs):    
                age = float(age_strs_tmp[i])
                if age > age_max and i not in iorder:
                    age_max = age
                    imax = i
            # Add index to list 
            iorder.append(imax)
        for i in range(nstrs):
            mm = iorder[i]
            age_strs.append(age_strs_tmp[mm])        
        event_starts.append(event_start)
        age_strs_all.append(age_strs)
    return age_strs_all, topIDs, event_starts


def read_source_maps(
                        map_dir_path, age_strs, topID, event_start, 
                        prefix, units, top_name, fext, AOI_np_ini
):   
    map_dict = {}
    for i, age_str in enumerate(age_strs):        
        event = event_start + i
        age = float(age_str)        
        map_file_name = (
                            prefix + "_" + units
                          + "_EV_" + str(event)
                          + "_t" + str(topID)
                          + "_AGE_" + age_str 
                          + "_TOP_" + top_name + fext
                         )
        (
            scalars_xy, nx, ny, dx, dy, 
            xmin, xmax, ymin, ymax, AOI_np
        ) = map_tools.read_ZMAP(map_dir_path, map_file_name, AOI_np_ini)
        map_dict[age] = scalars_xy
    return (
            map_dict, nx, ny, dx, dy, 
            xmin, ymin, xmax, ymax, AOI_np, 
            scalars_xy
        )


@jit(nopython=True, cache=True)
def sum_post_trap_APIs_in_polys_loop(
                                        APIbulk_xy_np, pp, pkey, pcoors_np, 
                                        nx, ny, xmin, ymin, dx, dy, AOI_np
):
    sum_API = 0.0
    sum_nAPI = 0.0    
    # Loop over all map coordinates
    for i in range(nx):         
        for j in range(ny): # Rows            
            AOI_flag = AOI_np[i][j]            
            if AOI_flag == 1:
                mx = xmin + dx*float(i)
                my = ymin + dy*float(j)
                API = APIbulk_xy_np[i][j]
                inpoly = math_tools.inside_outside_poly(pcoors_np, mx, my)
                if inpoly == True and API > 0:
                    sum_API = sum_API + API
                    sum_nAPI = sum_nAPI + 1
    
    return sum_API, sum_nAPI


def sum_post_trap_APIs_in_polys(
                                output_dir_path, pp, poly_dict_list,
                                poly_APIsum_dict_list, poly_APInsum_dict_list, 
                                APIbulk_xy_np, nx, ny,
                                xmin, ymin, dx, dy, AOI_np
):
    poly_keys = list(poly_dict_list[pp].keys())
    for pkey in poly_keys:
        pcoors = poly_dict_list[pp][pkey]
        pcoors_np = np.asarray(pcoors)        
        sum_API, nsum_API = sum_post_trap_APIs_in_polys_loop(
                                                    APIbulk_xy_np, pp, pkey,
                                                    pcoors_np, nx, ny, 
                                                    xmin, ymin, dx, dy, AOI_np
                                                )        
        poly_APIsum_dict_list[pp][pkey] = (
                                            poly_APIsum_dict_list[pp][pkey] 
                                            + sum_API
                                        )
        poly_APInsum_dict_list[pp][pkey] = (
                                            poly_APInsum_dict_list[pp][pkey] 
                                            + nsum_API
                                        )


@jit(nopython=True, cache=True)
def sum_post_trap_masses_in_polys_loop(
                                        trap_age, pp, pkey, mkey, age, 
                                        pcoors_np, poly_age_start, 
                                        poly_age_end, nx, ny, xmin, ymin, 
                                        dx, dy, scalars_PO_xy_np,
                                        scalars_PFG_xy_np, scalars_PDG_xy_np, 
                                        scalars_SFG_xy_np, AOI_np, res_age, 
                                        itypeGOR
):
    sum_mPO = 0.0
    sum_mPDG = 0.0
    sum_mPFG = 0.0
    sum_mSFG = 0.0
    for i in range(nx):
        for j in range(ny):
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:
                mx = xmin + dx*float(i)
                my = ymin + dy*float(j)
                mPO = scalars_PO_xy_np[i][j]
                mPFG = scalars_PFG_xy_np[i][j]
                mPDG = scalars_PDG_xy_np[i][j]
                mSFG = scalars_SFG_xy_np[i][j]
                if mPO < 0: mPO = 0
                if mPDG < 0: mPDG = 0
                if mPFG < 0: mPFG = 0
                if mSFG < 0: mSFG = 0
                inpoly = math_tools.inside_outside_poly(pcoors_np, mx, my)
                if inpoly == True:
                    ifind = 0
                    if itypeGOR == 0: # cummulative
                        if age < trap_age and age >= res_age:
                            ifind = 1
                    else: # incremental
                        if age < trap_age and age == res_age:
                            ifind = 1
                    if ifind == 1:
                        if poly_age_end == 0.0:
                            poly_age_end = -1.0                            
                        if age < poly_age_start and age >= poly_age_end:
                            sum_mPO = sum_mPO + mPO 
                            sum_mPDG = sum_mPDG + mPDG
                            sum_mPFG = sum_mPFG + mPFG
                            sum_mSFG = sum_mSFG + mSFG
    return sum_mPO, sum_mPDG, sum_mPFG, sum_mSFG


def sum_post_trap_masses_in_polys(
                                    top_name, output_dir_path, pp, trap_age,
                                    poly_age_start, poly_age_end, 
                                    poly_PO_dict_list, poly_PDG_dict_list,
                                    poly_PFG_dict_list, poly_SFG_dict_list, 
                                    poly_tOil_dict_list,
                                    poly_tGas_dict_list, poly_sGas_dict_list, 
                                    nx, ny, dx, dy, xmin, ymin,
                                    AOI_np, age_strs, poly_dict_list,
                                    map_dict_PO, map_dict_PFG, map_dict_PDG,
                                    map_dict_SFG, res_age, itypeGOR
):
    poly_keys = list(poly_dict_list[pp].keys())
    map_keys = list(map_dict_PO.keys())
    for pkey in poly_keys:
        pcoors = poly_dict_list[pp][pkey]
        pcoors_np = np.asarray(pcoors)
        for im, mkey in enumerate(map_keys):
            scalars_PO_xy = map_dict_PO[mkey]
            scalars_PO_xy_np = np.asarray(scalars_PO_xy)            
            scalars_PFG_xy = map_dict_PFG[mkey]
            scalars_PFG_xy_np = np.asarray(scalars_PFG_xy)             
            scalars_PDG_xy = map_dict_PDG[mkey]
            scalars_PDG_xy_np = np.asarray(scalars_PDG_xy)         
            scalars_SFG_xy = map_dict_SFG[mkey]
            scalars_SFG_xy_np = np.asarray(scalars_SFG_xy)     
            age = float(age_strs[im])
            (
                sum_mPO, sum_mPDG, 
                sum_mPFG, sum_mSFG
            ) = sum_post_trap_masses_in_polys_loop(
                                                trap_age, pp, pkey, mkey, age,
                                                pcoors_np, poly_age_start, 
                                                poly_age_end, nx, ny, xmin, 
                                                ymin, dx, dy,
                                                scalars_PO_xy_np, 
                                                scalars_PFG_xy_np, 
                                                scalars_PDG_xy_np,
                                                scalars_SFG_xy_np, AOI_np, 
                                                res_age, itypeGOR
                                            )            
            poly_PO_dict_list[pp][pkey] = (
                                            poly_PO_dict_list[pp][pkey] 
                                            + sum_mPO)
            poly_PDG_dict_list[pp][pkey] = (
                                            poly_PDG_dict_list[pp][pkey] 
                                            + sum_mPDG)
            poly_PFG_dict_list[pp][pkey] = (
                                            poly_PFG_dict_list[pp][pkey] 
                                            + sum_mPFG)
            poly_SFG_dict_list[pp][pkey] = (
                                            poly_SFG_dict_list[pp][pkey] 
                                            + sum_mSFG)      
            poly_tOil_dict_list[pp][pkey] = poly_PO_dict_list[pp][pkey]
            poly_tGas_dict_list[pp][pkey] = (
                                            poly_PDG_dict_list[pp][pkey] 
                                            + poly_PFG_dict_list[pp][pkey])
            poly_sGas_dict_list[pp][pkey] = poly_SFG_dict_list[pp][pkey]
            
            
@jit(nopython=True, cache=True)
def sum_post_trap_masses_at_grid_nodes_loop(
                                            nx, ny, AOI_np, scalars_PO_xy_np,
                                            scalars_PFG_xy_np, 
                                            scalars_PDG_xy_np, 
                                            scalars_SFG_xy_np, tPO_xy_np,
                                            tPFG_xy_np, tPDG_xy_np, tSFG_xy_np
):
    for i in range(nx):
        for j in range(ny): # Rows
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:
                mPO = scalars_PO_xy_np[i][j]
                mPFG = scalars_PFG_xy_np[i][j]
                mPDG = scalars_PDG_xy_np[i][j]
                mSFG = scalars_SFG_xy_np[i][j]
                if mPO < 0: mPO = 0
                if mPDG < 0: mPDG = 0
                if mPFG < 0: mPFG = 0
                if mSFG < 0: mSFG = 0
                mPOi = tPO_xy_np[i][j]
                mPFGi = tPDG_xy_np[i][j]
                mPDGi = tPFG_xy_np[i][j]
                mSFGi = tSFG_xy_np[i][j]
                tPO_xy_np[i][j]=mPOi+mPO
                tPFG_xy_np[i][j]=mPFGi+mPFG
                tPDG_xy_np[i][j]=mPDGi+mPDG
                tSFG_xy_np[i][j]=mSFGi+mSFG


def sum_post_trap_masses_at_grid_nodes(
        trap_age, nx, ny, dx, dy, xmin, ymin,
        AOI_np, age_strs, map_dict_PO, map_dict_PFG, 
        map_dict_PDG, map_dict_SFG, tPO_xy_np, tPDG_xy_np, 
        tPFG_xy_np, tSFG_xy_np, res_age, itypeGOR
):
    idebug = 0
    if idebug == 1: 
        print("--> Summing post trap masses at grid nodes, trap_age: ", 
              trap_age)
    map_keys = list(map_dict_PO.keys())
    # Loop over all ages for this source rock
    for im, mkey in enumerate(map_keys):
        scalars_PO_xy = map_dict_PO[mkey]
        scalars_PO_xy_np = np.asarray(scalars_PO_xy)
        scalars_PFG_xy = map_dict_PFG[mkey]
        scalars_PFG_xy_np = np.asarray(scalars_PFG_xy)         
        scalars_PDG_xy = map_dict_PDG[mkey]
        scalars_PDG_xy_np = np.asarray(scalars_PDG_xy)       
        scalars_SFG_xy = map_dict_SFG[mkey]
        scalars_SFG_xy_np = np.asarray(scalars_SFG_xy)
        age = float(age_strs[im])
        ifind = 0
        if itypeGOR == 0: # cummulative
            if age < trap_age and age >= res_age:
                ifind = 1
        else: # incrementa;
            if age < trap_age and age == res_age:
                ifind = 1
        if ifind == 1:         
            sum_post_trap_masses_at_grid_nodes_loop(
                                        nx, ny, AOI_np, scalars_PO_xy_np,
                                        scalars_PFG_xy_np, scalars_PDG_xy_np, 
                                        scalars_SFG_xy_np, tPO_xy_np,
                                        tPFG_xy_np, tPDG_xy_np, tSFG_xy_np
                                        )


@jit(nopython=True, cache=True)
def calc_POmassfrac_at_grid_nodes_loop(
                                        nx, ny, AOI_np, scalars_PO_xy_np,
                                        tPO_xy_np, tPOsrc_xy_np, 
                                        POmassfrac_xy_np
):   
    # Loop over all map coordinates
    for i in range(nx):
        for j in range(ny): # Rows  
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1: # Only perform calc if within AOI
                mPO = scalars_PO_xy_np[i][j]                
                mPOi = tPOsrc_xy_np[i][j]                
                tPOsrc_xy_np[i][j]=mPOi+mPO
    for i in range(nx):         
        for j in range(ny): # Rows  
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1: # Only perform calc if within AOI                
                if tPO_xy_np[i][j] > 0:
                    POmassfrac_xy_np[i][j] = tPOsrc_xy_np[i][j]/tPO_xy_np[i][j]
                else:
                    POmassfrac_xy_np[i][j] = 0

                
def calc_POmassfrac_at_grid_nodes(
                                    trap_age, nx, ny, dx, dy, xmin, ymin,
                                    AOI_np, age_strs, map_dict_PO, tPO_xy_np, 
                                    POmassfrac_xy_np, res_age, itypeGOR
):
    map_keys = list(map_dict_PO.keys())    
    tPOsrc_xy_np = np.zeros((nx,ny))    
    # Loop over all ages for this source rock
    for im, mkey in enumerate(map_keys):        
        scalars_PO_xy = map_dict_PO[mkey]
        scalars_PO_xy_np = np.asarray(scalars_PO_xy)         
        age = float(age_strs[im])
        ifind = 0
        if itypeGOR == 0: # cummulative
            if age < trap_age and age >= res_age:
                ifind = 1
        else: # incrementa;
            if age < trap_age and age == res_age:
                ifind = 1                
        if ifind == 1:
            calc_POmassfrac_at_grid_nodes_loop(
                                                nx, ny, AOI_np, 
                                                scalars_PO_xy_np,
                                                tPO_xy_np, tPOsrc_xy_np, 
                                                POmassfrac_xy_np
                                                )


@jit(nopython=True, cache=True)
def calc_avgTR_at_grid_nodes_loop(
                                    nx, ny, AOI_np, scalars_TR_xy_np,
                                    TRmin_xy_np, TRmax_xy_np
):
    # Loop over all map coordinates
    for i in range(nx):
        for j in range(ny): # Rows
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1: # Only perform calc if within AOI
                TRc = scalars_TR_xy_np[i][j]
                if TRc < TRmin_xy_np[i][j]:
                    TRmin_xy_np[i][j] = TRc
                if TRc > TRmax_xy_np[i][j]:
                    TRmax_xy_np[i][j] = TRc


@jit(nopython=True, cache=True)
def calc_bulk_api(
                    nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                    AOI_np, APIbulk_xy_np,
                    APIbulk_denom_xy_np, con_fac_xy_np, 
                    rho_gs, cm3_scf, cm3_bbl
):
    APIf = 22 #38
    # Loop over all map coordinates
    for i in range(nx):
        for j in range(ny): # Rows
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1: # Only perform calc if within AOI
                if APIbulk_xy_np[i][j] > 0:
                    if APIbulk_denom_xy_np[i][j] > 0:
                        oil_api = APIbulk_xy_np[i][j]/APIbulk_denom_xy_np[i][j]
                        if oil_api <=0:
                            oil_api = APIf
                        APIbulk_xy_np[i][j] = oil_api
                        sgo = 141.5/(131.5+oil_api)
                        rho_os = sgo*1e6 # g/m3
                        con_fac_tmp = rho_os/rho_gs*cm3_bbl/cm3_scf
                        con_fac_xy_np[i][j] = con_fac_tmp
                    else:
                        oil_api = APIf
                        APIbulk_xy_np[i][j] = APIf
                        APIbulk_xy_np[i][j] = oil_api
                        sgo = 141.5/(131.5+oil_api)
                        rho_os = sgo*1e6 # g/m3
                        con_fac_tmp = rho_os/rho_gs*cm3_bbl/cm3_scf
                        con_fac_xy_np[i][j] = con_fac_tmp
                else:
                    oil_api = APIf
                    APIbulk_xy_np[i][j] = APIf
                    APIbulk_xy_np[i][j] = oil_api
                    sgo = 141.5/(131.5+oil_api)
                    rho_os = sgo*1e6 # g/m3
                    con_fac_tmp = rho_os/rho_gs*cm3_bbl/cm3_scf
                    con_fac_xy_np[i][j] = con_fac_tmp

                   
@jit(nopython=True, cache=True)
def get_mid_point_TR_and_API(
                                nx, ny, AOI_np, 
                                TRmin_xy_np, TRmax_xy_np,
                                TRavg_xy_np, MF_xy_np, 
                                APIavg_xy_np, APIbulk_xy_np, 
                                APIbulk_denom_xy_np
):   
    APIo = 22
    APIf = 42
    TRo = 0.1
    TRf = 0.9
    dAPI_dTR = (APIf-APIo)/(TRf-TRo)
    # Loop over all map coordinates
    for i in range(nx):
        for j in range(ny): # Rows            
            AOI_flag = AOI_np[i][j]           
            if AOI_flag == 1: # Only perform calc if within AOI 
                TRavg = (TRmin_xy_np[i][j] + TRmax_xy_np[i][j])/2.0
                TRavg_xy_np[i][j] = TRavg
                if TRavg <= TRo:
                    APIavg = APIo
                elif TRavg >= TRf:
                    APIavg = APIf
                else:
                    APIavg = APIo + dAPI_dTR*(TRavg-TRo)
                APIavg_xy_np[i][j] = APIavg
                APIbulk_ini = APIbulk_xy_np[i][j]
                APIbulk_xy_np[i][j] = APIbulk_ini+APIavg*MF_xy_np[i][j]
                APIbulk_denom_ini = APIbulk_denom_xy_np[i][j]
                APIbulk_denom_xy_np[i][j] = APIbulk_denom_ini + MF_xy_np[i][j]
            else:
                APIbulk_xy_np[i][j] = -99999.0
                
                                         
def calc_avgTR_at_grid_nodes(
                                trap_age, nx, ny, dx, dy, xmin, ymin,
                                AOI_np, age_strs, map_dict_TR, MF_xy_np, 
                                TRavg_xy_np, APIavg_xy_np,
                                APIbulk_xy_np, APIbulk_denom_xy_np, 
                                res_age, itypeGOR,
                                output_dir_path, xmax, ymax
):
    map_keys = list(map_dict_TR.keys())
    TRmin_xy_np = np.ones((nx,ny))*1e32
    TRmax_xy_np = np.ones((nx,ny))*-1e32
    # Loop over all ages for this source rock
    for im, mkey in enumerate(map_keys):
        scalars_TR_xy = map_dict_TR[mkey]
        scalars_TR_xy_np = np.asarray(scalars_TR_xy)
        age = float(age_strs[im])
        ifind = 0
        if itypeGOR == 0: # cummulative
            if age < trap_age and age >= res_age:
                ifind = 1
        else: # incrementa;
            if age < trap_age and age == res_age:
                ifind = 1  
        if ifind == 1:
            calc_avgTR_at_grid_nodes_loop(
                                    nx, ny, AOI_np, scalars_TR_xy_np,
                                    TRmin_xy_np, TRmax_xy_np
                                    )
    get_mid_point_TR_and_API(
                                nx, ny, AOI_np, TRmin_xy_np, TRmax_xy_np,
                                TRavg_xy_np, MF_xy_np, APIavg_xy_np, 
                                APIbulk_xy_np, APIbulk_denom_xy_np
                                )

    
@jit(nopython=True, cache=True)
def make_GOR_grids_from_polys_loop(
                                    age, pp, pkey, nx, ny, dx, dy, 
                                    xmin, ymin, AOI_np,
                                    pcoors_np, poly_pGOR_np, 
                                    poly_tGOR_np, poly_APIsum_np, 
                                    poly_APInsum_np, pGOR_xy_np, 
                                    tGOR_xy_np, tOil_xy_np, 
                                    tGas_xy_np, sGas_xy_np, poly_tOil_np,
                                    poly_tGas_np, poly_sGas_np, API_xy_np
):  
    for i in range(nx):
        for j in range(ny): # Rows
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1: # Only perform calc if within AOI
                mx = xmin + dx*float(i)
                my = ymin + dy*float(j)
                inpoly = math_tools.inside_outside_poly(pcoors_np, mx, my)
                # If we are in the polygon then assign the polygons values to map node
                if inpoly == True:
                    pGOR_xy_np[i][j] = poly_pGOR_np[pkey]
                    tGOR_xy_np[i][j] = poly_tGOR_np[pkey]
                    tOil_xy_np[i][j] = poly_tOil_np[pkey]
                    tGas_xy_np[i][j] = poly_tGas_np[pkey]
                    sGas_xy_np[i][j] = poly_sGas_np[pkey]
                    if poly_APIsum_np[pkey] > 0:
                        if poly_APInsum_np[pkey] > 0:
                            API_xy_np[i][j] = (
                                    poly_APIsum_np[pkey]/poly_APInsum_np[pkey])
                        else:
                            API_xy_np[i][j] = -99999.0
                    else:
                        API_xy_np[i][j] = -99999.0
                        
                        
def make_GOR_grids_from_polys(
                        age, pp, nx, ny, dx, dy, xmin, ymin, AOI_np,
                        poly_dict, poly_pGOR, poly_tGOR, 
                        poly_tOil, poly_tGas, poly_sGas,
                        poly_APIsum, poly_APInsum, API_xy_np, 
                        pGOR_xy_np, tGOR_xy_np,
                        tOil_xy_np, tGas_xy_np, sGas_xy_np
):    
    poly_keys = list(poly_dict.keys())
    tkeys = list(poly_tGOR.keys())
    nkeys = len(tkeys)
    poly_tGOR_np = np.zeros(nkeys)
    for i in range(nkeys):
        poly_tGOR_np[i]=poly_tGOR[i]
    tkeys = list(poly_pGOR.keys())
    nkeys = len(tkeys)
    poly_pGOR_np = np.zeros(nkeys)
    for i in range(nkeys):
        poly_pGOR_np[i]=poly_pGOR[i]
    tkeys = list(poly_tOil.keys())
    nkeys = len(tkeys)
    poly_tOil_np = np.zeros(nkeys)
    for i in range(nkeys):
        poly_tOil_np[i]=poly_tOil[i]
    tkeys = list(poly_tGas.keys())
    nkeys = len(tkeys)
    poly_tGas_np = np.zeros(nkeys)
    for i in range(nkeys):
        poly_tGas_np[i]=poly_tGas[i]
    tkeys = list(poly_sGas.keys())
    nkeys = len(tkeys)
    poly_sGas_np = np.zeros(nkeys)
    for i in range(nkeys):
        poly_sGas_np[i]=poly_sGas[i]
    tkeys = list(poly_APIsum.keys())
    nkeys = len(tkeys)
    poly_APIsum_np = np.zeros(nkeys)
    for i in range(nkeys):
        poly_APIsum_np[i]=poly_APIsum[i]
    tkeys = list(poly_APInsum.keys())
    nkeys = len(tkeys)
    poly_APInsum_np = np.zeros(nkeys)
    for i in range(nkeys):
        poly_APInsum_np[i]=poly_APInsum[i]    
    nkeys = len(poly_keys)
    # Loop over each polygon
    for pkey in range(nkeys): #poly_keys:        
        pcoors = poly_dict[pkey]
        pcoors_np = np.asarray(pcoors)
        pkey = int(pkey)
        # Then loop over all map coordinates and check if 
        # coodinates are within the polygon
        make_GOR_grids_from_polys_loop(
                                        age, pp, pkey, nx, ny, dx, dy, 
                                        xmin, ymin, AOI_np,
                                        pcoors_np, poly_pGOR_np, 
                                        poly_tGOR_np, poly_APIsum_np, 
                                        poly_APInsum_np,
                                        pGOR_xy_np, tGOR_xy_np, 
                                        tOil_xy_np, tGas_xy_np, sGas_xy_np,
                                        poly_tOil_np, poly_tGas_np, 
                                        poly_sGas_np, API_xy_np
                                        )

        
@jit(nopython=True, cache=True)
def calc_integrated_apex_GOR(
                                GOR_apex_xy_np, pOil_apex_xy_np, 
                                totGas_apex_xy_np,
                                APIbulk_apex_xy_np, APInsum_apex_xy_np, 
                                GOR_max, nx, ny, AOI_np
):
    for i in range(nx):
        for j in range(ny): # Rows            
            AOI_flag = AOI_np[i][j]            
            if AOI_flag == 1: # Only perform calc if within AOI                
                mO = pOil_apex_xy_np[i][j]
                mG = totGas_apex_xy_np[i][j]
                if mO > 0:
                    GOR = mG/mO
                else:
                    if mG > 0:
                        GOR = GOR_max
                    else:
                        GOR = 0                
                if GOR > GOR_max:
                    GOR = GOR_max                
                GOR_apex_xy_np[i][j] = GOR                
                if APIbulk_apex_xy_np[i][j] > 0:
                    if APInsum_apex_xy_np[i][j] > 0:
                        (
                            APIbulk_apex_xy_np[i][j]
                        ) = APIbulk_apex_xy_np[i][j]/APInsum_apex_xy_np[i][j]
                    else:
                        APIbulk_apex_xy_np[i][j] = -99999.0
                else:
                    APIbulk_apex_xy_np[i][j] = -99999.0

    
@jit(nopython=True, cache=True)
def calc_total_apex_masses(
                            nx, ny, AOI_np, tOil_xy_np, 
                            tGas_xy_np, sGas_xy_np,
                            API_xy_np, pOil_apex_xy_np, 
                            pGas_apex_xy_np, sGas_apex_xy_np, 
                            totGas_apex_xy_np,
                            APIbulk_apex_xy_np, APInsum_apex_xy_np
):
    for i in range(nx):     
        for j in range(ny): # Rows    
            AOI_flag = AOI_np[i][j]           
            if AOI_flag == 1: # Only perform calc if within AOI
                pOil_apex_xy_np[i][j] = pOil_apex_xy_np[i][j] + tOil_xy_np[i][j]
                pGas_apex_xy_np[i][j] = pGas_apex_xy_np[i][j] + tGas_xy_np[i][j]
                sGas_apex_xy_np[i][j] = sGas_apex_xy_np[i][j] + sGas_xy_np[i][j]
                totGas_apex_xy_np[i][j] = (
                                            totGas_apex_xy_np[i][j] 
                                            + sGas_xy_np[i][j] 
                                            + tGas_xy_np[i][j]
                                          )
                if API_xy_np[i][j] > 0:
                    APIbulk_apex_xy_np[i][j] = (
                                                APIbulk_apex_xy_np[i][j] 
                                                + API_xy_np[i][j]
                                               )
                    APInsum_apex_xy_np[i][j] = APInsum_apex_xy_np[i][j] + 1
                else:
                    APIbulk_apex_xy_np[i][j] = -99999.0
                
def calc_apexGOR_polys(
                        age, pp, output_dir_path, nx, ny, dx, dy, 
                        xmin, xmax, ymin,
                        ymax, AOI_np, poly_dict_list, 
                        pname, poly_pGOR_dict_list,
                        poly_tGOR_dict_list, poly_PO_dict_list, 
                        poly_PDG_dict_list,
                        poly_PFG_dict_list, poly_SFG_dict_list, 
                        poly_tOil_dict_list,
                        poly_tGas_dict_list, poly_sGas_dict_list, 
                        poly_APIsum_dict_list,
                        poly_APInsum_dict_list, pGOR_xy_np, tGOR_xy_np,
                        pOil_apex_xy_np, pGas_apex_xy_np,
                        sGas_apex_xy_np, totGas_apex_xy_np,
                        con_fac, GOR_max, APIbulk_apex_xy_np, 
                        APInsum_apex_xy_np, res_age_str
):
    tOil_xy_np = np.zeros((nx,ny))
    tGas_xy_np = np.zeros((nx,ny))
    sGas_xy_np = np.zeros((nx,ny))
    API_xy_np = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            pGOR_xy_np[i][j]=0
            tGOR_xy_np[i][j]=0
    poly_dict = poly_dict_list[pp]
    poly_keys = list(poly_dict.keys())
    for pkey in poly_keys:
        mPO = poly_PO_dict_list[pp][pkey]
        mPDG = poly_PDG_dict_list[pp][pkey]
        mPFG = poly_PFG_dict_list[pp][pkey]
        mSFG = poly_SFG_dict_list[pp][pkey]
        if mPO < 0: mPO = 0
        if mPDG < 0: mPDG = 0
        if mPFG < 0: mPFG = 0
        if mSFG < 0: mSFG = 0
        if mPO > 0:
            pGOR = (mPFG + mPDG)/mPO
            tGOR = (mPFG + mPDG + mSFG)/mPO
        else:
            if mPFG + mPDG > 0:
                pGOR = GOR_max
            else:
                pGOR = 0            
            if mPFG + mPDG + mSFG > 0:
                tGOR = GOR_max
            else:
                tGOR = 0
        if pGOR > GOR_max:
            pGOR = GOR_max
        if tGOR > GOR_max:
            tGOR = GOR_max
        poly_pGOR_dict_list[pp][pkey] = pGOR
        poly_tGOR_dict_list[pp][pkey] = tGOR
    make_GOR_grids_from_polys(
                                age, pp, nx, ny, dx, dy, xmin, ymin, 
                                AOI_np, poly_dict,
                                poly_pGOR_dict_list[pp], 
                                poly_tGOR_dict_list[pp], 
                                poly_tOil_dict_list[pp],
                                poly_tGas_dict_list[pp], 
                                poly_sGas_dict_list[pp], 
                                poly_APIsum_dict_list[pp],
                                poly_APInsum_dict_list[pp], 
                                API_xy_np, pGOR_xy_np, tGOR_xy_np,
                                tOil_xy_np, tGas_xy_np, sGas_xy_np
                                )
        
    calc_total_apex_masses(
                            nx, ny, AOI_np, tOil_xy_np, 
                            tGas_xy_np, sGas_xy_np,
                            API_xy_np, pOil_apex_xy_np, 
                            pGas_apex_xy_np, sGas_apex_xy_np,
                            totGas_apex_xy_np,
                            APIbulk_apex_xy_np, 
                            APInsum_apex_xy_np
                            )

    
@jit(nopython=True)
def calc_GOR_nodes_loop(
                        nx, ny, dx, dy, AOI_np, pGOR_xy_np, tGOR_xy_np,
                        tPO_xy_np, tPDG_xy_np, tPFG_xy_np, tSFG_xy_np, 
                        con_fac, GOR_max, tOil_xy_np,
                        tGas_xy_np, sGas_xy_np, pGas_xy_np
):
    # Initialize arrays
    for i in range(nx):
        for j in range(ny):
            pGOR_xy_np[i][j] = 0
            tGOR_xy_np[i][j] = 0
    for i in range(nx):
        for j in range(ny):
            mPO = tPO_xy_np[i][j]
            mPDG = tPDG_xy_np[i][j]
            mPFG = tPFG_xy_np[i][j]
            mSFG = tSFG_xy_np[i][j]
            if mPO < 0: mPO = 0
            if mPDG < 0: mPDG = 0
            if mPFG < 0: mPFG = 0
            if mSFG < 0: mSFG = 0
            if mPO > 0:
                pGOR = (mPFG + mPDG)/mPO
                tGOR = (mPFG + mPDG + mSFG)/mPO
            else:
                if mPFG + mPDG > 0:
                    pGOR = GOR_max
                else:
                    pGOR = -99999.0 
                if mPFG + mPDG + mSFG > 0:
                    tGOR = GOR_max
                else:
                    tGOR = -99999.0
            if pGOR > GOR_max:
                pGOR = GOR_max
            if tGOR > GOR_max:
                tGOR = GOR_max
            pGOR_xy_np[i][j] = pGOR
            tGOR_xy_np[i][j] = tGOR
            tOil_xy_np[i][j] = mPO
            tGas_xy_np[i][j] = mPDG+mPFG+mSFG
            sGas_xy_np[i][j] = mSFG
            pGas_xy_np[i][j] = mPDG+mPFG


def quick_convert_GOR(
                        GOR_xy_np, GORnew_xy_np, con_fac_xy_np, 
                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, 
                        AOI_np
):    
    for i in range(nx):
        for j in range(ny): # Rows
            AOI_flag = AOI_np[i][j]
            if AOI_flag == 1:
                if GOR_xy_np[i][j] > 0:
                    GORnew_xy_np[i][j] = GOR_xy_np[i][j]*con_fac_xy_np[i][j] 
                else:
                    GORnew_xy_np[i][j] = -99999
            else:
                GORnew_xy_np[i][j] = -99999


def calc_GOR_nodes(
                    output_dir_path, nx, ny, dx, dy, 
                    xmin, xmax, ymin, ymax, AOI_np, 
                    pGOR_xy_np, tGOR_xy_np, tPO_xy_np, 
                    tPDG_xy_np, tPFG_xy_np,
                    tSFG_xy_np, con_fac, GOR_max, 
                    con_fac_xy_np, res_age_str, sGORflag
):
    tOil_xy_np = np.zeros((nx,ny))
    tGas_xy_np = np.zeros((nx,ny))
    pGas_xy_np = np.zeros((nx,ny))
    sGas_xy_np = np.zeros((nx,ny))
    calc_GOR_nodes_loop(
                        nx, ny, dx, dy, AOI_np, pGOR_xy_np, tGOR_xy_np,
                        tPO_xy_np, tPDG_xy_np, tPFG_xy_np, tSFG_xy_np, 
                        con_fac, GOR_max, tOil_xy_np,
                        tGas_xy_np, sGas_xy_np, pGas_xy_np
                    )
    my_file_name = ("tVERTICAL_GOR_g_g_post_trap"
                    +"_res"+res_age_str+"_"+sGORflag)
    map_tools.make_output_file_ZMAP_v4(
                                        output_dir_path, my_file_name, 
                                        tGOR_xy_np, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, AOI_np
                                    )
    GORnew_xy_np = np.zeros((nx,ny))
    quick_convert_GOR(
                        tGOR_xy_np, GORnew_xy_np, con_fac_xy_np, 
                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np
                    )
    my_file_name = ("tVERTICAL_GOR_scf_bbl_post_trap"
                    + "_res" + res_age_str + "_" + sGORflag)
    map_tools.make_output_file_ZMAP_v4(
                                        output_dir_path, my_file_name, 
                                        GORnew_xy_np, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, AOI_np
                                    )
    GORnew_xy_np = np.zeros((nx,ny))
    quick_convert_GOR(
                        pGOR_xy_np, GORnew_xy_np, con_fac_xy_np, 
                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np
                    )
    my_file_name = ("pVERTICAL_GOR_scf_bbl_post_trap"
                    + "_res" + res_age_str + "_" + sGORflag)
    map_tools.make_output_file_ZMAP_v4(
                                        output_dir_path, my_file_name, 
                                        GORnew_xy_np, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, AOI_np
                                    )
    my_file_name = ("pVERTICAL_GOR_g_g_post_trap"
                    + "_res" + res_age_str + "_" + sGORflag)
    map_tools.make_output_file_ZMAP_v4(
                                        output_dir_path, my_file_name, 
                                        pGOR_xy_np, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, AOI_np
                                    )
    my_file_name = ("pVERTICAL_Oil_Tg_post_trap"
                    + "_res" + res_age_str + "_" + sGORflag)
    map_tools.make_output_file_ZMAP_v4(
                                        output_dir_path, my_file_name, 
                                        tOil_xy_np, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, AOI_np
                                    )
    my_file_name = ("pVERTICAL_Gas_Tg_post_trap"
                    + "_res" + res_age_str + "_" + sGORflag)
    map_tools.make_output_file_ZMAP_v4(
                                        output_dir_path, my_file_name, 
                                        pGas_xy_np, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, AOI_np
                                    )
    my_file_name = ("secVERTICAL_Gas_Tg_post_trap"
                    + "_res" + res_age_str + "_" + sGORflag)
    map_tools.make_output_file_ZMAP_v4(
                                        output_dir_path, my_file_name, 
                                        sGas_xy_np, nx, ny, dx, dy, 
                                        xmin, xmax, ymin, ymax, AOI_np
                                    )


def calc_bulk_gor(
                    trap_age, oil_api, sgg, yield_dir_path, 
                    tr_dir_path, poly_dir_path, output_dir_path, 
                    poly_file_name_list, poly_age_start_list,
                    poly_age_end_list, src_top_names, 
                    res_ages, itypeGOR
):
    if itypeGOR == 0:
        sGORflag = "cumu"
    else:
        sGORflag = "incr"
    GOR_max = 8.5 # g/g
    rho_a = 1292.0 # Density of air at surface g/m3
    (
        age_strs_all, topIDs, event_starts
    ) = get_yield_map_info(os.path.join(yield_dir_path, "iYield_PO_Tg"), 
                          src_top_names)
    npoly_sets = len(poly_file_name_list)
    sgo = 141.5/(131.5+oil_api)
    rho_os = sgo*1e6 # g/m3
    rho_gs = rho_a*sgg # g/m3
    cm3_scf = 28316.846711688
    cm3_bbl = 158987.2956
    con_fac = rho_os/rho_gs*cm3_bbl/cm3_scf
    # Repeat this calculation for each reservoir age
    print("Calculating bulk GOR maps with sGORflag = ", sGORflag)
    for rr, res_age in enumerate(res_ages):
        print(">> Working on reservoir age: ", res_age)
        res_age_str = str(res_age)
        #******************************************
        # Initialize polygon dictionaries and lists
        #******************************************
        if npoly_sets > 0:
            poly_dict_list = []
            for i in range(npoly_sets):
                poly_file_path = os.path.join(poly_dir_path, 
                                              poly_file_name_list[i])
                poly_dict = fileIO.read_polys_simple(poly_file_path)
                poly_dict_list.append(poly_dict)
            poly_PO_dict_list = []
            poly_PDG_dict_list = []
            poly_PFG_dict_list = []
            poly_SFG_dict_list = []
            poly_pGOR_dict_list = []
            poly_tGOR_dict_list = []
            poly_tOil_dict_list = []
            poly_tGas_dict_list = []
            poly_sGas_dict_list = []
            poly_APIsum_dict_list = []
            poly_APInsum_dict_list = []
            for i in range(npoly_sets):
                # Initialize mass dictionaries for poly set 1
                poly_PO_dict = {}
                poly_PDG_dict = {}
                poly_PFG_dict = {}
                poly_SFG_dict = {}
                poly_pGOR_dict = {}
                poly_tGOR_dict = {}
                poly_tOil_dict = {}
                poly_tGas_dict = {}
                poly_sGas_dict = {}
                poly_APIsum_dict = {}
                poly_APInsum_dict = {}
                poly_keys = list(poly_dict_list[i].keys())
                for key in poly_keys:    
                    poly_PO_dict[key] = 0
                    poly_PDG_dict[key] = 0
                    poly_PFG_dict[key] = 0
                    poly_SFG_dict[key] = 0
                    poly_tGOR_dict[key] = 0
                    poly_pGOR_dict[key] = 0
                    poly_tOil_dict[key] = 0
                    poly_tGas_dict[key] = 0
                    poly_sGas_dict[key] = 0
                    poly_APIsum_dict[key] = 0
                    poly_APInsum_dict[key] = 0
                poly_PO_dict_list.append(poly_PO_dict)
                poly_PDG_dict_list.append(poly_PDG_dict)
                poly_PFG_dict_list.append(poly_PFG_dict)
                poly_SFG_dict_list.append(poly_SFG_dict)
                poly_pGOR_dict_list.append(poly_pGOR_dict)
                poly_tGOR_dict_list.append(poly_tGOR_dict)
                poly_tOil_dict_list.append(poly_tOil_dict)
                poly_tGas_dict_list.append(poly_tGas_dict)
                poly_sGas_dict_list.append(poly_sGas_dict)
                poly_APIsum_dict_list.append(poly_APIsum_dict)
                poly_APInsum_dict_list.append(poly_APInsum_dict)           
        # Inititalize AOI numpy array
        AOI_np_ini = np.zeros((1,1))        
        nsrc = len(src_top_names)        
        map_dict_TR_all = {}
        map_dict_PO_all = {}
        map_dict_PFG_all = {}
        map_dict_PDG_all = {}
        map_dict_SFG_all = {}        
        map_dict_MASSFRAC = {}        
        map_dict_TRAVG = {}
        map_dict_APIAVG = {}        
        for i in range(nsrc):            
            top_name = src_top_names[i]
            topID = topIDs[i]
            age_strs = age_strs_all[i]
            event_start = event_starts[i]
            # Define map dictionary for transformation ratio
            prefix = "TR"
            units="frac"
            fext = ".dat"
            (
                map_dict_TR, nx, ny, dx, dy, 
                xmin, ymin, xmax, ymax, AOI_np, 
                scalars_xy
             ) = read_source_maps(
                                     tr_dir_path, age_strs, topID, event_start,
                                     prefix, units, top_name, fext, AOI_np_ini
                                    )
            map_dict_TR_all[i] = map_dict_TR.copy()
            # Define map dictionary for primary oil
            prefix = "iYield_PO"
            units="Tg"
            fext = ".dat"
            (
                map_dict_PO, nx, ny, dx, dy, 
                xmin, ymin, xmax, ymax, AOI_np, 
                scalars_xy
            ) = read_source_maps(
                                 os.path.join(yield_dir_path, "iYield_PO_Tg"), 
                                 age_strs, topID, 
                                 event_start, prefix, units, top_name, 
                                 fext, AOI_np_ini
                                )
            map_dict_PO_all[i] = map_dict_PO.copy()
            # Define map dictionary for primary free gas
            prefix = "iYield_PFG"
            units="Tg"
            fext = ".dat"
            (
                map_dict_PFG, nx, ny, dx, dy, 
                xmin, ymin, xmax, ymax, AOI_np, 
                scalars_xy
             ) = read_source_maps(
                                 os.path.join(yield_dir_path, "iYield_PFG_Tg"), 
                                 age_strs, topID, 
                                 event_start, prefix, units, top_name, 
                                 fext, AOI_np_ini
                                )
            map_dict_PFG_all[i] = map_dict_PFG.copy()
            # Define map dictionary for primary dissolved gas
            prefix = "iYield_PDG"
            units="Tg"
            fext = ".dat"
            (
                map_dict_PDG, nx, ny, dx, dy, 
                xmin, ymin, xmax, ymax, AOI_np, 
                scalars_xy
            ) = read_source_maps(
                                os.path.join(yield_dir_path, "iYield_PDG_Tg") , 
                                age_strs, topID, 
                                event_start, prefix, units, top_name, 
                                fext, AOI_np_ini
                            )
            map_dict_PDG_all[i] = map_dict_PDG.copy()
            # Define map dictionary for secondary free gas
            prefix = "iYield_SFG"
            units="Tg"
            fext = ".dat"
            (
                map_dict_SFG, nx, ny, dx, dy, 
                xmin, ymin, xmax, ymax, AOI_np, 
                scalars_xy
            ) = read_source_maps(
                                os.path.join(yield_dir_path, "iYield_SFG_Tg"), 
                                age_strs, topID, 
                                event_start, prefix, units, top_name, 
                                fext, AOI_np_ini
                            )
            map_dict_SFG_all[i] = map_dict_SFG.copy()
        #**********************************************************************
        # Calculate total mass sums for vertical migration at each node
        #**********************************************************************
        for i in range(nsrc):
            top_name = src_top_names[i]
            topID = topIDs[i]
            age_strs = age_strs_all[i]
            event_start = event_starts[i]
            map_dict_PO = map_dict_PO_all[i]
            map_dict_PFG = map_dict_PFG_all[i]
            map_dict_PDG = map_dict_PDG_all[i]
            map_dict_SFG = map_dict_SFG_all[i]
            # Initialize new calculation grids
            if i == 0:        
                pGOR_xy_np = np.zeros((nx,ny))
                tGOR_xy_np = np.zeros((nx,ny))
                tPO_xy_np = np.zeros((nx,ny))
                tPDG_xy_np = np.zeros((nx,ny))
                tPFG_xy_np = np.zeros((nx,ny))
                tSFG_xy_np = np.zeros((nx,ny))
            # Calculate sums for vertical migration
            sum_post_trap_masses_at_grid_nodes(
                                                trap_age, nx, ny, dx, dy, 
                                                xmin, ymin, AOI_np, age_strs, 
                                                map_dict_PO, map_dict_PFG, 
                                                map_dict_PDG, map_dict_SFG, 
                                                tPO_xy_np, tPDG_xy_np, 
                                                tPFG_xy_np, tSFG_xy_np, 
                                                res_age, itypeGOR
                                            )
        #**********************************************************************
        # Calculate vertical post-trap mass fractions for each source
        #**********************************************************************
        for i in range(nsrc):
            top_name = src_top_names[i]
            topID = topIDs[i]
            age_strs = age_strs_all[i]
            event_start = event_starts[i]
            map_dict_PO = map_dict_PO_all[i]
            map_dict_PFG = map_dict_PFG_all[i]
            map_dict_PDG = map_dict_PDG_all[i]
            map_dict_SFG = map_dict_SFG_all[i]
            # Initialize new calculation grids      
            POmassfrac_xy_np = np.zeros((nx,ny))
            # Calculate sums for vertical migration
            calc_POmassfrac_at_grid_nodes(
                                            trap_age, nx, ny, dx, dy, 
                                            xmin, ymin, AOI_np, age_strs, 
                                            map_dict_PO, tPO_xy_np, 
                                            POmassfrac_xy_np, res_age, 
                                            itypeGOR
                                        )
            map_dict_MASSFRAC[i] = np.copy(POmassfrac_xy_np)
            my_file_name = (
                            "tVERTICAL_MassFrac_PO_"
                            + top_name +"_res" + res_age_str + "_" + sGORflag)
            map_tools.make_output_file_ZMAP_v4(
                                                output_dir_path, my_file_name, 
                                                map_dict_MASSFRAC[i], nx, ny, 
                                                dx, dy, xmin, xmax, ymin, 
                                                ymax, AOI_np
                                            )
        #**********************************************************************
        # Calculate average vertical TR and API for each source at each x-y
        #**********************************************************************
        APIbulk_xy_np = np.zeros((nx,ny))
        APIbulk_denom_xy_np = np.zeros((nx,ny))
        con_fac_xy_np = np.zeros((nx,ny))
        for i in range(nsrc):
            top_name = src_top_names[i]
            topID = topIDs[i]
            age_strs = age_strs_all[i]
            event_start = event_starts[i]
            map_dict_TR = map_dict_TR_all[i]
            MF_xy_np = map_dict_MASSFRAC[i]
            TRavg_xy_np = np.zeros((nx,ny))
            APIavg_xy_np = np.zeros((nx,ny))
            calc_avgTR_at_grid_nodes(
                                        trap_age, nx, ny, dx, dy, xmin, ymin, 
                                        AOI_np, age_strs, map_dict_TR, 
                                        MF_xy_np, TRavg_xy_np, APIavg_xy_np,
                                        APIbulk_xy_np, APIbulk_denom_xy_np, 
                                        res_age, itypeGOR, output_dir_path, 
                                        xmax, ymax
                                        )
            map_dict_TRAVG[i] = np.copy(TRavg_xy_np)
            map_dict_APIAVG[i] = np.copy(APIavg_xy_np)
            my_file_name = ("tVERTICAL_TRavg_"+top_name
                            +"_res"+res_age_str+"_"+sGORflag)
            map_tools.make_output_file_ZMAP_v4(
                                            output_dir_path, my_file_name,
                                            map_dict_TRAVG[i], nx, ny, dx, dy, 
                                            xmin, xmax, ymin, ymax, AOI_np
                                            )
            my_file_name = ("tVERTICAL_APIavg_"+top_name
                            +"_res"+res_age_str+"_"+sGORflag)
            map_tools.make_output_file_ZMAP_v4(
                                            output_dir_path, my_file_name,
                                            map_dict_APIAVG[i], nx, ny, dx, dy, 
                                            xmin, xmax, ymin, ymax, AOI_np
                                            )
        calc_bulk_api(
                        nx, ny, dx, dy, xmin, xmax, ymin, ymax, AOI_np,
                        APIbulk_xy_np, APIbulk_denom_xy_np, con_fac_xy_np, 
                        rho_gs, cm3_scf, cm3_bbl
                        )
        my_file_name = "tVERTICAL_APIbulk"+"_res"+res_age_str+"_"+sGORflag
        map_tools.make_output_file_ZMAP_v4(
                                            output_dir_path, my_file_name,
                                            APIbulk_xy_np, nx, ny, dx, dy, 
                                            xmin, xmax, ymin, ymax, AOI_np
                                            )
        #**********************************************************************
        # Calculate and output vertical GOR
        #**********************************************************************
        calc_GOR_nodes(
                        output_dir_path, nx, ny, dx, dy, 
                        xmin, xmax, ymin, ymax, AOI_np, 
                        pGOR_xy_np, tGOR_xy_np, tPO_xy_np,
                        tPDG_xy_np, tPFG_xy_np,
                        tSFG_xy_np, con_fac, GOR_max, 
                        con_fac_xy_np, 
                        res_age_str, sGORflag
                        )
        #**********************************************************************
        # Calculate mass sums for evolving polygons
        #**********************************************************************
        for i in range(nsrc):
            top_name = src_top_names[i]
            topID = topIDs[i]
            age_strs = age_strs_all[i]
            event_start = event_starts[i]
            map_dict_PO = map_dict_PO_all[i]
            map_dict_PFG = map_dict_PFG_all[i]
            map_dict_PDG = map_dict_PDG_all[i]
            map_dict_SFG = map_dict_SFG_all[i]
            if npoly_sets > 0:
                # Calculate sums for apex GOR
                for pp in range(npoly_sets):
                    poly_age_start=poly_age_start_list[pp]
                    poly_age_end=poly_age_end_list[pp]
                    sum_post_trap_masses_in_polys(
                                                top_name, output_dir_path, pp,
                                                trap_age, poly_age_start, 
                                                poly_age_end,
                                                poly_PO_dict_list, 
                                                poly_PDG_dict_list, 
                                                poly_PFG_dict_list,
                                                poly_SFG_dict_list, 
                                                poly_tOil_dict_list, 
                                                poly_tGas_dict_list,
                                                poly_sGas_dict_list, 
                                                nx, ny, dx, dy, xmin, ymin,
                                                AOI_np, age_strs,
                                                poly_dict_list, map_dict_PO, 
                                                map_dict_PFG, map_dict_PDG,
                                                map_dict_SFG, res_age, 
                                                itypeGOR
                                            )
        #**********************************************************************
        # Calculate sum of API's in polys and number of nodes
        #**********************************************************************
        if npoly_sets > 0:
            # Calculate sums for apex GOR
            for pp in range(npoly_sets):
                sum_post_trap_APIs_in_polys(
                                            output_dir_path, pp,
                                            poly_dict_list,
                                            poly_APIsum_dict_list, 
                                            poly_APInsum_dict_list, 
                                            APIbulk_xy_np,
                                            nx, ny, xmin, ymin, dx, dy, AOI_np
                                        )
        #**********************************************************************
        # Calculate and output APEX GOR
        #**********************************************************************  
        if npoly_sets > 0:
            pOil_apex_xy_np = np.zeros((nx,ny))
            pGas_apex_xy_np = np.zeros((nx,ny))
            sGas_apex_xy_np = np.zeros((nx,ny))
            totGas_apex_xy_np = np.zeros((nx,ny))
            GOR_apex_xy_np = np.zeros((nx,ny))
            APIbulk_apex_xy_np = np.zeros((nx,ny))
            APInsum_apex_xy_np = np.zeros((nx,ny))
            for pp in range(npoly_sets):
                age=poly_age_end_list[pp]
                pname="poly"+str(pp)+"_"+str(age)+"Ma"
                calc_apexGOR_polys(
                                    age, pp, output_dir_path, nx, ny, dx, dy, 
                                    xmin, xmax, ymin, ymax, AOI_np, 
                                    poly_dict_list, pname, poly_pGOR_dict_list,
                                    poly_tGOR_dict_list, poly_PO_dict_list, 
                                    poly_PDG_dict_list,
                                    poly_PFG_dict_list, poly_SFG_dict_list, 
                                    poly_tOil_dict_list,
                                    poly_tGas_dict_list, poly_sGas_dict_list, 
                                    poly_APIsum_dict_list,
                                    poly_APInsum_dict_list, pGOR_xy_np, 
                                    tGOR_xy_np,
                                    pOil_apex_xy_np, pGas_apex_xy_np, 
                                    sGas_apex_xy_np, totGas_apex_xy_np,
                                    con_fac, GOR_max, APIbulk_apex_xy_np, 
                                    APInsum_apex_xy_np, res_age_str
                                )
            calc_integrated_apex_GOR(
                                        GOR_apex_xy_np, pOil_apex_xy_np, 
                                        totGas_apex_xy_np,
                                        APIbulk_apex_xy_np, APInsum_apex_xy_np, 
                                        GOR_max, nx, ny, AOI_np
                                    )
            my_file_name = ("MIGPOLY_API_post_trap"
                            +"_res"+res_age_str+"_"+sGORflag)
            map_tools.make_output_file_ZMAP_v4(
                                                output_dir_path, my_file_name,
                                                APIbulk_apex_xy_np, nx, ny, 
                                                dx, dy, xmin, xmax, ymin, ymax, 
                                                AOI_np
                                            )        
            my_file_name = ("MIGPOLY_GOR_g_g_post_trap"
                            +"_res"+res_age_str+"_"+sGORflag)
            map_tools.make_output_file_ZMAP_v4(
                                                output_dir_path, my_file_name,
                                                GOR_apex_xy_np, nx, ny, dx, dy, 
                                                xmin, xmax, ymin, ymax, AOI_np
                                            )
            my_file_name = ("MIGPOLY_GOR_scf_bbl_post_trap"
                            +"_res"+res_age_str+"_"+sGORflag)
            map_tools.make_output_file_ZMAP_v4(
                                                output_dir_path, my_file_name,
                                                GOR_apex_xy_np*con_fac, nx, ny, 
                                                dx, dy, xmin, xmax, ymin, ymax, 
                                                AOI_np
                                            )
            my_file_name = ("MIGPOLY_PrimaryGas_Tg_post_trap"
                            +"_res"+res_age_str+"_"+sGORflag)
            map_tools.make_output_file_ZMAP_v4(
                                                output_dir_path, my_file_name,
                                                pGas_apex_xy_np, nx, ny, dx, 
                                                dy, xmin, xmax, ymin, ymax, 
                                                AOI_np
                                            )
            my_file_name = ("MIGPOLY_PrimaryOil_Tg_post_trap"+"_res"
                            +res_age_str+"_"+sGORflag)
            map_tools.make_output_file_ZMAP_v4(
                                                output_dir_path, my_file_name,
                                                pOil_apex_xy_np, nx, ny, dx, 
                                                dy, xmin, xmax, ymin, ymax, 
                                                AOI_np
                                            )
            my_file_name = ("MIGPOLY_SecondaryGas_Tg_post_trap"
                            +"_res"+res_age_str+"_"+sGORflag)
            map_tools.make_output_file_ZMAP_v4(
                                                output_dir_path, my_file_name,
                                                sGas_apex_xy_np, nx, ny, dx, 
                                                dy, xmin, xmax, ymin, ymax, 
                                                AOI_np
                                            )
            my_file_name = ("MIGPOLY_TotalGas_Tg_post_trap"+"_res"
                            +res_age_str+"_"+sGORflag)
            map_tools.make_output_file_ZMAP_v4(
                                                output_dir_path, my_file_name,
                                                totGas_apex_xy_np, nx, ny, dx, 
                                                dy, xmin, xmax, ymin, ymax, 
                                                AOI_np
                                            )
        if npoly_sets < 1:
            print("!!! No polygon sets were found so APEX "
                  "GOR calculations were skipped !!!")