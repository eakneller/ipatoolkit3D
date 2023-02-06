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
import numpy as np
import map_tools
import fluid_props


def calc_charge_maps(
                    input_dir_path, wb_dir, wb_name, 
                    res_dir, res_name,
                    temp_dir, temp_name, 
                    bGOR_dir, bGOR_name, sgrav_gas,
                    secgas_name, pgas_name, poil_name, 
                    api_name, free_gas_fraction_theshold,
                    single_phase_risk_range, sGOR_at_charp, 
                    charp, exp_fac, a_gas, b_gas,
                    rho_sea, rho_brine, name_elem, sGORflag,
                    idepth_type
):
    output_dir_path = os.path.join(input_dir_path, "Charge_History_ZMaps")
    if os.path.isdir(output_dir_path) != True:
        os.mkdir(output_dir_path)
    iuse_sGOR_cor = 1
    Oil_API_avg = 30.0
    cm3_scf = 28316.85
    cm3_bbl = 158987.30
    rho_air_surface = 0.001292 # g/cm3    
    rho_gas_surface = rho_air_surface * sgrav_gas
    sgrav_oil = 141.5/(131.5 + Oil_API_avg)
    rho_oil_surface = sgrav_oil # g/cm3
    sGOR_scf_bbl_max = 1e6
    bGOR_max_g_g = (
                    sGOR_scf_bbl_max*cm3_scf/cm3_bbl*
                        (rho_gas_surface/rho_oil_surface)
                   )
    # calculate GOR_threshold_gas_risk using 100C
    TF = 9.0/5.0*100.0 + 32.0
    (
        GOR_threshold_gas_risk_scf_bbl, 
        fpress_int
    ) = fluid_props.calc_intersection_sat_history(
                            TF, Oil_API_avg,
                            sgrav_gas, rho_gas_surface, rho_oil_surface, 
                            sGOR_at_charp, charp, exp_fac,
                            sGOR_scf_bbl_max, cm3_scf, cm3_bbl, 
                            a_gas, b_gas, 1
                            )
    GOR_threshold_gas_risk_g_g = (
                                  GOR_threshold_gas_risk_scf_bbl*
                                      cm3_scf/cm3_bbl*
                                          (rho_gas_surface/rho_oil_surface)
                                 )
    GOR_ls = (
                GOR_threshold_gas_risk_g_g 
              - GOR_threshold_gas_risk_g_g*single_phase_risk_range
             )
    GOR_hs = (
                GOR_threshold_gas_risk_g_g 
              + GOR_threshold_gas_risk_g_g*single_phase_risk_range
             )
    AOI_np_ini = np.zeros((1,1))
    if sGORflag == "cumu":
        direc = "cumulative"
    else:
        direc = "incremental"
    if idepth_type == 0:
        direc_depth = "Initial"
    else:
        direc_depth = "Updated"
    # read api map
    (
        api_xy_np, nx, ny, dx, dy, 
        xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(os.path.join(bGOR_dir, "tVERTICAL_APIbulk", direc), 
                            api_name, AOI_np_ini)       
    # read top reservoir map    
    (
        res_xy_np, nx, ny, dx, dy, 
        xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(os.path.join(res_dir, direc_depth), res_name, 
                           AOI_np_ini)
    # read temp reservoir map    
    (
        temp_xy_np, nx, ny, dx, dy, 
        xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(temp_dir, temp_name, AOI_np_ini)
    # read bulk GOR (Note this is no longer used and is recalculated below) 
    (
        bGOR_xy_np, nx, ny, dx, dy, 
        xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(os.path.join(bGOR_dir, "tVERTICAL_GOR_g_g", direc), 
                            bGOR_name, AOI_np_ini)
    # read primary gas Tg 
    (
        pgasTg_xy_np, nx, ny, dx, dy, 
        xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(os.path.join(bGOR_dir, "pVERTICAL_Gas_Tg", direc), 
                            pgas_name, AOI_np_ini)
    # read secondary gas Tg 
    (
        secgasTg_xy_np, nx, ny, dx, dy, 
        xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(os.path.join(bGOR_dir, "secVERTICAL_Gas_Tg", direc) , 
                            secgas_name, AOI_np_ini)
    # read primary oil Tg 
    (
        poilTg_xy_np, nx, ny, dx, dy, 
        xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(os.path.join(bGOR_dir, "pVERTICAL_Oil_Tg", direc), 
                           poil_name, AOI_np_ini)
    # read water bottom map
    (
        wb_xy_np, nx, ny, dx, dy, 
        xmin, xmax, ymin, ymax, AOI_np
    ) = map_tools.read_ZMAP(os.path.join(wb_dir, direc_depth), wb_name, 
                           AOI_np_ini)
    iphase_xy_np = np.ones((nx, ny))*-99999.0
    #gas_risk_xy_np = np.ones((nx, ny))*-99999.0
    resoilGOR_xy_np = np.ones((nx, ny))*-99999.0
    resgasGOR_xy_np = np.ones((nx, ny))*-99999.0
    fpress_xy_np = np.ones((nx, ny))*-99999.0
    tempC_xy_np = np.ones((nx, ny))*-99999.0
    bGOR_xy_np = np.ones((nx, ny))*-99999.0
    bGOR_scf_bbl_xy_np = np.ones((nx, ny))*-99999.0    
    gasTg_xy_np = np.ones((nx, ny))*-99999.0
    oilTg_xy_np = np.ones((nx, ny))*-99999.0    
    for i in range(nx):         
        for j in range(ny):         
            AOI_flag = AOI_np[i][j]            
            if AOI_flag == 1:
                # -1 undefine, 0=understatured oil, 1 = single phase, 
                # 2 = dual phase, 3 = undersaturated gas
                iphase = -1                
                wb = wb_xy_np[i][j] # meters subsea
                dres = res_xy_np[i][j] # meters subsea
                temp = temp_xy_np[i][j] # C
                TF = 9.0/5.0*temp + 32.0                
                moil_g = poilTg_xy_np[i][j]*1e12
                mpgas_g = pgasTg_xy_np[i][j]*1e12
                msecgas_g = secgasTg_xy_np[i][j]*1e12
                mgas_g = mpgas_g + msecgas_g
                if moil_g < 0: 
                    moil_g = 0
                if mpgas_g < 0: 
                    mpgas_g = 0
                if msecgas_g < 0: 
                    msecgas_g = 0
                if mgas_g < 0: 
                    mgas_g = 0                
                gasTg_xy_np[i][j] = mgas_g/1e12
                oilTg_xy_np[i][j] = moil_g/1e12              
                if moil_g > 0:
                    bGOR = (mgas_g)/moil_g
                else:         
                    if mgas_g > 0:
                        bGOR = bGOR_max_g_g
                    else:
                        bGOR = -99999.0  
                if bGOR > bGOR_max_g_g:
                    bGOR = bGOR_max_g_g
                bGOR_xy_np[i][j] = bGOR
                bGOR_scf_bbl = (bGOR*cm3_bbl/cm3_scf
                                        *(rho_oil_surface/rho_gas_surface))
                bGOR_scf_bbl_xy_np[i][j] = bGOR_scf_bbl
                # calculate hydrostatic pressure
                fpress_psi = (
                               (
                                rho_sea*9.8*wb + rho_brine*9.8*(dres - wb)
                               )/6894.76
                             )
                fpress_xy_np[i][j] = fpress_psi
                tempC_xy_np[i][j] = temp_xy_np[i][j]
                Oil_API = api_xy_np[i][j]
                if Oil_API <= 0:
                    Oil_API = 38.0
                (   sGOR_oil_g_g, sGOR_oil_scf_bbl, 
                    sGOR_gas_g_g, sGOR_gas_scf_bbl
                ) = fluid_props.calc_saturation_gor(
                                        fpress_psi, Oil_API, sgrav_gas, 
                                        rho_gas_surface, rho_oil_surface,
                                        TF, sGOR_at_charp, charp, exp_fac, 
                                        sGOR_scf_bbl_max, cm3_scf, cm3_bbl,
                                        a_gas, b_gas, iuse_sGOR_cor
                                        )
                if bGOR <= 0:
                    iphase = -99999.0                 
                else:
                    if sGOR_oil_g_g >= sGOR_gas_g_g: # single phase below the v
                        iphase = 1.0 
                        if bGOR <= GOR_ls:                         
                            resoilGOR_xy_np[i][j] = bGOR_scf_bbl
                            resgasGOR_xy_np[i][j] = -99999.0
                        elif bGOR >= GOR_hs:
                            resoilGOR_xy_np[i][j] = -99999
                            resgasGOR_xy_np[i][j] = bGOR_scf_bbl
                        elif (bGOR <= GOR_threshold_gas_risk_g_g 
                              and bGOR > GOR_ls
                              ):
                            resoilGOR_xy_np[i][j] = bGOR_scf_bbl
                            resgasGOR_xy_np[i][j] = -99999.0
                        elif (bGOR > GOR_threshold_gas_risk_g_g 
                              and bGOR < GOR_hs
                              ):
                            resoilGOR_xy_np[i][j] = -99999
                            resgasGOR_xy_np[i][j] = bGOR_scf_bbl
                    elif bGOR <= sGOR_oil_g_g: # single phase oil
                        iphase = 0.0
                        resoilGOR_xy_np[i][j] = bGOR_scf_bbl
                        resgasGOR_xy_np[i][j] = -99999
                    elif bGOR >= sGOR_gas_g_g: # single phase gas
                        iphase = 3.0 
                        resoilGOR_xy_np[i][j] = -99999
                        resgasGOR_xy_np[i][j] = bGOR_scf_bbl
                    else: # dual phase
                        (   mDO, mDG, mL_g, mFG_g
                        ) = fluid_props.calc_liquid_and_free_gas_mass(
                                        3, moil_g, mgas_g,
                                        sGOR_oil_g_g, sGOR_gas_g_g, 
                                        )
                        iphase = 2.0 # dual phase
                        resoilGOR_xy_np[i][j] = sGOR_oil_scf_bbl
                        resgasGOR_xy_np[i][j] = sGOR_gas_scf_bbl
                iphase_xy_np[i][j] = iphase
    my_file_name = "TRAP_iphase_map" + "_" + name_elem + "_" + sGORflag
    map_tools.make_output_file_ZMAP_v4(
            output_dir_path, my_file_name, iphase_xy_np, nx, ny, dx,
            dy, xmin, xmax, ymin, ymax, AOI_np)
    my_file_name = "TRAP_oilGOR_scf_bbl" + "_" + name_elem + "_" + sGORflag
    map_tools.make_output_file_ZMAP_v4(
            output_dir_path, my_file_name, resoilGOR_xy_np, nx, ny, dx,
            dy, xmin, xmax, ymin, ymax, AOI_np)
    my_file_name = "TRAP_gasGOR_scf_bbl" + "_" + name_elem + "_" + sGORflag
    map_tools.make_output_file_ZMAP_v4(
            output_dir_path, my_file_name, resgasGOR_xy_np, nx, ny, dx,
            dy, xmin, xmax, ymin, ymax, AOI_np)
    my_file_name = "TRAP_fpress_psi" + "_" + name_elem + "_" + sGORflag
    map_tools.make_output_file_ZMAP_v4(
            output_dir_path, my_file_name, fpress_xy_np, nx, ny, dx,
            dy, xmin, xmax, ymin, ymax, AOI_np)
    my_file_name = "TRAP_tempC" + "_" + name_elem + "_" + sGORflag
    map_tools.make_output_file_ZMAP_v4(
            output_dir_path, my_file_name, tempC_xy_np, nx, ny, dx,
            dy, xmin, xmax, ymin, ymax, AOI_np)
    my_file_name = "Charge_bGOR_g_g" + "_" + name_elem + "_" + sGORflag
    map_tools.make_output_file_ZMAP_v4(
            output_dir_path, my_file_name, bGOR_xy_np, nx, ny, dx,
            dy, xmin, xmax, ymin, ymax, AOI_np)
    my_file_name = "Charge_bGOR_scf_bbl" + "_" + name_elem + "_" + sGORflag
    map_tools.make_output_file_ZMAP_v4(
            output_dir_path, my_file_name, bGOR_scf_bbl_xy_np, nx, ny, dx,
            dy, xmin, xmax, ymin, ymax, AOI_np)
    my_file_name = "Charge_Oil_Tg" + "_" + name_elem + "_" + sGORflag
    map_tools.make_output_file_ZMAP_v4(
            output_dir_path, my_file_name, oilTg_xy_np, nx, ny, dx,
            dy, xmin, xmax, ymin, ymax, AOI_np)
    my_file_name = "Charge_Gas_Tg" + "_" + name_elem + "_" + sGORflag
    map_tools.make_output_file_ZMAP_v4(
            output_dir_path, my_file_name, gasTg_xy_np, nx, ny, dx,
            dy, xmin, xmax, ymin, ymax, AOI_np)
    print("------------------------------------------------------------------")
    print("Saturation State Integer Flags")
    print("0 = Single Phase Undersaturated Oil")
    print("1 = Deep Single Phase")
    print("2 = Dual-Phase Saturated Oil and Gas")
    print("3 = Single Phase Undersaturated Gas")
    print("------------------------------------------------------------------")


def calc_charge_maps_main(
                            output_path, sGORflag, res_age_strs, 
                            res_tID, ev_IDs,
                            res_name_top, wb_names, idepth_type, 
                            sGOR_at_charp, charp, exp_fac, a_gas,
                            b_gas, rho_sea, rho_brine, sgrav_gas
):
    """ Calculate generative GOR and phase state assuming vertical migration
    
    This function calculates iphase saturation state maps using integrated 
    post-trap oil and gas charge maps produced by the calc_bulk_GOR function. 
    Trap surface pressure and temperature history maps are also exported. 
    An additional calculation performed by this function involves calculating 
    trap GOR's taking reservoir pressure, temeprature and saturation model 
    into account. Total charge oil and gas and reservoir pressure and 
    temperature history maps are also exported and bulk total integrated 
    post-trap GOR is recalculated and exported as Charge_GOR. 
    
    """
    # Used to define gas risk when mass of free gas 
    # exceeds this fraction gas factor is 1
    free_gas_fraction_theshold = 0.25
    # Used to smooth out gas risk in single phase deep domain
    single_phase_risk_range = 0.2
    print("Calculating phase maps for sGORflag = ", sGORflag)
    for ir, age_str in enumerate(res_age_strs): 
        age = float(age_str)
        name_elem = "res" + age_str
        print(">>> Working on reservoir age: ", age)
        if idepth_type == 0:
            prefix = "DEPTH_Initial"
        else:
            prefix = "DEPTH_Updated"
        # Name of water bottom map
        wb_name = (prefix + "_EV_" + ev_IDs[ir] + "_t" + ev_IDs[ir]
                   +"_AGE_" + age_str + "_TOP_" + wb_names[ir] + ".dat")    
        # Name of reservoir depth map in subsea meters
        res_name = (prefix + "_EV_" + ev_IDs[ir] + "_" + res_tID
                    + "_AGE_" + age_str + "_TOP_" + res_name_top + ".dat")
        # Name of temperature file
        temp_name = ("TEMP_C_EV_" + ev_IDs[ir] + "_" + res_tID
                     + "_AGE_" + age_str + "_TOP_" + res_name_top + ".dat")
        bGOR_name = ("tVERTICAL_GOR_g_g_post_trap" + "_res"
                     + age_str + "_" + sGORflag + ".dat")
        secgas_name = ("secVERTICAL_Gas_Tg_post_trap" + "_res"
                       + age_str + "_" + sGORflag + ".dat")
        pgas_name = ("pVERTICAL_Gas_Tg_post_trap" + "_res"
                     + age_str + "_" + sGORflag + ".dat")
        poil_name = ("pVERTICAL_Oil_Tg_post_trap" + "_res"
                     + age_str + "_" + sGORflag + ".dat")
        api_name = ("tVERTICAL_APIbulk" + "_res"
                    + age_str + "_" + sGORflag + ".dat")       
        bGOR_dir = os.path.join(output_path, "GOR_Bulk")
        # Path to depth maps files with same 
        # resolution and zmap structure as inputs
        depth_dir = os.path.join(output_path, "Depth_ZMaps")
        # path to temperature directory
        temp_dir = os.path.join(output_path, "Temperature_ZMaps")
        wb_dir = depth_dir
        res_dir = depth_dir
        calc_charge_maps(
                            output_path, wb_dir, wb_name, res_dir,
                            res_name, temp_dir, temp_name, 
                            bGOR_dir, bGOR_name, sgrav_gas,
                            secgas_name, pgas_name, poil_name, api_name,
                            free_gas_fraction_theshold, 
                            single_phase_risk_range,
                            sGOR_at_charp, charp, exp_fac, 
                            a_gas, b_gas, rho_sea,
                            rho_brine, name_elem, sGORflag, idepth_type
                            )