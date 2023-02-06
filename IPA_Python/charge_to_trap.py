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
import os
import glob
import fileIO
import Bulk_GOR
import charge_history
import trap_charge_history


def move_files(pf_list, output_path, target_direc, ioutput_main):
    for pf in pf_list:
        file_path_list = glob.glob(
                                    os.path.join(output_path, pf + "*")
                                  )
        for file_path1 in file_path_list:
            file_name = os.path.basename(file_path1)
            if "_incr" in file_name:
                dir_path = os.path.join(target_direc, "incremental")
                if os.path.isdir(dir_path) != True:
                    os.mkdir(dir_path)
                file_path2 = os.path.join(dir_path, file_name)
            elif "_cumu" in file_name:
                dir_path = os.path.join(target_direc, "cumulative")
                if os.path.isdir(dir_path) != True:
                    os.mkdir(dir_path)
                file_path2 = os.path.join(dir_path, file_name)
            else:
                dir_path = target_direc
                file_path2 = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path2) == True:
                try:
                    os.remove(file_path2)
                except:
                    if ioutput_main == 1: 
                        print("Unable to delete old file : ", 
                              file_path2)
            if os.path.isfile(file_path1) == True:
                try:
                    os.rename(file_path1, file_path2)        
                except:
                    if ioutput_main == 1: 
                        print("Unable to move file: "
                              "file_name, file_path1 : ", 
                              file_name, file_path1)

                                      
def charge_to_trap(model, riskmodel, ioutput_main, process,
                   OILAPI_list, GASGRAV_list):
    input_path = model.input_path
    output_path = model.output_path
    itype3D = model.itype3D
    src_top_names = model.src_top_names
    imass_gen = model.imass_gen
    trap_age = model.trap_age
    poly_file_name_list = model.poly_file_name_list
    poly_age_start_list = model.poly_age_start_list
    poly_age_end_list = model.poly_age_end_list
    res_ages = model.res_ages
    ioutput_burial = model.ioutput_burial
    icalc_temp = model.icalc_temp
    ioutput_TEMP = model.ioutput_TEMP
    iupdate_PWD = model.iupdate_PWD
    res_age_strs = model.res_age_strs
    res_tID = model.res_tID
    res_ev_IDs = model.res_ev_IDs
    res_name_top = model.res_name_top
    res_wb_names = model.res_wb_names  
    src_index_list_bulk_gor = riskmodel.src_index_list_bulk_gor  
    if itype3D == 1:
        if imass_gen == 2:
            nsrc = len(src_top_names)
            if nsrc > 0:                       
                tt1 = time.time()
                nvals = len(OILAPI_list)
                vsum = 0.0
                for val in OILAPI_list:
                    vsum = vsum + val
                if nvals > 0:
                    oil_api = vsum/float(nvals)
                else:
                    oil_api = 30
                    print("!!! Did not find any Oil API's !!!")
                nvals = len(GASGRAV_list)
                vsum = 0.0
                for val in GASGRAV_list:
                    vsum = vsum + val
                if nvals > 0:
                    sgg = vsum/float(nvals)
                else:
                    sgg = 0.75
                    print("!!! Did not find any gas gravities !!!")
                print(">> Oil API and gas gravity used for Bulk GOR "
                      "conversion to surface conditions: ", oil_api, sgg)
                src_top_names_final = []
                for isrc, srcname in enumerate(src_top_names):
                    if src_index_list_bulk_gor[0] == -1:
                        src_top_names_final.append(srcname)
                    elif isrc in src_index_list_bulk_gor:
                        src_top_names_final.append(srcname)                        
                if ioutput_main == 1: 
                    print(">>> Final source top names : ", src_top_names_final)
                if len(src_top_names_final) > 0:
                    yield_dir_path = os.path.join(output_path, 
                                                  "SRC_Yield_Incremental")
                    tr_dir_path = os.path.join(output_path,
                                               "Transformation_Ratio_ZMaps")                    
                    itypeGOR = 0
                    Bulk_GOR.calc_bulk_gor(
                                    trap_age, oil_api, sgg, yield_dir_path, 
                                    tr_dir_path, input_path, output_path, 
                                    poly_file_name_list, poly_age_start_list, 
                                    poly_age_end_list, src_top_names_final, 
                                    res_ages, itypeGOR
                                    )
                    itypeGOR = 1
                    Bulk_GOR.calc_bulk_gor(
                                    trap_age, oil_api, sgg, yield_dir_path, 
                                    tr_dir_path, input_path, output_path, 
                                    poly_file_name_list, poly_age_start_list, 
                                    poly_age_end_list, src_top_names_final, 
                                    res_ages, itypeGOR
                                    )                                                 
                    new_direc = "GOR_Bulk"
                    target_direc = os.path.join(output_path, new_direc)                    
                    if os.path.isdir(target_direc) != True:
                        os.mkdir(target_direc) 
                    pf_list_all = [
                                   "secVERTICAL_Gas_Tg", 
                                   "pVERTICAL_Gas_Tg",
                                   "pVERTICAL_Oil_Tg",
                                   "pVERTICAL_GOR_g_g",
                                   "pVERTICAL_GOR_scf_bbl",
                                   "tVERTICAL_APIavg",
                                   "tVERTICAL_APIbulk",
                                   "tVERTICAL_GOR_g_g",
                                   "tVERTICAL_GOR_scf_bbl",
                                   "tVERTICAL_MassFrac",
                                   "tVERTICAL_TRavg",
                                   "MIGPOLY"
                                   ]
                    for pf in pf_list_all:
                        new_direc = os.path.join("GOR_Bulk", pf)
                        target_direc = os.path.join(output_path, new_direc)                    
                        if os.path.isdir(target_direc) != True:
                            os.mkdir(target_direc) 
                        move_files([pf], output_path, 
                                   target_direc, ioutput_main)
                    tt2 = time.time()
                    print ("Finished calculating bulk GOR: cpu(s) : ", tt2-tt1)
                    print ("Mem (MB) : ",process.memory_info().rss/1e6)
                    if (ioutput_burial == 1 and icalc_temp == 1 
                        and ioutput_TEMP == 2):
                        # Read input file
                        input_file_path = os.path.join(input_path, 
                                                       "ipa_traps.csv")
                        (
                            ntraps, traps_list, 
                            traps_dict, 
                            traps_output_dict
                        ) = fileIO.read_traps_csv(input_file_path)
                        #  0: Oil API gravity
                        OIL_API = traps_list[0]
                        #  1: Gas specific gravity (surface w.r.t. air)
                        sgrav_gas = traps_list[1]
                        #  2: Oil density factor
                        oil_den_mod_fac = traps_list[2]
                        #  3: Gas density factor
                        gas_den_mod_fac = traps_list[3]
                        #  4: Gas density change (kg/m3)
                        delta_rho_gas_kg_m3 = traps_list[4]
                        #  5: Sea Water Density kg/m3
                        rho_sea = traps_list[5]
                        #  6: Brine Density kg/m3
                        rho_brine = traps_list[6]
                        #  7: Gas Saturation GOR Parameter A
                        a_gas = traps_list[7]
                        #  8: Gas Saturation GOR Parameter B
                        b_gas = traps_list[8]
                        #  9:Saturation GOR correction at 
                        # characteristic pressure  (scf/bbl)
                        sGOR_at_charp = traps_list[9]
                        # 10: Characteristic Pressure (psi)
                        charp = traps_list[10]
                        # 11: Exponential factor used to 
                        # produce a better match to EOS models
                        exp_fac = traps_list[11]
                        # 12: GOR limit (scf/bbl)
                        sGOR_scf_bbl_max = traps_list[12]
                        # 13: Cracking On/Off
                        # 14: Biodegradation On/Off
                        # 15: r_max: maximum biodegradation rate g/m2/Myr
                        # 16: T1: Temperature (C ) below which biodegradation 
                        # rate is at maximum
                        # 17: sigma: Temperature range (C ) for the decrease 
                        # in biodegradation rate
                        # 18: Non-biodegradable fraction %
                        if ioutput_main == 1:
                            print(">> Inputs from ipa_traps.csv")
                            print(">>   OIL_API : ", OIL_API)
                            print(">>   sgrav_gas : ", sgrav_gas)
                            print(">>   oil_den_mod_fac : ", oil_den_mod_fac)
                            print(">>   gas_den_mod_fac : ", gas_den_mod_fac)
                            print(">>   delta_rho_gas_kg_m3 : ", 
                                                          delta_rho_gas_kg_m3)
                            print(">>   rho_sea : ", rho_sea)
                            print(">>   rho_brine : ", rho_brine)
                            print(">>   a_gas : ", a_gas)
                            print(">>   b_gas : ", b_gas)
                            print(">>   sGOR_at_charp : ", sGOR_at_charp)
                            print(">>   charp : ", charp)
                            print(">>   exp_fac : ", exp_fac)
                            print(">>   sGOR_scf_bbl_max : ", sGOR_scf_bbl_max)                               
                            print(">>   ntraps: ", ntraps)
                        tt1 = time.time()
                        idepth_type = 0
                        if iupdate_PWD == 1:
                            idepth_type = 1
                        sGORflag = "cumu"
                        charge_history.calc_charge_maps_main(
                                    output_path, sGORflag, res_age_strs, 
                                    res_tID, res_ev_IDs, 
                                    res_name_top, res_wb_names, idepth_type, 
                                    sGOR_at_charp, charp, exp_fac, a_gas, 
                                    b_gas, rho_sea, rho_brine, sgrav_gas
                                    )
                        sGORflag = "incr"
                        charge_history.calc_charge_maps_main(
                                    output_path, sGORflag, res_age_strs, 
                                    res_tID, res_ev_IDs, 
                                    res_name_top, res_wb_names, idepth_type, 
                                    sGOR_at_charp, charp, exp_fac, a_gas, 
                                    b_gas, rho_sea, rho_brine, sgrav_gas
                                    )
                        new_direc = "Charge_History_ZMaps"
                        target_direc = os.path.join(output_path, new_direc)                 
                        if os.path.isdir(target_direc) != True:
                            os.mkdir(target_direc) 
                        pf_list_all = [
                                       "Charge_bGOR_scf_bbl",
                                       "Charge_bGOR_g_g",
                                       "Charge_Gas_Tg",
                                       "Charge_Oil_Tg",
                                       "TRAP_fpress_psi",
                                       "TRAP_gasGOR_scf_bbl",
                                       "TRAP_iphase",
                                       "TRAP_oilGOR_scf_bbl",
                                       "TRAP_tempC",
                                       ]
                        for pf in pf_list_all:
                            new_direc = os.path.join("Charge_History_ZMaps", 
                                                     pf)
                            target_direc = os.path.join(output_path, new_direc)                   
                            if os.path.isdir(target_direc) != True:
                                os.mkdir(target_direc) 
                            move_files(
                                    [pf], 
                                    os.path.join(output_path, 
                                                 "Charge_History_ZMaps"), 
                                    target_direc, ioutput_main
                                    )
                        tt2 = time.time()
                        if ioutput_main == 1: 
                            print ("Finished calculating charge maps: "
                                   "cpu(s) : ", tt2-tt1)
                        if ntraps > 0:
                            tt1 = time.time()
                            trap_charge_history.calc_trap_masses_main(
                                        output_path, sGORflag, res_age_strs, 
                                        res_tID, res_ev_IDs, res_name_top,
                                        res_wb_names, idepth_type, 
                                        sGOR_at_charp, charp, exp_fac, a_gas, 
                                        b_gas, rho_sea, rho_brine, sgrav_gas, 
                                        oil_den_mod_fac, gas_den_mod_fac,
                                        delta_rho_gas_kg_m3, sGOR_scf_bbl_max, 
                                        ntraps, traps_dict, traps_output_dict
                                        )                      
                            tt2 = time.time()
                            if ioutput_main == 1: 
                                print ("Finished calculating trap masses: "
                                       "cpu(s) : ", tt2-tt1)
                                print ("Mem (MB) : ",
                                       process.memory_info().rss/1e6, "\n")
                        else:
                            if ioutput_main == 1: 
                                print (">> Trap masses are not calculated "
                                       "since there are no traps.")
                    else:
                        
                        if ioutput_main == 1: 
                            print (">> Phase maps are not calculated due to"
                                   " input selection: ioutput_burial, "
                                   "icalc_temp, ioutput_temp : ",
                                     ioutput_burial, icalc_temp, ioutput_TEMP)
                else:
                    if ioutput_main == 1: 
                        print(" !!! No sources were included in "
                              "source index list !!!")
            else:
                if ioutput_main == 1: 
                    print (" !!! No sources were defined in petroleum "
                           "systems element column of Model Inputs tab !!!")