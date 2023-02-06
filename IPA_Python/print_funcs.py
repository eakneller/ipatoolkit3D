# -*- coding: utf-8 -*-
import numpy as np

        
def debug_q_uc(ts_Myr, q_crustal_vec, i, j):
    iprob = 9
    jprob = 98
    ntimes = ts_Myr.size
    ifind_last = 0
    if i == iprob and j == jprob:
        print("q_uc_vec at node i, j : ", iprob, jprob, "ntimes", ntimes)
        for mm in range(ntimes):
            t = ts_Myr[mm]
            print(mm, t, q_crustal_vec[mm])
            if mm > 0 and t == 0.0 and ifind_last == 0:
                ifind_last = 1
                q_last = q_crustal_vec[mm-1]
                print("found last value : ", q_last)
            if mm > 0 and t == 0.0:
                print("clean q_crustal_vec : ", q_last)
#    if i == iprob+2 and j == jprob+2:
#        print("q_uc_vec at node i, j : ", iprob+2, jprob+2, "ntimes", ntimes)
#        for mm in range(ntimes):
#            print(mm, ts_Myr[mm], q_crustal_vec[mm])    

    
def print_finfo(ioutput_main, process, str_desc, dt_sec):
    if ioutput_main == 1: 
        print (
                str_desc + " : cpu(s) : ", dt_sec, 
                " : Mem (MB) : ",process.memory_info().rss/1e6)


def print_info(str_desc, val_list):
    for i, val in enumerate(val_list):
        if i == 0:
            vstr = str(val)
        else:
            vstr = vstr + " : " + str(val)
    print(">> " + str_desc + " : ", vstr)


def print_statement(desc):
    print(">> " + desc)


def print_warning(desc):
    print("!!!! " + desc + " !!!!")
    

def print_lithology_table(lith_dict):   
    print("------------------------------------------------------------------")
    print("Lithology Table")
    print("------------------------------------------------------------------")
    keys = list(lith_dict.keys())
    for key in keys:
        print(key, lith_dict[key])
    print("------------------------------------------------------------------")

    
def print_params(param_input_dict):
    keys = list(param_input_dict.keys())
    print("------------------------------------------------------------------")
    print("Input Parameters")
    print("------------------------------------------------------------------")
    for key in keys:
        if key not in ["deltaSL_list","PWD_interp_ages","swit_list"]:
            strout = key + " = " + str(param_input_dict[key])
            print(strout)
    print("------------------------------------------------------------------")


def print_strat(strat_input_dict):
    keys = list(strat_input_dict.keys())
    nkeys = len(keys)
    print("------------------------------------------------------------------")
    print("Stratigraphic Parameters")
    print("------------------------------------------------------------------")
    for i in range(nkeys):
        m = nkeys-1-i
        key = keys[m]
        if key not in ["deltaSL_list","PWD_interp_ages","swit_list"]:
            strout = str(key) + " = " + str(strat_input_dict[key])
            print(strout)
    print("------------------------------------------------------------------")
    

def print_deltas(riskmodel):
    output_dict = {                
                "max delta_bghf":np.max(riskmodel.delta_bghf_xy_np), 
                "max delta_xth":np.max(riskmodel.delta_xth_xy_np), 
                "max delta_mantlefac":np.max(riskmodel.delta_mantlefac_xy_np), 
                "max delta_defage":np.max(riskmodel.delta_defage_xy_np), 
                "max delta_hfred":np.max(riskmodel.delta_hfred_xy_np), 
                "max delta_Telastic":np.max(riskmodel.delta_Telastic), 
                "max delta_pdecay":np.max(riskmodel.delta_pdecay_xy_np), 
                "max delta_spor":np.max(riskmodel.delta_spor_xy_np), 
                "max delta_toc":np.max(riskmodel.delta_toc_xy_np), 
                "max delta_HI":np.max(riskmodel.delta_HI_xy_np), 
                "max delta_thick":np.max(riskmodel.delta_thick_xy_np), 
                "max delta_oilapi":np.max(riskmodel.delta_oilapi_xy_np), 
                "max delta_gasgrav":np.max(riskmodel.delta_gasgrav_xy_np),
                "max delta_porth":np.max(riskmodel.delta_porth_xy_np), 
                "max delta_adsth":riskmodel.delta_adsth, 
                "delta_inert":riskmodel.delta_inert, 
                "idelta_kinetics":riskmodel.idelta_kinetics, 
                "idelta_gas_frac":riskmodel.idelta_gas_frac, 
                "idelta_polar_frac":riskmodel.idelta_polar_frac, 
                "delta_sat_gor":riskmodel.delta_sat_gor, 
                "src_index_list_bulk_gor":riskmodel.src_index_list_bulk_gor
            }
    keys = list(output_dict.keys())
    print("------------------------------------------------------------------")
    print("Risk deltas")
    print("------------------------------------------------------------------")
    for key in keys:
        print(key, " : ", output_dict[key])
    print("------------------------------------------------------------------")

   
def print_depth_maps_0Ma(event_dict_bs, tops_list_bs):
    print("Depth Maps")
    keys = list(event_dict_bs.keys())
    nevents_bs = len(keys)
    event_ID_list_bs = keys[:]
    event_ID = event_ID_list_bs[nevents_bs-1]
    ntops = len(tops_list_bs)
    for kk in range(ntops):
        event_index = tops_list_bs[kk][14][event_ID]
        print(tops_list_bs[kk][1][event_index])


def print_events_when_tops_are_present(tops_list_bs):
    ntops = len(tops_list_bs)
    for kk in range(ntops):
        tname = tops_list_bs[kk][6]
        print("")
        print("Top ", tname, 
              "was present during the following events from old to young")
        print("Event ID list for when top is present")
        print(tops_list_bs[kk][0])


def print_top_info(tops_list_bs, event_dict_bs, nx, ny):
    nxmid = int(nx/2)
    nymid = int(ny/2)
    ntops = len(tops_list_bs)
    # define all event keys (integers from 0 to N) 
    # from oldest to youngest
    keys = list(event_dict_bs.keys())
    nevents_bs = len(keys)
    event_ID_list_bs = keys[:]
    # event ID of last event. This event ended at present day.
    # Only working on last event (i.e. present-day)
    event_ID = event_ID_list_bs[nevents_bs-1]
    for jj in range(ntops):
        kk = ntops - jj - 1    
        event_index = tops_list_bs[kk][14][event_ID]
        z_this_tops = tops_list_bs[kk][1]
        print("kk, event_index, event_ID : ", kk, event_index, event_ID)
        print("z_this_tops :")
        for i, top in enumerate(z_this_tops):
            print(i, top[nxmid][nymid])
        print("")
        if kk > 0:
            nelem = len(tops_list_bs[kk-1][0])
            event_index2 = nelem - 1
            z_older_tops = tops_list_bs[kk-1][1]
            print("kk-1, event_index2 : ", kk-1, event_index2)
            print("z_older_tops : ")
            for i, top in enumerate(z_older_tops):
                print(i, top[nxmid][nymid])
            print("")