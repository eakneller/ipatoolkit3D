# -*- coding: utf-8 -*-


def print_mid_point(event_age, age, output_dict, nx, ny):
    i = int(nx/2)
    j = int(ny/2)
    keys = list(output_dict.keys())
    print("event_age: ", event_age, " : top age: ", age, " : i, j: ", i, j)
    for key in keys:
        scalars_xy = output_dict[key]
        print(key, " : ", scalars_xy[i][j])
    

def print_main_temp_array(comment, istart, icount_out, T_t):
    
    input_dict = {
                    "istart":istart,
                    "icount_out":icount_out,
                    "T_t":T_t
                }
    istart = input_dict["istart"]
    icount_out = input_dict["icount_out"]
    T_t = input_dict["T_t"]
    print("")
    print("------------------------------------------------------------------")
    print("Main temperature array: ", comment, " istart, icount_out : ", istart, icount_out)
    print("------------------------------------------------------------------")
    for i in range(istart + 1 + icount_out):
        print("Time step ", i)
        print(T_t[i])
    

def print_FD_IMP_input(
                        bc_itype, istart, t_start_Ma, 
                        dt_rift_s, dt_diff_s, dz,
                        zs, Q_zs, cp_zs, alpha_zs, rho_zs, 
                        k_zs, T_t, shf_t, Te_t, 
                        q_thermal_t, ts_Myr, avg_Tc_t, 
                        avg_Tm_t, z_moho_t, z_mantle_t,
                        L, sec_yr, T_top, 
                        T_bottom, irho_var, dt_out,
                        moho_l, mantle_l, rho_crust, 
                        rho_mantle, alpha_mantle, Telastic
):
    input_dict = {
                "bc_itype":bc_itype, 
                "istart":istart, 
                "t_start_Ma":t_start_Ma, 
                "dt_rift_s":dt_rift_s, 
                "dt_diff_s":dt_diff_s,
                "dz":dz,
                "zs":zs, 
                "Q_zs":Q_zs, 
                "cp_zs":cp_zs, 
                "alpha_zs":alpha_zs, 
                "rho_zs":rho_zs, 
                "k_zs":k_zs, 
#                "T_t":T_t, 
#                "shf_t":shf_t, 
#                "Te_t":Te_t, 
#                "q_thermal_t":q_thermal_t, 
#                "ts_Myr":ts_Myr, 
#                "avg_Tc_t":avg_Tc_t, 
#                "avg_Tm_t":avg_Tm_t, 
#                "z_moho_t":z_moho_t, 
#                "z_mantle_t":z_mantle_t,
                "L":L, 
                "sec_yr": sec_yr, 
                "T_top":T_top, 
                "T_bottom":T_bottom, 
                "irho_var":irho_var, 
                "dt_out":dt_out,
                "moho_l":moho_l, 
                "mantle_l":mantle_l, 
                "rho_crust":rho_crust, 
                "rho_mantle":rho_mantle, 
                "alpha_mantle":alpha_mantle, 
                "Telastic":Telastic            
            }
    print("")
    print("------------------------------------------------------------------")
    print("call_FD_IMP_input: istart, t_start_Ma : ", input_dict["istart"], input_dict["t_start_Ma"])
    print("------------------------------------------------------------------")
    keys = list(input_dict.keys())
    for key in keys:
        print(key, input_dict[key])
        
        
def print_outputs_for_FD_calc(icount_out, nnodes, T_t_L, shf_t_L):
    print("------------------------------------------------------------------")
    print("FD_calc_output")
    print("------------------------------------------------------------------")
    for i in range(icount_out):
        print("icount_out, shf_t_L : ", i, shf_t_L[i])
    print("..................................................................")
    for i in range(icount_out):
        for inode in range(nnodes):
            T = T_t_L[i][inode]
            print("icount_out, inode, T_t_L : ", i, inode, T)
    return 1         
         
def print_inputs_for_FD_calc(
                                istart, t_start_Ma, tend, dt,
                                dz, zs_L, k_zs_L, cp_zs_L, alpha_zs_L, 
                                rho_zs_L, Q_zs_L, bc_itype,
                                HP_itype, T_top, T_bottom, 
                                moho_locp, mantle_locp, L, sec_yr,
                                T_t_L, shf_t_L, ts_Myr_L, avg_Tc_t_L, 
                                avg_Tm_t_L, z_moho_t_L, z_mantle_t_L,
                                irho_var, dt_out, nnodes, ntimes, mat, 
                                vec, coef_list, T_t, Ts_old_L,
                                xc, yc, xm, ym
):
    input_dict = {
                    "istart":istart, 
                    "t_start_Ma":t_start_Ma, 
                    "tend":tend, 
                    "dt":dt,
                    "dz":dz, 
                    "zs_L":zs_L, 
                    "k_zs_L":k_zs_L, 
                    "cp_zs_L":cp_zs_L, 
                    "alpha_zs_L":alpha_zs_L, 
                    "rho_zs_L":rho_zs_L, 
                    "Q_zs_L":Q_zs_L, 
                    "bc_itype":bc_itype,
                    "HP_itype":HP_itype, 
                    "T_top":T_top, 
                    "T_bottom":T_bottom, 
                    "moho_locp":moho_locp, 
                    "mantle_locp":mantle_locp, 
                    "L":L, 
                    "sec_yr":sec_yr,
#                    "T_t_L":T_t_L, 
#                    "shf_t_L":shf_t_L, 
#                    "ts_Myr_L":ts_Myr_L, 
#                    "avg_Tc_t_L":avg_Tc_t_L, 
#                    "avg_Tm_t_L":avg_Tm_t_L, 
#                    "z_moho_t_L":z_moho_t_L, 
#                    "z_mantle_t_L":z_mantle_t_L,
                    "irho_var":irho_var, 
                    "dt_out":dt_out, 
                    "nnodes":nnodes, 
                    "ntimes":ntimes, 
#                    "mat":mat, 
#                    "vec":vec,
#                    "coef_list":coef_list, 
#                    "T_t":T_t, 
                    "Ts_old_L":Ts_old_L,
#                    "xc":xc, 
#                    "yc":yc, 
#                    "xm":xm, 
#                    "ym":ym            
            }
    print("------------------------------------------------------------------")
    print("FD_calc_input: istart, t_start_Ma : ", input_dict["istart"], input_dict["t_start_Ma"])
    print("------------------------------------------------------------------")
    keys = list(input_dict.keys())
    for key in keys:
        print(key, input_dict[key])


def print_mesh1D_info(
                        age_event, age_event_prev, iuse_temp_dep_k, 
                        nrelax, nsublayers, nnodes, 
                        inode_max, event_type_ID, kappa_lith, 
                        sec_per_myr, T_top, q_bottom, 
                        tot_sed_thick_prev, tot_sed_thick,
                        zs, kg_zs, Qg_zs, phi_o_zs, Q_zs, c_zs, 
                        maxfb_zs, Tprev_zs, Tini_zs, Lthick_prev, 
                        layer_thick, elem_thick, elem_tops, 
                        elem_bottoms, layer_thick_pd, erothick, 
                        k_layer_zs, k_elem_zs, kg_up_zs, dzi, Hxdzi, 
                        Qbi, T_base, Tc_ss, T_trans, k_water, Q_water
):
    print("event_type_ID, age_event, age_event_prev : ", 
          event_type_ID, age_event, age_event_prev)
    print("iuse_temp_dep_k, nrelax, nsublayers : ", 
          iuse_temp_dep_k, nrelax, nsublayers)
    print("inode_max, kappa_lith, sec_per_myr : ",
          inode_max, kappa_lith, sec_per_myr)
    print("T_top, q_bottom, tot_sed_thick_prev, tot_sed_thick : ",
          T_top, q_bottom, tot_sed_thick_prev, tot_sed_thick)
    print("k_water, Q_water, erothick : ", k_water, Q_water, erothick)
    adict = {
                "zs":zs, 
                "kg_zs":kg_zs, 
                "Qg_zs":Qg_zs, 
                "phi_o_zs":phi_o_zs, 
                "Q_zs":Q_zs, 
                "c_zs":c_zs, 
                "maxfb_zs":maxfb_zs, 
                "Tprev_zs":Tprev_zs, 
                "Tini_zs":Tini_zs, 
                "Lthick_prev":Lthick_prev, 
                "layer_thick":layer_thick, 
                "ekem_thick":elem_thick, 
                "elem_tops":elem_tops, 
                "elem_bottoms":elem_bottoms, 
                "layer_thick_pd":layer_thick_pd,
                "k_layer_zs":k_layer_zs, 
                "k_elem_zs":k_elem_zs, 
                "kg_up_zs":kg_up_zs, 
                "dzi":dzi, 
                "Hxdzi":Hxdzi, 
                "Qbi":Qbi, 
                "T_base":T_base, 
                "Tc_ss":Tc_ss, 
                "T_trans":T_trans
        }
    keys = list(adict.keys())
    for key in keys:
        print("Array ", key)
        print(adict[key])
        print("")