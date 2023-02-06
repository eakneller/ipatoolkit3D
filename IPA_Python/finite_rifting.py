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
import math
import time
import FDME_HeatConduction
import steady_state_thermal
import math_tools
import advect_lithosphere
import density
import subsidence


@jit(nopython=True, cache=True)
def calc_q_uc(
                ntime, q_uc_t, z_moho_t, 
                L, rho_w, rho_crust, L_crust_ref):
    tco = L_crust_ref
    for i in range(ntime):
        tc = z_moho_t[i]    
        q_load = (tco-tc)*9.81*(rho_crust-rho_w)    
        q_uc_t[i] = q_load/1e6


@jit(nopython=True, cache=True)
def call_FD_IMP(
        bc_itype, istart, t_start_Ma, tend, 
        dt, dz, zs, Q_zs, cp_zs, alpha_zs, 
        rho_zs, k_zs, T_t, shf_t, Te_t,
        q_thermal_t, ts_Myr, avg_Tc_t, 
        avg_Tm_t, z_moho_t, z_mantle_t, L, sec_yr,
        T_top, T_bottom, irho_var, dt_out, 
        moho_locp, mantle_locp, rho_crust, 
        rho_mantle, alpha, Telastic
):    
    HP_itype = 0
    ntimes = int(tend/dt) + 1
    nnodes = zs.size
    mat = np.zeros((nnodes-2, nnodes-2))
    vec = np.zeros((nnodes-2, 1))
    coef_list = np.zeros((nnodes, 5))    
    # L stands for local...    
    # Create numpy arrays from lists
    zs_L = np.zeros((nnodes))
    Q_zs_L = np.zeros((nnodes))
    cp_zs_L = np.zeros((nnodes))
    alpha_zs_L = np.zeros((nnodes)) 
    rho_zs_L = np.zeros((nnodes))
    k_zs_L = np.zeros((nnodes))
    # initialize numpy arrays and get thickness of model
    L = 125000
    for j in range(nnodes):
        zs_L[j] = zs[j]
        Q_zs_L[j] = Q_zs[j]
        cp_zs_L[j] = cp_zs[j]
        alpha_zs_L[j] = alpha_zs[j]
        rho_zs_L[j] = rho_zs[j]
        k_zs_L[j] = k_zs[j]
        if j == nnodes - 1:
            L = zs[j]*1000        
    T_L = np.zeros((nnodes))
    Ts_old_L = np.zeros((nnodes))
    xc = np.zeros((nnodes))
    yc = np.zeros((nnodes))
    xm = np.zeros((nnodes))
    ym = np.zeros((nnodes))
    T_t_L = np.zeros((ntimes, nnodes))
    shf_t_L = np.zeros((ntimes))
    ts_Myr_L = np.zeros((ntimes))
    avg_Tc_t_L = np.zeros((ntimes))
    avg_Tm_t_L = np.zeros((ntimes))
    z_moho_t_L = np.zeros((ntimes))
    z_mantle_t_L = np.zeros((ntimes))
    # copy initial temperature to Ts_old array
    for ii in range(nnodes):
        Ts_old_L[ii] = T_t[istart][ii]
    icount_out = FDME_HeatConduction.FD_conduction_kvar_IMP(
                                        istart, t_start_Ma, tend, dt,
                                        dz, zs_L, k_zs_L, cp_zs_L, alpha_zs_L, 
                                        rho_zs_L, Q_zs_L, bc_itype,
                                        HP_itype, T_top, T_bottom, 0.0, 
                                        moho_locp, mantle_locp, L, sec_yr,
                                        T_t_L, shf_t_L, ts_Myr_L, avg_Tc_t_L, 
                                        avg_Tm_t_L, z_moho_t_L, z_mantle_t_L,
                                        irho_var, dt_out, nnodes, ntimes, mat, 
                                        vec, coef_list, T_L, Ts_old_L,
                                        xc, yc, xm, ym
                                    )
    for kkk in range(icount_out):        
        Te = 0.0
        load_thermal = 0.0
        T_dist_min = 1e32
        g = 9.81        
        mmm_close = -99999
        for mmm in range(nnodes):            
            zD = zs[mmm]*1000
            T = T_t_L[kkk][mmm]
            Tss = T_bottom/L*zD
            tc_f = z_moho_t_L[kkk]            
            dT = T-Tss            
            T_dist = abs(T-Telastic)            
            if T_dist < T_dist_min:
                Te = zD
                T_dist_min = T_dist
                mmm_close = mmm
            if zD <= tc_f:
                rho = rho_crust
            else:
                rho = rho_mantle
            if mmm < nnodes-1:
                dz = (zs[mmm+1]-zs[mmm])*1000.0
                load_thermal = load_thermal + g*rho*alpha*dT*dz
            T_t[istart + 1 + kkk][mmm] = T_t_L[kkk][mmm]
        # Calculate a more accurate Te
        T_close = T_t_L[kkk][mmm_close]
        zD_close = zs[mmm_close]*1000
        if T_close == Telastic:
            Te = zD_close
        elif T_close > Telastic:
            T_shallow = T_t_L[kkk][mmm_close-1]
            zD_shallow = zs[mmm_close-1]*1000
            Te = (
                    zD_shallow 
                    + (zD_close-zD_shallow)/(T_close-T_shallow)
                                                        *(Telastic-T_shallow)
                )
        elif T_close < Telastic:
            T_deeper = T_t_L[kkk][mmm_close+1]
            zD_deeper = zs[mmm_close+1]*1000
            Te = (
                    zD_close 
                    + (zD_deeper-zD_close)/(T_deeper-T_close)
                                                        *(Telastic-T_close)
                )
        Te_t[istart + 1 + kkk] = Te
        q_thermal_t[istart + 1 + kkk] = load_thermal/1e6
        shf_t[istart + 1 + kkk] = shf_t_L[kkk]
        ts_Myr[istart + 1 + kkk] = ts_Myr_L[kkk]
        avg_Tc_t[istart + 1 + kkk] = avg_Tc_t_L[kkk]
        avg_Tm_t[istart + 1 + kkk] = avg_Tm_t_L[kkk]
        z_moho_t[istart + 1 + kkk] = z_moho_t_L[kkk]
        z_mantle_t[istart + 1 + kkk] = z_mantle_t_L[kkk]
    return icount_out


@jit(nopython=True, cache=True)
def generate_initial_mesh(nnodes, dz, zs_np):
    # Generate initial zs
    for i in range(nnodes):
        z = float(i)*dz    
        zs_np[i]=z/1000.0


@jit(nopython=True, cache=True)
def update_diff_time(ndiff_time, dt_rift_s, dt_diff_s, dt):
    while dt_diff_s > dt:                        
        ndiff_time = ndiff_time + 1
        dt_diff_s = dt_rift_s/float(ndiff_time)
    return ndiff_time, dt_diff_s


@jit(nopython=True, cache=True)
def initialize_inputs_finite_and_instan_rifting(
                HP_itype, 
                T_bottom, nphases, dt_rift_Myr, icon_kappa,
                k_mantle, rho_mantle, cp_mantle, alpha_mantle,
                k_crust, rho_crust, alpha_crust, 
                Q_crust, Q_mantle, Ao, z_moho,
                delta_1, delta_2, delta_3,
                Beta_1, Beta_2, Beta_3,
                Llith, dz_lith, dt_Myr, 
                t_pr1, t_pr2, t_pr3,
                rift_time_1, rift_time_2, rift_time_3
):
    # Seconds per year
    sec_yr = 365.0*24.0*60.0*60.0
    dt_rift = dt_rift_Myr    
    # Limit number of phases to 3
    if nphases > 3:
        nphases = 3
    if nphases < 1:
        nphases = 1
    # Initialize heat production check flag
    HP_check = 0
    inicon_itype = 0
    L = Llith
    dz = dz_lith
    ifinite_rift = 1
    if HP_itype == 0:
        #"TWOLAYER"
        if Q_crust > 0.0:
            HP_check = 1
        elif Q_mantle > 0.0:
            HP_check = 1
    elif HP_itype == 1:
        #"EXP":
        if Ao > 0.0:
            HP_check = 1        
    if HP_check == 1:
        # Use steady-state with HP
        inicon_itype = 1
    else:
        # Use linear without HP   
        inicon_itype = 0
    # 0 = temperature bc; 1 = heat flow
    bc_itype = 0
    dt = dt_Myr*1e6*sec_yr
    t_pr1 = t_pr1*1e6*sec_yr
    t_pr2 = t_pr2*1e6*sec_yr
    t_pr3 = t_pr3*1e6*sec_yr
    # for instantaneous
    t_end = t_pr1
    if rift_time_1 < dt_rift:
        rift_time_1 = dt_rift
    if rift_time_2 < dt_rift:
        rift_time_2 = dt_rift
    if rift_time_3 < dt_rift:
        rift_time_3 = dt_rift
    ntime_rift_1 = int(rift_time_1/dt_rift)
    dt_rift_1 = rift_time_1/float(ntime_rift_1)
    ntime_rift_2 = int(rift_time_2/dt_rift)
    dt_rift_2 = rift_time_2/float(ntime_rift_2)
    ntime_rift_3 = int(rift_time_3/dt_rift)
    dt_rift_3 = rift_time_3/float(ntime_rift_3)
    ntime_riftp = np.zeros((3), dtype=np.int64)
    ntime_riftp[0] = ntime_rift_1
    ntime_riftp[1] = ntime_rift_2
    ntime_riftp[2] = ntime_rift_3
    ntime_riftp_max = int(np.max(ntime_riftp))
    rift_timep = np.zeros((3))
    rift_timep[0] = rift_time_1
    rift_timep[1] = rift_time_2
    rift_timep[2] = rift_time_3
    dt_riftp = np.zeros((3))
    dt_riftp[0] = dt_rift_1
    dt_riftp[1] = dt_rift_2
    dt_riftp[2] = dt_rift_3
    t_prp = np.zeros((3))
    t_prp[0] = t_pr1
    t_prp[1] = t_pr2
    t_prp[2] = t_pr3
    Betap = np.zeros((3))
    Betap[0] = Beta_1
    Betap[1] = Beta_2
    Betap[2] = Beta_3
    deltap = np.zeros((3))
    deltap[0] = delta_1
    deltap[1] = delta_2
    deltap[2] = delta_3
    z_moho_1 = z_moho
    z_moho_2 = z_moho_1/delta_1
    z_moho_3 = z_moho_2/delta_2
    z_mohop = np.zeros((3))
    z_mohop[0] = z_moho_1
    z_mohop[1] = z_moho_2
    z_mohop[2] = z_moho_3
    return (
            sec_yr, dt_rift, nphases,
            inicon_itype, bc_itype,
            L, dz, ifinite_rift,
            dt, t_end, t_pr1, t_pr2, t_pr3,
            rift_time_1, rift_time_2, rift_time_3,
            ntime_riftp, rift_timep, dt_riftp, t_prp,
            Betap, deltap, z_mohop, ntime_riftp_max   
        )


@jit(nopython=True, cache=True)
def initialize_FD_grids(
                        L, dz, ngrids_max, nphases,
                        z_mohop, HP_itype, k_crust, k_mantle,
                        Q_crust, Q_mantle, alpha_crust, alpha_mantle,
                        rho_crust, rho_mantle, cp_crust, cp_mantle,
                        Ao, ar
):
    nnodes_ini = int(L/dz) + 1
    dz = L/float(nnodes_ini-1)
    # number of nodes
    nnodes = int(L/dz) + 1
    # Generate initial zs
    zs = np.zeros((nnodes))
    generate_initial_mesh(nnodes, dz, zs)
    zs_ini = np.copy(zs)
    #imesh_t = np.zeros((ngrids_max))
    #imesh_t = [0]
    #imesh_tp = np.zeros((ngrids_max, 2))
    #imesh_tp = [[0,0]]
    # Lists for each rift phase
    zs_ktp = np.zeros((3, ngrids_max, nnodes))
    for jj in range(nnodes):
        zs_ktp[0][0][jj] = zs[jj]
        zs_ktp[1][0][jj] = zs[jj]
        zs_ktp[2][0][jj] = zs[jj]
    k_ztp = np.zeros((3, ngrids_max, nnodes))
    Q_ztp = np.zeros((3, ngrids_max, nnodes))
    cp_ztp = np.zeros((3, ngrids_max, nnodes))
    rho_ztp = np.zeros((3, ngrids_max, nnodes))
    alpha_ztp = np.zeros((3, ngrids_max, nnodes))
    for iphase in range(nphases):
        z_moho_tmp = z_mohop[iphase]
        for j, z in enumerate(zs):
            if z <= z_moho_tmp/1000.0:
                ktmp = k_crust
                #"TWOLAYER"
                if HP_itype == 0:
                    Qtmp = Q_crust
                else:
                    Qtmp = Ao*math.exp(-z*1000.0/ar)
                cptmp = cp_crust
                rhotmp = rho_crust
                alphatmp = alpha_crust
            else:
                ktmp = k_mantle
                #"TWOLAYER"
                if HP_itype == 0:
                    Qtmp = Q_mantle
                else:
                    Qtmp = Q_mantle
                cptmp = cp_mantle
                rhotmp = rho_mantle
                alphatmp = alpha_mantle
            k_ztp[iphase][0][j] = ktmp
            Q_ztp[iphase][0][j] = Qtmp
            cp_ztp[iphase][0][j] = cptmp
            rho_ztp[iphase][0][j] = rhotmp
            alpha_ztp[iphase][0][j] = alphatmp
    return (
            nnodes_ini, dz, nnodes, 
            zs, zs_ini, zs_ktp, k_ztp, Q_ztp,
            cp_ztp, rho_ztp, alpha_ztp
            
            )


@jit(nopython=True, cache=True)
def calc_kappa(k_ztp, cp_ztp, rho_ztp, rho_fac, irho_var):
    kappa = -1e32
    for jj, k_tmp in enumerate(k_ztp[0][0]):
        if irho_var == 0:
            kappa_tmp = k_tmp/rho_ztp[0][0][jj]/cp_ztp[0][0][jj]
        else:
            # Take into account maximum density reduction
            kappa_tmp = k_tmp/(rho_ztp[0][0][jj]-rho_fac)/cp_ztp[0][0][jj]
        if kappa_tmp > kappa:
            kappa = kappa_tmp
    return kappa
 

@jit(nopython=True, cache=True)    
def calc_initial_geotherm(
                            inicon_itype, zs, L, z_moho, 
                            T_top, T_bottom, k_crust, k_mantle,
                            zs_ktp, k_ztp, Q_ztp
):
    # Initital Geotherm
    if inicon_itype == 0:
        nnodes = zs.size # len(zs)
        #zs_np = np.asarray(zs)
        Tc_ini = np.zeros((nnodes))
        steady_state_thermal.linear_geotherm_v3(
                                            L, z_moho, zs, Tc_ini,
                                            nnodes, T_top, T_bottom, 
                                            k_crust, k_mantle
                                        )
    elif inicon_itype == 1:
        zs_ktp_np = zs_ktp[0][0]
        k_ztp_np = k_ztp[0][0]
        Q_ztp_np = Q_ztp[0][0]
        nnodes = zs_ktp[0][0].size
        Tc_ini = np.zeros((nnodes))
        dzi_np = np.zeros((nnodes))
        steady_state_thermal.steady_state_nlayer_TempBCs_v2(
                                                 nnodes, T_top, T_bottom, 
                                                 zs_ktp_np, k_ztp_np, 
                                                 Q_ztp_np, Tc_ini, dzi_np
                                                )
    return Tc_ini


@jit(nopython=True, cache=True)
def get_total_timesteps(sec_yr, dt, t_prp, dt_riftp):
    ntime_tot = int(t_prp[0]/dt) + 1 + (t_prp[1]/dt) + 1 + (t_prp[2]/dt) + 1
    ntime_tot = (
                    ntime_tot 
                    + int(dt_riftp[0]*1e6*sec_yr/dt) + 1
                    + int(dt_riftp[1]*1e6*sec_yr/dt) + 1
                    + int(dt_riftp[2]*1e6*sec_yr/dt) + 1
                )
    npad = int(ntime_tot/2)
    ntime_tot = int(ntime_tot + npad)
    return ntime_tot


@jit(nopython=True, cache=True) 
def initial_hf_and_Te(Tc_ini, zs_ini, k_ztp, Telastic, T_bottom, L):
    T1 = Tc_ini[0]
    T2 = Tc_ini[1]
    dzz = (zs_ini[1]-zs_ini[0])*1000.0   
    kavg = (k_ztp[0][0][1]+k_ztp[0][0][0])/2.0
    hf = -kavg*(T2-T1)/dzz*1000.0
    Te_ini = Telastic/(T_bottom/L)
    
    return hf, Te_ini


@jit(nopython=True, cache=True) 
def initialize_solution_arrays(
                                ntime_tot, nnodes, Tc_ini, hf, Te_ini, 
                                avg_Tc, avg_Tm, z_moho, L
):
    ts_Myr = np.zeros((ntime_tot))
    T_t = np.zeros((ntime_tot, nnodes))
    shf_t = np.zeros((ntime_tot))
    Te_t = np.zeros((ntime_tot))
    q_thermal_t = np.zeros((ntime_tot))
    z_moho_t = np.zeros((ntime_tot))
    z_mantle_t = np.zeros((ntime_tot))
    avg_Tc_t = np.zeros((ntime_tot))
    avg_Tm_t = np.zeros((ntime_tot))
    rho_c_t = np.zeros((ntime_tot))
    rho_m_t = np.zeros((ntime_tot))
    sub_t = np.zeros((ntime_tot))
    q_uc_t = np.zeros((ntime_tot))
    avg_Tc_t = np.zeros((ntime_tot))
    avg_Tm_t = np.zeros((ntime_tot))
    z_mantle_t = np.zeros((ntime_tot))
    shf_t[0] = hf
    Te_t[0]  = Te_ini
    avg_Tc_t[0] = avg_Tc
    avg_Tm_t[0] = avg_Tm
    z_moho_t[0] = z_moho
    z_mantle_t[0] = L
    for jj in range(nnodes):
        T_t[0][jj] = Tc_ini[jj]
    return (
            ts_Myr, T_t, shf_t, q_thermal_t, z_moho_t, z_mantle_t, avg_Tc_t,
            avg_Tm_t, rho_c_t, rho_m_t, sub_t, q_uc_t, Te_t
        )


@jit(nopython=True, cache=True)
def calculate_kinematic_parameters(
                                    nphases, ntime_riftp_max, z_mohop, 
                                    dt_riftp, ntime_riftp, deltap, Betap,
                                    rift_itype, L, zs, dz,
                                    HP_itype, rho_crust, cp_crust, 
                                    k_crust, Q_crust, alpha_crust, Ao, ar,
                                    rho_mantle, cp_mantle, k_mantle, Q_mantle,
                                    alpha_mantle, k_ztp, Q_ztp, cp_ztp, 
                                    alpha_ztp, rho_ztp
):
    rift_timesp = np.zeros((nphases, ntime_riftp_max))
    moho_locp = np.zeros((nphases, ntime_riftp_max))
    mantle_locp = np.zeros((nphases, ntime_riftp_max))
    duz_mohop = np.zeros((nphases, ntime_riftp_max))
    duz_mantlep = np.zeros((nphases, ntime_riftp_max))
    for iphase in range(nphases):
        z_moho_tmp = z_mohop[iphase]
        dt_rift_tmp = dt_riftp[iphase]
        ntime_rift_tmp = int(ntime_riftp[iphase])
        delta_tmp = deltap[iphase]
        Beta_tmp = Betap[iphase]
        # Calculate rift parameters
        if rift_itype == 1: 
            (
                rift_times, moho_loc, 
                mantle_loc, duz_moho, 
                duz_mantle
            ) = advect_lithosphere.advect_boundaries_Jarvis80(
                                            z_moho_tmp, L, dt_rift_tmp,
                                            ntime_rift_tmp, delta_tmp, 
                                            Beta_tmp, ntime_riftp_max
                                        )
        else:
            (
                rift_times, moho_loc, 
                mantle_loc, duz_moho, 
                duz_mantle
            ) = advect_lithosphere.advect_boundaries_Karner97(
                                                z_moho_tmp, L, dt_rift_tmp,
                                                ntime_rift_tmp, delta_tmp, 
                                                Beta_tmp, ntime_riftp_max
                                            )
        for jj in range(ntime_rift_tmp + 1):
            rift_timesp[iphase][jj] = rift_times[jj]
            moho_locp[iphase][jj] = moho_loc[jj]
            mantle_locp[iphase][jj] = mantle_loc[jj]
            duz_mohop[iphase][jj] = duz_moho[jj]
            duz_mantlep[iphase][jj] = duz_mantle[jj]
        nnodes = zs.size
        ilast = ntime_rift_tmp
        for i in range(ntime_riftp[iphase]):
            if i < ilast:
                z_moho_tmp = moho_loc[i + 1]
                for j in range(nnodes):
                    z = float(j)*dz
                    if z <= z_moho_tmp:
                        ktmp = k_crust
                        Qtmp = Q_crust
                         #"TWOLAYER"
                        if HP_itype == 0:
                            Qtmp = Q_crust
                        else:
                            Qtmp = Ao*math.exp(-z/ar)
                        cptmp = cp_crust
                        rhotmp = rho_crust
                        alphatmp = alpha_crust
                    else:
                        ktmp = k_mantle
                        #"TWOLAYER"
                        if HP_itype == 0:
                            Qtmp = Q_mantle
                        else:
                            Qtmp = Q_mantle
                        cptmp = cp_mantle
                        rhotmp = rho_mantle
                        alphatmp = alpha_mantle
                    k_ztp[iphase][i + 1][j] = ktmp
                    Q_ztp[iphase][i + 1][j] = Qtmp
                    cp_ztp[iphase][i + 1][j] = cptmp
                    rho_ztp[iphase][i + 1][j] = rhotmp
                    alpha_ztp[iphase][i + 1][j] = alphatmp
    return(
            rift_timesp, moho_locp, mantle_locp,
            duz_mohop, duz_mantlep
        )
    

@jit(nopython=True, cache=True)
def correct_for_instan_advection(
                                    nnodes, istart, iphase, igrid, 
                                    T, k_ztp, Q_ztp, cp_ztp, 
                                    rho_ztp, alpha_ztp, T_t,
                                    moho_locp, mantle_locp
):
    for j in range(nnodes):
        T_t[istart][j] = T[j]
        k_ztp[iphase][igrid][j] = k_ztp[iphase][igrid + 1][j]
        Q_ztp[iphase][igrid][j] = Q_ztp[iphase][igrid + 1][j]
        cp_ztp[iphase][igrid][j] = cp_ztp[iphase][igrid + 1][j]
        rho_ztp[iphase][igrid][j] = rho_ztp[iphase][igrid + 1][j]
        alpha_ztp[iphase][igrid][j] = alpha_ztp[iphase][igrid + 1][j]    
    moho_l = moho_locp[iphase][igrid + 1]
    mantle_l = mantle_locp[iphase][igrid + 1]
    return moho_l, mantle_l


@jit(nopython=True, cache=True)
def finite_rifting(
                    dt_Myr, dt_rift_Myr, dt_out, 
                    k_crust, k_mantle, cp_crust, cp_mantle,
                    alpha_crust, alpha_mantle,
                    rho_crust, rho_mantle, rho_sea, Llith, z_moho,
                    dz_lith, T_top, T_bottom, 
                    HP_itype, Q_crust, Q_mantle, Ao, ar,
                    rift_itype, nphases, delta_1, Beta_1, 
                    rift_time_1, t_pr1, delta_2, Beta_2, 
                    rift_time_2, t_pr2, delta_3, Beta_3, 
                    rift_time_3, t_pr3, icon_kappa, irho_var, 
                    iup_in_air, itype_rho_a,
                    L_crust_ref, L_lith_ref, 
                    rho_crust_ref, rho_mantle_ref, Telastic
):
    (
        sec_yr, dt_rift, nphases,
        inicon_itype, bc_itype,
        L, dz, ifinite_rift,
        dt, t_end, t_pr1, t_pr2, t_pr3,
        rift_time_1, rift_time_2, rift_time_3,
        ntime_riftp, rift_timep, dt_riftp, t_prp,
        Betap, deltap, z_mohop, ntime_riftp_max
    ) = initialize_inputs_finite_and_instan_rifting(
                    HP_itype, 
                    T_bottom, nphases, dt_rift_Myr, icon_kappa,
                    k_mantle, rho_mantle, cp_mantle, alpha_mantle,
                    k_crust, rho_crust, alpha_crust, 
                    Q_crust, Q_mantle, Ao, z_moho,
                    delta_1, delta_2, delta_3,
                    Beta_1, Beta_2, Beta_3,
                    Llith, dz_lith, dt_Myr, 
                    t_pr1, t_pr2, t_pr3,
                    rift_time_1, rift_time_2, rift_time_3
                    )
    ntime_tot = get_total_timesteps(sec_yr, dt, t_prp, dt_riftp)
    ngrids_max = 50
    (
        nnodes_ini, dz, nnodes, 
        zs, zs_ini, zs_ktp, k_ztp, Q_ztp,
        cp_ztp, rho_ztp, alpha_ztp
            
    ) = initialize_FD_grids(
                        L, dz, ngrids_max, nphases,
                        z_mohop, HP_itype, k_crust, k_mantle,
                        Q_crust, Q_mantle, alpha_crust, alpha_mantle,
                        rho_crust, rho_mantle, cp_crust, cp_mantle,
                        Ao, ar
                        )
    Tc_ini = calc_initial_geotherm(
                                inicon_itype, zs, L, z_moho, 
                                T_top, T_bottom, k_crust, k_mantle,
                                zs_ktp, k_ztp, Q_ztp
                                )
    # Initital heat flow
    hf, Te_ini = initial_hf_and_Te(
                                    Tc_ini, zs_ini, k_ztp,
                                    Telastic, T_bottom, L
                                    )
    xc = np.zeros((nnodes))
    yc = np.zeros((nnodes))
    xm = np.zeros((nnodes))
    ym = np.zeros((nnodes))
    avg_Tc, avg_Tm = math_tools.get_avg_temps_v2(
                                                Tc_ini, zs, 
                                                xc, yc, xm, ym, 
                                                z_moho, L, nnodes
                                                )
    # Initialize Solution Lists
    (
        ts_Myr, T_t, 
        shf_t, q_thermal_t, 
        z_moho_t, z_mantle_t, 
        avg_Tc_t, avg_Tm_t, 
        rho_c_t, rho_m_t, 
        sub_t, q_uc_t, Te_t
    ) =  initialize_solution_arrays(
                                    ntime_tot, nnodes, Tc_ini, 
                                    hf, Te_ini, avg_Tc, avg_Tm,
                                    z_moho, L
                                    )
    (
         rift_timesp, moho_locp, 
         mantle_locp, duz_mohop, 
         duz_mantlep
     ) = calculate_kinematic_parameters(
                                    nphases, ntime_riftp_max, z_mohop, 
                                    dt_riftp, ntime_riftp, deltap, Betap,
                                    rift_itype, L, zs, dz,
                                    HP_itype, rho_crust, cp_crust, 
                                    k_crust, Q_crust, alpha_crust, Ao, ar,
                                    rho_mantle, cp_mantle, k_mantle, Q_mantle,
                                    alpha_mantle, k_ztp, Q_ztp, cp_ztp, 
                                    alpha_ztp, rho_ztp
                                    )
    number_of_timesteps = 1
    for iphase in range(nphases):
        ntime_rift_tmp = int(ntime_riftp[iphase])
        ilast = ntime_rift_tmp
        dt_rift = dt_riftp[iphase]
        t_end = t_prp[iphase]
        t_add = 0.0
        if iphase > 0:
            for jj in range(iphase):
                t_add = t_add + rift_timep[jj] + t_prp[jj]/(1e6*sec_yr)
        t_add = t_add
        # for each displacement step of rift phase...
        for i, t_rift in enumerate(rift_timesp[iphase]):
            if i < ilast:
                t_start_Ma = rift_timesp[iphase][i] + t_add
                k_zs = k_ztp[iphase][i + 1]
                cp_zs = cp_ztp[iphase][i + 1]
                rho_zs = rho_ztp[iphase][i + 1]
                Q_zs = Q_ztp[iphase][i + 1]
                alpha_zs = alpha_ztp[iphase][i + 1]
                dt_rift_s = dt_rift*1e6*sec_yr
                ndiff_time = int(dt_rift_s/dt)
                if ndiff_time < 0:
                    ndiff_time = 1
                dt_diff_s = dt_rift_s/float(ndiff_time)
                (
                    ndiff_time, dt_diff_s
                ) = update_diff_time(ndiff_time, dt_rift_s, dt_diff_s, dt)
                zz_moho = moho_locp[iphase][i]
                zz_mantle = mantle_locp[iphase][i]
                # We need to avoid using initial values at i == 0
                duz_moho_i = duz_mohop[iphase][i + 1]
                duz_mantle_i = duz_mantlep[iphase][i + 1]
                # Inititalize with the last temperature entry
                istart = number_of_timesteps - 1
                T = advect_lithosphere.advect_and_interpolate(
                                    T_t[istart], zs, zz_moho, zz_mantle, 
                                    duz_moho_i, duz_mantle_i
                                    )
                # Correct for instantaneous advection
                moho_l, mantle_l = correct_for_instan_advection(
                                                nnodes, istart, iphase, i, 
                                                T, k_ztp, Q_ztp, cp_ztp, 
                                                rho_ztp, alpha_ztp, T_t,
                                                moho_locp, mantle_locp
                                                )
                icount_out = call_FD_IMP(
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
                                        )
                number_of_timesteps = number_of_timesteps + icount_out
        istart = number_of_timesteps - 1
        ilast = ntime_rift_tmp
        z_moho_f = moho_locp[iphase][ilast]
        z_mantle_f = mantle_locp[iphase][ilast]
        k_zs = k_ztp[iphase][ilast]
        cp_zs = cp_ztp[iphase][ilast]
        rho_zs = rho_ztp[iphase][ilast]
        Q_zs = Q_ztp[iphase][ilast]
        alpha_zs = alpha_ztp[iphase][ilast]
        t_start_Ma = ts_Myr[istart]          
        ntimes = int(t_end/dt) + 1           
        # Now update time step used for diffusion
        dt_orig = dt
        dt = t_end/float(ntimes-1)
        icount_out = call_FD_IMP(
                                    bc_itype, istart, t_start_Ma, t_end, 
                                    dt, dz, zs, Q_zs, cp_zs, alpha_zs, 
                                    rho_zs, k_zs, T_t, shf_t, Te_t, 
                                    q_thermal_t, ts_Myr, avg_Tc_t, 
                                    avg_Tm_t, z_moho_t, z_mantle_t,
                                    L, sec_yr, T_top, T_bottom, 
                                    irho_var, dt_out, z_moho_f, z_mantle_f, 
                                    rho_crust, rho_mantle, alpha_mantle, 
                                    Telastic
                                    )
        number_of_timesteps = number_of_timesteps + icount_out
        # reset timestep to original value
        dt = dt_orig
    nntime = avg_Tc_t.size
    rho_c_t = np.zeros((nntime))
    rho_m_t = np.zeros((nntime))
    density.calc_rho(
                        nntime, rho_c_t, rho_m_t, z_moho_t, 
                        L, alpha_crust, alpha_mantle, rho_crust, 
                        rho_mantle, avg_Tc_t, avg_Tm_t
                    )
    nntime = rho_c_t.size
    subsidence.calc_sub(
                            nntime, sub_t, z_moho_t, L, 
                            rho_sea, rho_c_t,
                            rho_m_t, alpha_crust, T_bottom, 
                            iup_in_air, rho_crust, rho_mantle, itype_rho_a,
                            L_crust_ref, L_lith_ref, rho_crust_ref, 
                            rho_mantle_ref
                        )
    q_uc_t = np.zeros((nntime))
    calc_q_uc(
                nntime, q_uc_t, z_moho_t, 
                L, rho_sea, rho_crust, L_crust_ref
            )
    return (
            number_of_timesteps, 
            ts_Myr, shf_t, sub_t, 
            Te_t, q_uc_t, q_thermal_t
        )


def rifting_RUNIT():
    itype_rho_a = 1
    # Output time step in Myr
    dt_out = 1.0
    dt_Myr = 1.0
    dt_rift_Myr = 1.0
### k_crust : Thermal conductivity of crust (W/m/K)              
### k_mantle : Thermal conductivity of mantle (W/m/K)           
### cp_crust : Specific heat of crust (J/kg/K)                  
### cp_mantle : Specific heat of mantle (J/kg/K)                 
### alpha_crust : Isobaric thermal expansivity of crust (K^-1)   
### alpha_mantle : Isobaric thermal expansivity of mantle (K^-1) 
### B_crust : Isothermal compressibility of crust (Pa^-1)        
### B_mantle : Isothermal compressibility of mantle (Pa^-1)      
### rho_crust : 0 degree C density of the crust (kg/m^3)         
### rho_mantle : 0 degree C density of the mantle (kg/m^3)      
### rho_sea : Density of sea water (kg/m^3)                     
### Llith : Depth of model & initial thickness of litho. (m)    
### z_moho : Initial crustal thickness (m)                      
### dz_lith: Finite difference node spacing for lithosphere (m) 
### gamma_MPa_K : Clapeyron slope for solidus (MPa/C)          
### Tmelt_o: Zero pressure melting temperature for solidus (C)   
    k_crust = 3.0
    k_mantle = 3.0
    cp_crust = 1250.0
    cp_mantle = 1051.051
    alpha_crust = 3.4e-5
    alpha_mantle = 3.4e-5
    rho_crust = 2800.0
    rho_mantle = 3330.0
    rho_sea = 1030.0
    Llith = 125000.0
    z_moho = 32000.0
    dz_lith = 5000.0
### T_top: Temperature at the top of the model in degrees C           
    T_top = 0.0        
### T_bottom: Temperature at the base of model (deg C)       
    T_bottom = 1330.0
    #q_bottom = -50e-3
### HP_itype                                              
### Q_crust: Crustal heat production for two layer model (W/m^3)  
### Q_mantle: Mantle heat production for two layer model (W/m^3) 
### Ao: Surface heat production for exponential model (W/m^3)   
### ar: Heat production depth constant for exponential model (m)  
    HP_itype = 0
    Q_crust = 0
    Q_mantle = 0
    Ao = 0
    ar = 8400.0
### rift_itype: 0 = constant velocity, 1 = constant strain rate     
### dt_rift: incremental rift time step used for finite rifting     
### nphases: number of rift phases (max = 3)                        
    rift_itype = 1
    #dt_rift = dt_rift_Myr
    nphases = 1
##############                                                       
# Rift Phase 1                                                     
##############                                                      
### delta_1: Stretching parameter for crust                      
### Beta_1: stretching parameter for mantle
### rift_time_1: Duration of rift phase Myr                   
### t_pr1: Length of post-rift phase (Myr)                 
##############                                                 
# Rift Phase 2                                               
##############                                                 
### delta_2: Stretching parameter for crust           
### Beta_2: stretching parameter for mantle                
### rift_time_2: Duration of rift phase Myr                  
### t_pr2: Length of post-rift phase (Myr)             
##############                                                   
# Rift Phase 3                                                    
##############                                           
### delta_3: Stretching parameter for crust                
### Beta_3: stretching parameter for mantle       
### rift_time_3: Duration of rift phase Myr                 
### t_pr3: Length of post-rift phase (Myr)          
    delta_1 = 2
    Beta_1 = 2
    rift_time_1 = 5
    t_pr1 = 100
    delta_2 = 1
    Beta_2 = 2.0
    rift_time_2 = 10.0 
    t_pr2 = 40.0
    delta_3 = 1.0
    Beta_3 =2.0
    rift_time_3 = 10.0 
    t_pr3 = 200.0
### icon_kappa: 0 = variable kappa; 1 = constant kappa in lithos.
### irho_var: 0 = constant rho in heat equation; 1 = variable rho
### iup_in_air: 0 = ulift into water, 1 = uplift into air
    icon_kappa = 1
    irho_var = 0
    iup_in_air = 1
    t1 = time.time()
    Telastic = 100
    L_crust_ref = z_moho
    L_lith_ref = Llith
    rho_crust_ref = rho_crust
    rho_mantle_ref = rho_mantle
    (
        number_of_timesteps,
        ts_Myr, shf_t, 
        sub_t, Te_t, 
        q_uc_t, q_thermal_t
    ) = finite_rifting.finite_rifting(
                dt_Myr, dt_rift_Myr, dt_out, 
                k_crust, k_mantle, cp_crust, cp_mantle, 
                alpha_crust, alpha_mantle, 
                rho_crust, rho_mantle, rho_sea, Llith, z_moho, 
                dz_lith, T_top, T_bottom, 
                HP_itype, Q_crust, Q_mantle, Ao, ar, 
                rift_itype, nphases, delta_1, Beta_1, 
                rift_time_1, t_pr1, delta_2, Beta_2, 
                rift_time_2, t_pr2, delta_3, Beta_3, 
                rift_time_3, t_pr3, icon_kappa, irho_var, 
                iup_in_air, itype_rho_a, 
                L_crust_ref, L_lith_ref, 
                rho_crust_ref, rho_mantle_ref, Telastic
            )  
    t2 = time.time()
    nsub = len(sub_t)
    print ("ts_Myr, final tt_sub : ", ts_Myr[nsub-1], sub_t[nsub-1])    
    tMyr = 1.0
    ttsub = math_tools.linear_interp_v2(tMyr, ts_Myr, sub_t)
    print ("ts_Myr, tt_sub (interp) : ", tMyr, ttsub)
    hf_interp = math_tools.linear_interp_v2(tMyr, ts_Myr, shf_t)
    print ("ts_Myr, hf (interp) : ", tMyr, hf_interp)
    Te_interp = math_tools.linear_interp_v2(tMyr, ts_Myr, Te_t)
    print ("ts_Myr, Te (interp) : ", tMyr, Te_interp)    
    q_uc_interp = math_tools.linear_interp_v2(tMyr, ts_Myr, q_uc_t)
    print ("ts_Myr, q_uc (interp) MPa, q_uc_ana MPa : ", 
           tMyr, q_uc_interp/1e6, 
           (z_moho-z_moho/delta_1)*9.81*(rho_crust-rho_sea)/1e6)
    q_thermal_interp = math_tools.linear_interp_v2(tMyr, ts_Myr, q_thermal_t)
    print ("ts_Myr, q_thermal (interp) MPa : ", tMyr, q_thermal_interp/1e6)
    print ("rifting_RUNIT() cpu(s) : ", t2-t1)