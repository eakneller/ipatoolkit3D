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
import numpy as np
from numba import jit
import math_tools


@jit(nopython=True, cache=True)
def temp_transfer(nnodes, T_t_L, T, icount_out):
    for kkk in range(nnodes):
        T_t_L[icount_out][kkk] = T[kkk]
    return 1


@jit(nopython=True, cache=True)
def update_solu_vec(nnodes, T, T_top, T_base, vec_sol2, icount_solu):
    for j in range(nnodes):
            # Top Boundary Condition
            if j == 0:
                    T[j] = T_top
             # Bottom Boundary Condition
            elif j == nnodes - 1:
                    T[j] = T_base
            else:
                    Tn = vec_sol2[icount_solu][0]
                    icount_solu = icount_solu + 1
                    T[j]=Tn
    return icount_solu


@jit(nopython=True, cache=True)
def copy_and_apply_Tbcs(
                        i, jmax, nnodes, bc_coef1, bc_coef2, bc_coef3, 
                        coef_list, vec, T_top, T_base, T, Ts_old, bc_itype
):
    # Calculate temperature for each node at timestep i
    # Copy new temperature array if beyond first calculated time step
    # Otherwise, use information from last time step of main solution array
    if i > 1:                            
        for ii in range(nnodes):
            # Initialize Ts_old
            Ts_old[ii] = T[ii]
    # Calculate the temperature at the base of the model
    if bc_itype == 1: # bc_itype: 1 = heat flow bc
            #T_base = (To_m + (-2.0*q_bottom/dz_u
            #                     -2.0*k_uu/dz_u/dz_u*(To_m-To_u)+Q)*dt/rho/cp)
            To_u = Ts_old[jmax-1]
            To_m = Ts_old[jmax]
            T_base = To_m + bc_coef1 + bc_coef2*(To_m - To_u) + bc_coef3
    for j in range(1, nnodes-1): 
        # Looping over nodes right-hand side vector  
            T2o = Ts_old[j]
            Q2 = coef_list[j][0]
            rho_cp_dt = coef_list[j][1]
            c1 = coef_list[j][2]
            #c2 = coef_list[j][3]
            c3 = coef_list[j][4]            
            if j == 1: 
                # Node is next to upper boundary condition
                f = rho_cp_dt*T2o + Q2 - c1*T_top
            elif j == nnodes - 2: 
                # Node is next to lower boundary condition
                f = rho_cp_dt*T2o + Q2 - c3*T_base
            else:
                f = rho_cp_dt*T2o + Q2
            vec[j-1][0] = f


@jit(nopython=True, cache=True)
def build_matrix_and_vector(
                            nnodes, dt, zs, Q_zs, alpha_zs, 
                            rho_zs, k_zs, cp_zs,
                            coef_list, mat, vec, 
                            bc_itype, q_bottom, T_bottom
):
    T_base = 0.0
    bc_coef1 = 0.0
    bc_coef2 = 0.0
    bc_coef3 = 0.0
    jmax = 0
    alpha = 0.0
    alpha2 = 0.0
    icount_a = 0
    for j in range(nnodes): 
        # Looping over nodes, build matrix and vector
        if j > 0 and j < nnodes - 1:
                dz1 = (zs[j] - zs[j - 1])*1000.0
                dz2 = (zs[j + 1] - zs[j])*1000.0
                if dz1 == 0.0:
                    dz1 = 1.0
                if dz2 == 0.0:
                    dz2 = 1.0
                W = dz1+dz2
                Q2 = Q_zs[j]
                cp2 = cp_zs[j]
                alpha2 = alpha_zs[j]
                rho2 = rho_zs[j]
                k1 = k_zs[j - 1]
                k2 = k_zs[j]
                k3 = k_zs[j + 1]
                if k1 == 0:
                    k1 = 3.0
                elif k2 == 0.0:
                    k2 = 3.0
                elif k3 == 0.0:
                    k3 = 3.0
                # Harmonic averages for layered configuration
                kA = 2.0*k1*k2/(k1 + k2)
                kB = 2.0*k3*k2/(k3 + k2)                
                c1 = -2.0*kA/W/dz1    
                c2 = 2.0*kA/W/dz1 + 2.0*kB/W/dz2 + rho2*cp2/dt                
                c3 = -2.0*kB/W/dz2    
                rho_cp_dt = rho2*cp2/dt
                coef_list[j][0] = Q2
                coef_list[j][1] = rho_cp_dt
                coef_list[j][2] = c1
                coef_list[j][3] = c2
                coef_list[j][4] = c3
                icount_b = 0
                for k in range(nnodes): 
                    # Looping over all nodes
                    if k > 0 and k < nnodes-1: 
                        # Add coeffcients only for unknowns
                        if k == j - 1:
                            ck = c1
                        elif k == j:
                            ck = c2
                        elif k == j + 1:
                            ck = c3
                        else:
                            ck = 0.0
                        mat[icount_a][icount_b] = ck
                        icount_b = icount_b + 1
                vec[icount_a][0] = 0.0                
                icount_a = icount_a + 1                
        else:
                coef_list[j][0] = 0.0
                coef_list[j][1] = 0.0
                coef_list[j][2] = 0.0
                coef_list[j][3] = 0.0
                coef_list[j][4] = 0.0
        if bc_itype == 1: 
            # bc_itype: 1 = heat flow bc
            jmax = int(nnodes - 1)
            dz_u = (zs[jmax] - zs[jmax - 1])*1000.0
            if dz_u == 0.0:
                dz_u = 1.0
            k_u = k_zs[jmax - 1]
            k_m = k_zs[jmax]
            if k_u == 0.0:
                k_u = 3.0
            if k_m == 0.0:
                k_m = 3.0
            k_uu = 2.0*k_u*k_m/(k_u + k_m)
            Q = Q_zs[jmax]
            cp = cp_zs[jmax]
            alpha = alpha_zs[jmax]
            rho = rho_zs[jmax]
            if rho*cp == 0.0:
                rho = 3330.0
                cp = 1250.0
            bc_coef1 = -2.0*q_bottom/dz_u*dt/rho/cp
            bc_coef2 = -2.0*k_uu/dz_u/dz_u*dt/rho/cp
            bc_coef3 = Q*dt/rho/cp
        T_base = T_bottom
    return T_base, bc_coef1, bc_coef2, bc_coef3, jmax, alpha, alpha2


@jit(nopython=True, cache=True)
def FD_conduction_kvar_IMP(
                                istart, t_start_Ma, t_end, dt, dz,
                                zs, k_zs, cp_zs, alpha_zs, rho_zs, Q_zs,
                                bc_itype, HP_itype, T_top, T_bottom, q_bottom, 
                                z1, z2, L, sec_yr,
                                T_t_L, shf_t_L, ts_Myr_L, avg_Tc_t_L, 
                                avg_Tm_t_L, z_moho_t_L, z_mantle_t_L,
                                irho_var, dt_out, nnodes, ntimes, mat, vec, 
                                coef_list, T, Ts_old, 
                                xc, yc, xm, ym
):
    icount_out = 0
    # Output time step in seconds
    dt_out = dt_out*1e6*sec_yr
    if dt > 0.0:
        # Number of diffusion time steps per output timestep
        nskip = int(dt_out/dt)
    else:
        nskip = 1
    # Index of last diffusion time step
    ilast = ntimes - 1
    # Inititalize time step counter
    icount_ts = 0
    # Build matrix and vector
    (
     T_base, bc_coef1, 
     bc_coef2, bc_coef3, 
     jmax, alpha, alpha2
    ) = build_matrix_and_vector(
                                nnodes, dt, zs, Q_zs, alpha_zs, rho_zs, 
                                k_zs, cp_zs, coef_list, mat, vec, 
                                bc_itype, q_bottom, T_bottom
                            )
    for i in range(1, ntimes):
        # duration of diffusion
        t = float(i)*dt
        # Copy new to old vectors and apply bc's
        copy_and_apply_Tbcs(
                            i, jmax, nnodes, bc_coef1, bc_coef2, bc_coef3,
                            coef_list, vec, T_top, T_base, T, Ts_old, bc_itype
                        )
        vec_sol2 = np.linalg.solve(mat, vec)
        icount_solu = 0
        icount_solu = update_solu_vec(
                                        nnodes, T, T_top, T_base, 
                                        vec_sol2, icount_solu
                                    )
        # Save time step information for every nskip diffusion time steps
        if icount_ts == nskip:
            isave = 1
        else:
            isave = 0
        # Save time step information for the last diffusion time step
        if i == ilast:
            isave = 1   
        if isave == 1:
            icount_ts = 0
            # Add temperature list to master list
            temp_transfer(nnodes, T_t_L, T, icount_out)
            # Calculate surface heat flow
            k_d = k_zs[1]
            k_m = k_zs[0]
            k_dd = (k_m+k_d)/2.0
            dz_top = (zs[1]-zs[0])*1000.0
            if dz_top > 0.0:
                shf = -k_dd*(T[1]-T[0])/dz_top*1000.0
            else:
                dz_top = 1.0
                shf = -k_dd*(T[1]-T[0])/dz_top*1000.0
            shf_t_L[icount_out]=shf
            # Calculate time in Myr for this output time step
            ts_Myr_L[icount_out] = t/(1e6*sec_yr) + t_start_Ma
            # Calculate average temperatures in upper
            # and lower layers defined by z1
            avg_Tc, avg_Tm = math_tools.get_avg_temps_v2(
                                                        T, zs, xc, yc,
                                                        xm, ym, z1, L, nnodes
                                                        )
            avg_Tc_t_L[icount_out] = avg_Tc
            avg_Tm_t_L[icount_out] = avg_Tm
            # Record the depth of the moho and mantle at this time
            z_moho_t_L[icount_out] = z1
            z_mantle_t_L[icount_out] = z1
            icount_out = icount_out + 1
        icount_ts = icount_ts + 1
    return icount_out


def FD_RUNIT():
    # *** This needs to be updated with numpy arrays ***
    sec_yr = 365.0*24.0*60.0*60.0
    L = 125000.0
    dz = 1000.0
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
    z1 = 10000.0 # Moho
    z2 = 100000.0 # Mantle tracking depth
    rho_crust = 2800.0
    rho_mantle = 3330.0
    cp = 1200.0
    alpha = 3.3e-5
    Q_crust = 0.0
    Q_mantle = 0.0
    k_mantle = 3.0
    k_crust = 3.0
    Tini = 1330.0
    Ts_ini = []
    nnodes = int(L/dz)+1
    zs = []
    rho_zs = []
    cp_zs = []
    alpha_zs = []
    Q_zs = []
    k_zs = []
    for i in range(nnodes):
        z = dz*float(i)/1000.0
        zs.append(z)
        if z <= z1:
            rho = rho_crust
            Q = Q_crust
            k = k_crust
        else:
            rho = rho_mantle
            Q = Q_mantle
            k = k_mantle
        rho_zs.append(rho)
        cp_zs.append(cp)
        alpha_zs.append(alpha)
        Q_zs.append(Q)
        k_zs.append(k)
        Ts_ini.append(Tini)
    T_t = [[0.0, Ts_ini]]
    bc_stype = "TEMP"
    HP_stype = "NA"
    T_top = 0.0
    T_bottom = 1330.0
    q_bottom = -50e3
    shf_t = []
    ts_Myr = []
    avg_Tc_t = []
    avg_Tm_t = []
    z_moho_t = []
    z_mantle_t = []
    irho_var = 0
    istart = 0
    DT_FAC_PRIMARY = 4.0
    kappa = k_mantle/rho_crust/cp
    dt_limit =(dz*dz)/kappa/DT_FAC_PRIMARY
    itype = 0
    if itype == 0:
        dt_Myr = 1.0
        dt = dt_Myr*sec_yr*1e6
    else:
        dt = dt_limit
    t_start_Ma = 0.0
    t_end_Myr = 300.0
    t_end = t_end_Myr*1e6*sec_yr # seconds
    dt_out = 50.0 # Myr
    print ("Solution type = ", itype, ": 0 = IMP, 1 = EXP")
    if itype == 0:
        tsolve1 = time.time()
        FD_conduction_kvar_IMP(
                                            istart, t_start_Ma, t_end, dt, 
                                            dz, zs, k_zs, cp_zs,
                                            alpha_zs, rho_zs, Q_zs, bc_stype, 
                                            HP_stype, T_top, T_bottom, 
                                            q_bottom, z1, z2,
                                            L,sec_yr, T_t, shf_t, ts_Myr, 
                                            avg_Tc_t, avg_Tm_t, z_moho_t, 
                                            z_mantle_t,irho_var, dt_out
                                        )
        tsolve2 = time.time()
        print (" FD_conduction_kvar_IMP cpu(s) : ", tsolve2-tsolve1)
    nt = len(T_t)
    for j in range(nt):
        if itype == 0:
            sflag = "IMP"
        else:
            sflag = "EXP"
        file_name = "TEMP_itype_"+sflag+"_nt_"+str(j)+".csv"
        fout = open(file_name, 'w')
        str_out = "dt_Myr,"+str(dt_Myr)+"\n"
        fout.write(str_out)
        str_out = "dz_km,"+str(dz)+"\n"
        fout.write(str_out)
        str_out = "t_Ma,"+str(T_t[j][0])+"\n"
        fout.write(str_out)
        T = T_t[j][1]
        for i, z, in enumerate(zs):
            str_out = str(z)+","+str(T[i])+"\n"
            fout.write(str_out)
        fout.close()