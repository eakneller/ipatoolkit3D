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
import math_tools
import math
from numba import jit


@jit(nopython=True, cache=True)
def Analytical_Temp_v2(
                        nnodes, alpha, kappa, 
                        rhoc, rhom, rhow, Telastic, g,
                        zs, ts_Myr, delta, Beta, L, 
                        T_bottom, z_moho, Tn
):
    n = 20
    if ts_Myr < 0.0:
        ts_Myr_f = 0.0
    else:
        ts_Myr_f = ts_Myr
    ts = ts_Myr_f*1e6*365*24*60*60 
    tau = L*L/kappa/math.pi/math.pi
    tc_o = z_moho
    tc_f = z_moho/delta
    Te = 0
    T_dist_min = 1e32
    q_uc = (tc_o - tc_f)*9.81*(rhoc-rhow)
    Beta_L = L*delta*Beta/(z_moho*Beta+(L-z_moho)*delta)
    T_dist_min = 1e32
    load_thermal = 0.0
    for i in range(nnodes):     
        z = zs[i]
        zD = z*1000
        zb = L-z*1000.0
        Tss = T_bottom-T_bottom*zb/L
        sumit = 0.0
        for j in range(n):
            nk = float(j+1)
            C1 = 2.0*(-1.0)**(nk+1.0)
            C2 = (
                    1.0/nk/nk/math.pi/math.pi*(
                                              (delta-Beta)*math.sin(
                                                     nk*math.pi*z_moho/L/delta)
                                             + Beta*math.sin(nk*math.pi/Beta_L)
                                            )
                )
            C = C1*C2
            fval = C*math.sin(nk*math.pi*zb/L)*math.exp(-nk*nk*ts/tau)
            sumit = sumit+fval
        T = Tss + T_bottom*sumit
        T_dist = abs(T-Telastic)
        if T_dist < T_dist_min:
            Te = zD
            T_dist_min = T_dist
            iclose = i
        dT = T-Tss
        Tn[i] = T
        if zD <= tc_f:
            rho = rhoc
        else:
            rho = rhom
        if i < nnodes-1:
            dz = (zs[i+1]-zs[i])*1000.0
            load_thermal = load_thermal + g*rho*alpha*dT*dz
    # Calculate a more accurate Te
    T_close = Tn[iclose]
    zD_close = zs[iclose]*1000
    if T_close == Telastic:
        Te = zD_close
    elif T_close > Telastic:
        T_shallow = Tn[iclose-1]
        zD_shallow = zs[iclose-1]*1000
        Te = (
                zD_shallow 
              + (zD_close-zD_shallow)/(T_close-T_shallow)*(Telastic-T_shallow)
            )
    elif T_close < Telastic:
        T_deeper = Tn[iclose+1]
        zD_deeper = zs[iclose+1]*1000
        Te = (
                zD_close 
                + (zD_deeper-zD_close)/(T_deeper-T_close)*(Telastic-T_close)
            )
    return load_thermal/1e6, q_uc/1e6, Te


@jit(nopython=True, cache=True)
def Analytical_HeatFlow_v2(
                            ts_Myr, hf_vec, delta, Beta, L, 
                            kappa, k, T_bottom, sec_yr, z_moho
                        ):
    iuse_McKenzie78 = 0
    iuse_Hellinger83 = 1
    n = 100
    tau = L*L/kappa/math.pi/math.pi
    if iuse_McKenzie78 == 1:
        con1 = k*T_bottom/L
        for i, tMyr in enumerate(ts_Myr):
            ts = tMyr*1e6*sec_yr
            sumit = 0.0
            for j in range(n):
                nf = float(j+1)
                p1 = Beta/nf/math.pi*math.sin(nf*math.pi/Beta)
                p2 = math.exp(-nf*nf*ts/tau)
                sumit = sumit + p1*p2
            shf = con1*(1.0+2.0*sumit)
            shf = -1.0*shf*1000.0
            hf_vec[i] = shf
    elif iuse_Hellinger83 == 1:
        con1 = k*T_bottom/L
        Beta_L = L*delta*Beta/(z_moho*Beta+(L-z_moho)*delta)
        for i, tMyr in enumerate(ts_Myr):
            ts = tMyr*1e6*sec_yr
            sumit = 0.0
            for j in range(n):
                nk = float(j+1)
                C1 = 2.0*(-1.0)**(nk+1.0)
                C2 = (
                        1.0/nk/nk/math.pi/math.pi*(
                            (delta-Beta)*math.sin(nk*math.pi*z_moho/L/delta)
                          + Beta*math.sin(nk*math.pi/Beta_L)
                         )
                    )
                C = C1*C2
                p1 = (-1.0)**(nk+1.0)
                fval = p1*nk*math.pi*C*math.exp(-nk*nk*ts/tau)
                sumit = sumit+fval
            shf = -con1*(1.0+sumit)*1000.0
            hf_vec[i]=shf
    return 1


@jit(nopython=True, cache=True)
def Analytical_Sub_v2(
                        nages, ts_Myr, subn, 
                        delta, Beta, L, kappa, k, T_bottom,
                        rho_m, rho_c, rho_w, alpha, sec_yr, 
                        z_moho, iup_in_air, itype_rho_a
):
    if itype_rho_a == 0:
            rho_a = rho_m*(1.0-alpha*T_bottom)
    else:
            rho_a = rho_m
    tau = L*L/kappa/math.pi/math.pi
    n = 100
    Sini = Analytical_Ini_Sub(
                                L, z_moho, rho_c, rho_m, rho_w, alpha,
                                T_bottom, delta, Beta, iup_in_air,
                                itype_rho_a
                            )
    if iup_in_air == 1:
        if Sini < 0.0:
            rho_w = 0.0
    iuse_Hellinger83 = 1
    iuse_McKenzie78 = 0
    if iuse_McKenzie78 == 1:
        con1 = 4.0/math.pi/math.pi*L*rho_m*alpha*T_bottom/(rho_a-rho_w)
        for i, tMyr in enumerate(ts_Myr):
            ts = tMyr*1e6*sec_yr
            sumit = 0.0
            for j in range(n):
                nf = float(j)
                p1 = 1.0/(2.0*nf+1.0)/(2.0*nf+1.0)
                p2 = (Beta/(2.0*nf+1.0)/math.pi*math.sin(
                                                    (2.0*nf+1.0)/Beta*math.pi))
                p3 = math.exp(-(2.0*nf+1.0)*(2.0*nf+1.0)*ts/tau)
                fval = p1*p2*p3
                sumit = sumit+fval
            e = con1*sumit
            if i == 0:
                e_ini = e
            sub = Sini + e_ini - e
            subn[i] = sub
    elif iuse_Hellinger83 == 1:
        con1 = 2.0*L*alpha*rho_m*T_bottom/(rho_a-rho_w)
        for i, tMyr in enumerate(ts_Myr):
            ts = tMyr*1e6*sec_yr
            sumit = 0.0
            for j in range(n):
                nf = float(j)
                nk = 2.0*nf+1
                Beta_L = L*delta*Beta/(z_moho*Beta+(L-z_moho)*delta)
                C1 = 2.0*(-1.0)**(nk+1.0)
                C2 = (
                        1.0/nk/nk/math.pi/math.pi/math.pi*(
                            (delta-Beta)*math.sin(nk*math.pi*z_moho/L/delta)
                            + Beta*math.sin(nk*math.pi/Beta_L)
                            )
                    )
                C = C1*C2
                fval = C/(2.0*nf+1.0)*math.exp(-(2.0*nf+1)*(2.0*nf+1)*ts/tau)
                sumit = sumit+fval
            e = con1*sumit
            if i == 0:
                e_ini = e
            sub = Sini + e_ini - e
            subn[i] = sub
    return 1


@jit(nopython=True, cache=True)
def Analytical_Ini_Sub(
                        L, z_moho, rho_c, rho_m, rho_w, alpha, 
                        T_bottom, delta, Beta, iup_in_air, itype_rho_a
):
    if itype_rho_a == 0:
            rho_a = rho_m*(1.0-alpha*T_bottom)
    else:
            rho_a = rho_m
    iuse_McKenzie78 = 0
    iuse_Hellinger83 = 1
    if iuse_McKenzie78 == 1:
        p1 = (rho_m-rho_c)*z_moho/L*(1.0-alpha*T_bottom*z_moho/L)
        p2 = alpha*T_bottom*rho_m/2.0
        p3 = rho_a-rho_w
        Sini = L*(p1-p2)*(1.0-1.0/Beta)/p3
    elif iuse_Hellinger83 == 1:
        icalc_again = 1
        for i in range(2):
            if icalc_again == 1:
                Beta_L = L*delta*Beta/(z_moho*Beta+(L-z_moho)*delta)
                gamma_L = 1.0-1.0/Beta_L
                gamma_c = 1.0-1.0/delta
                p1 = (rho_m-rho_c)*z_moho*(1.0-alpha*T_bottom*z_moho/2.0/L)
                p2 = alpha*rho_m*T_bottom*z_moho/2.0
                p2 = (p1-p2)*gamma_c
                p3 = (alpha*rho_m*T_bottom*(L-z_moho)/2.0)*gamma_L
                p4 = rho_a-rho_w
                Sini = (p2-p3)/p4
                if iup_in_air == 1:
                    if Sini < 0.0:
                        if rho_w > 0.0:
                            icalc_again = 1
                            rho_w = 0.0
                        else:
                            icalc_again = 0
                else:
                    icalc_again = 0
    return Sini


@jit(nopython=True, cache=True)
def Analytical_Final_Sub_v2(
                            L, z_moho, rho_crust, rho_mantle, 
                            rho_w, alpha, T_bottom, delta,
                            Beta, Tc_ini, Ts, zs,
                            xc, yc, xm, ym, nnodes, itype_rho_a
):
    if itype_rho_a == 0:
        rho_a = rho_mantle*(1.0-alpha*T_bottom)
    else:
        rho_a = rho_mantle
    # A correction was made to the equation in Watts (2001)
    iuse_Watts01 = 0
    # Also see Hellinger and Sclater (1980)
    iuse_Jarvis80 = 0
    iuse_Hellinger83 = 1
    if iuse_Watts01 == 1:
        alpha_c = alpha
        avg_Tc, avg_Tm = math_tools.get_avg_temps_v2(
                                                    Ts, zs, xc, yc, xm, ym, 
                                                    z_moho, L, nnodes
                                                )
        rho_co = rho_crust*(1.0-alpha_c*avg_Tc)
        rho_mo = rho_mantle*(1.0-alpha_c*avg_Tm)
        rho_cf = rho_crust*(1.0-alpha/2.0*T_bottom/Beta*z_moho/L)
        rho_mf = rho_mantle*(
                                  1.0
                                - alpha/2.0*T_bottom/L*z_moho/Beta
                                - alpha*T_bottom/2.0
                            )
        rho_a = rho_mantle
        p1 = L*(rho_mf-rho_mo) + z_moho*(rho_mo+rho_cf/Beta-rho_mf/Beta-rho_co)
        p2 = rho_a-rho_w
        Sf = p1/p2
        St = Sf
    elif iuse_Jarvis80 == 1: 
        # Use Jarvis and McKenzie, 1980; 
        # This is equivalent to Hellinger and Sclater, 1980
        p1 = (rho_mantle-rho_crust)*z_moho
        p2 = rho_a-rho_w
        p3 = (1.0-1.0/Beta)-alpha*T_bottom*z_moho/2.0/L*(1.0-1.0/Beta/Beta)
        St = p1/p2*p3
    elif iuse_Hellinger83 == 1:
        p1 = (rho_mantle-rho_crust)*z_moho
        p2 = 1.0 - alpha*T_bottom*z_moho*(1.0 + 1.0/delta)/2.0/L
        p3 = (1.0-1.0/delta)
        p4 = rho_a-rho_w
        St = p1*p2*p3/p4
    return St


def Analytical_Temp(zs, ts, delta, Beta, L, kappa, T_bottom, z_moho):
    n = 40
    iuse_McKenzie78 = 0
    iuse_Hellinger83 = 1
    tau = L*L/kappa/math.pi/math.pi
    Tn = []
    if iuse_McKenzie78 == 1:
        con1 = 2.0*T_bottom/math.pi
        for i, z in enumerate(zs):
            zb = L-z*1000.0
            Tss = T_bottom-T_bottom*zb/L
            sumit = 0.0
            for j in range(n):
                nf = float(j+1)
                p1 = ((-1.0)**(nf+1.0))/nf
                p2 = Beta/nf/math.pi*math.sin(nf*math.pi/Beta)
                p3 = math.exp(-nf*nf*ts/tau)*math.sin(nf*math.pi*zb/L)
                fval = con1*p1*p2*p3
                sumit = sumit + fval
            T = Tss + sumit
            Tn.append(T)
    elif iuse_Hellinger83 == 1:
        Beta_L = L*delta*Beta/(z_moho*Beta+(L-z_moho)*delta)
        for i, z in enumerate(zs):
            zb = L-z*1000.0
            Tss = T_bottom-T_bottom*zb/L
            sumit = 0.0
            for j in range(n):
                nk = float(j+1)
                C1 = 2.0*(-1.0)**(nk+1.0)
                C2 = (
                        1.0/nk/nk/math.pi/math.pi*(
                                            (delta-Beta)*math.sin(
                                             nk*math.pi*z_moho/L/delta)
                                           + Beta*math.sin(nk*math.pi/Beta_L)
                                          )
                    )
                C = C1*C2
                fval = C*math.sin(nk*math.pi*zb/L)*math.exp(-nk*nk*ts/tau)
                sumit = sumit+fval
            T = Tss + T_bottom*sumit
            Tn.append(T)
    return Tn