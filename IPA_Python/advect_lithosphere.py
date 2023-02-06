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
import math
import math_tools
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def do_advect(
                z_moho, z_mantle, duz_moho, duz_mantle, 
                nnodes, T, Tn, zs, zsn, data_xy
):
    L_moho = z_moho
    L_mantle = z_mantle-z_moho
    for i in range(nnodes):
        z = zs[i]*1000.0
        if z <= z_moho:
            Tn[i] = T[i]
            zn = z + duz_moho/L_moho*z
            zsn[i] = zn/1000.0
        else:
            Tn[i] = T[i]
            zn = z + duz_mantle/L_mantle*(z-z_moho)+duz_moho
            zsn[i] = zn/1000.0
    for i in range(nnodes):
        x = zsn[i]
        y = Tn[i]
        data_xy[i][0]=x
        data_xy[i][1]=y
    # Interpolate back to original mesh
    for i in range(nnodes):
        z = zs[i]
        Tf = math_tools.linear_interp_v2(nnodes, z, data_xy)
        T[i] = Tf    


@jit(nopython=True, cache=True)
def advect_and_interpolate(
                                T, zs, z_moho, z_mantle, 
                                duz_moho, duz_mantle
):
    nnodes = zs.size
    Tn = np.zeros((nnodes))
    zsn = np.zeros((nnodes))
    data_xy = np.zeros((nnodes, 2))
    do_advect(
                z_moho, z_mantle, duz_moho, 
                duz_mantle, nnodes, T, Tn, 
                zs, zsn, data_xy
            )
    return T


def advect_boundaries(z_moho, L, dt_rift, ntime_rift, delta, Beta):
    L_moho = z_moho
    L_mantle = L-L_moho
    # Final thicknesses of crust and mantle
    L_moho_f = L_moho/delta
    L_mantle_f = L_mantle/Beta
    # Total change in thickness (i.e. displacement)
    uzt_moho = L_moho_f-L_moho
    uzt_mantle = L_mantle_f-L_mantle
    duz_moho = uzt_moho/float(ntime_rift)
    duz_mantle = uzt_mantle/float(ntime_rift)
    uz_moho_tot = 0.0
    uz_mantle_tot = 0.0
    moho_loc = [z_moho]
    mantle_loc = [L]
    rift_times = [0.0]
    for i in range(ntime_rift):
        tMa = float(i+1)*dt_rift
        rift_times.append(tMa)
        uz_moho_tot = uz_moho_tot + duz_moho
        loc1 = z_moho + uz_moho_tot
        moho_loc.append(loc1)
        uz_mantle_tot = uz_mantle_tot + duz_mantle
        loc2 = loc1 + L-z_moho + uz_mantle_tot
        mantle_loc.append(loc2)
    return rift_times, moho_loc, mantle_loc, duz_moho, duz_mantle


def advect_boundaries_v2(z_moho, L, dt_rift, ntime_rift, delta, Beta):
    t_rift = dt_rift*float(ntime_rift)
    # Initialize rift lists
    moho_loc = [z_moho]
    duz_moho = [0.0]
    mantle_loc = [L]
    duz_mantle = [0.0]
    rwidth_c = [1.0]
    rwidth_m = [1.0]
    rift_times = [0.0]
    delta_i = [1.0]
    Beta_i = [1.0]
    # Calculate delta_i (crust)
    for i in range(ntime_rift):
        j = i+1
        tMa = float(j)*dt_rift
        rift_times.append(tMa)
        delta_n = 1.0 + 1.0/rwidth_c[i]*(delta-1.0)*dt_rift/t_rift
        rwidth_c_n = rwidth_c[i]*delta_n
        Beta_n = 1.0 + 1.0/rwidth_m[i]*(Beta-1.0)*dt_rift/t_rift
        rwidth_m_n = rwidth_m[i]*Beta_n
        delta_i.append(delta_n)
        rwidth_c.append(rwidth_c_n)
        Beta_i.append(Beta_n)
        rwidth_m.append(rwidth_m_n)
        Tco = moho_loc[i]
        Tmo = mantle_loc[i]-moho_loc[i]
        loc1 = Tco/delta_n
        moho_loc.append(loc1)
        duz_c = Tco/delta_n - Tco
        duz_moho.append(duz_c)
        loc2 = loc1 + Tmo/Beta_n
        mantle_loc.append(loc2)
        duz_m = Tmo/Beta_n-Tmo
        duz_mantle.append(duz_m)
    return rift_times, moho_loc, mantle_loc, duz_moho, duz_mantle


@jit(nopython=True, cache=True)
def advect_boundaries_Karner97(
                                z_moho, L, dt_rift, ntime_rift, 
                                delta, Beta, ntime_riftp_max
):
    t_rift = dt_rift*float(ntime_rift)
    # Initialize rift lists
    #moho_loc = [z_moho]
    moho_loc = np.zeros((ntime_riftp_max))
    moho_loc[0] = z_moho
    #duz_moho = [0.0]
    duz_moho = np.zeros((ntime_riftp_max))
    mantle_loc = np.zeros((ntime_riftp_max)) # = [L]
    mantle_loc[0] = L
    duz_mantle = np.zeros((ntime_riftp_max)) # = [0.0]
    #rwidth_c = [1.0]
    #rwidth_m = [1.0]
    rift_times  = np.zeros((ntime_riftp_max)) # = [0.0]
    delta_i = np.zeros((ntime_riftp_max)) # = [1.0]
    delta_i[0] =  1.0
    Beta_i = np.zeros((ntime_riftp_max)) # = [1.0]
    Beta_i[0] = 1.0
    # Calculate delta_i (crust)
    for i in range(ntime_rift + 1):
        j = i + 1
        tMa = float(j)*dt_rift
        rift_times[j] = tMa # .append(tMa)
        delta_n = 1.0 + (delta - 1.0)*float(j)*dt_rift/t_rift
        Beta_n = 1.0 + (Beta - 1.0)*float(j)*dt_rift/t_rift
        delta_i[j] = delta_n #.append(delta_n)
        Beta_i[j] = Beta_n #.append(Beta_n)
    for i in range(ntime_rift + 1):
        j = i + 1
        tMa = float(j)*dt_rift
        Tco = moho_loc[i]
        Tmo = mantle_loc[i]-moho_loc[i]
        delta_n = delta_i[j]/delta_i[j-1]
        Beta_n = Beta_i[j]/Beta_i[j-1]
        loc1 = Tco/delta_n
        moho_loc[j] = loc1 # .append(loc1)
        duz_c = Tco/delta_n - Tco
        duz_moho[j] = duz_c # .append(duz_c)
        loc2 = loc1 + Tmo/Beta_n
        mantle_loc[j] = loc2 #.append(loc2)
        duz_m = Tmo/Beta_n-Tmo
        duz_mantle[j] = duz_m #.append(duz_m)
    return rift_times, moho_loc, mantle_loc, duz_moho, duz_mantle


@jit(nopython=True, cache=True)
def advect_boundaries_Jarvis80(
                                z_moho, L, dt_rift, ntime_rift, 
                                delta, Beta, ntime_riftp_max
):
    t_rift = dt_rift*float(ntime_rift)
    # Initialize rift lists
    #moho_loc = [z_moho]
    moho_loc = np.zeros((ntime_riftp_max))
    moho_loc[0] = z_moho
    duz_moho = np.zeros((ntime_riftp_max)) # = [0.0]
    mantle_loc = np.zeros((ntime_riftp_max)) # = [L]
    mantle_loc[0] = L
    duz_mantle = np.zeros((ntime_riftp_max)) # = [0.0]
    #rwidth_c = [1.0]
    #rwidth_m = [1.0]
    rift_times = np.zeros((ntime_riftp_max)) # = [0.0]
    delta_i = np.zeros((ntime_riftp_max)) # = [1.0]
    delta_i[0] = 1.0
    Beta_i = np.zeros((ntime_riftp_max)) # = [1.0]
    Beta_i[0] = 1.0
    Gc = math.log(delta)/t_rift
    Gm = math.log(Beta)/t_rift
    # Calculate delta_i (crust)
    for i in range(ntime_rift + 1):
        j = i + 1
        tMa = float(j)*dt_rift
        rift_times[j] = tMa #.append(tMa)
        delta_n = math.exp(Gc*float(j)*dt_rift)
        Beta_n = math.exp(Gm*float(j)*dt_rift)
        delta_i[j] = delta_n #.append(delta_n)
        Beta_i[j] = Beta_n #.append(Beta_n)
    for i in range(ntime_rift + 1):
        j = i + 1
        tMa = float(j)*dt_rift
        Tco = moho_loc[i]
        Tmo = mantle_loc[i] - moho_loc[i]
        delta_n = delta_i[j]/delta_i[j-1]
        Beta_n = Beta_i[j]/Beta_i[j-1]
        loc1 = Tco/delta_n
        moho_loc[j] = loc1 #.append(loc1)
        duz_c = Tco/delta_n - Tco
        duz_moho[j] = duz_c #.append(duz_c)
        loc2 = loc1 + Tmo/Beta_n
        mantle_loc[j] = loc2 #.append(loc2)
        duz_m = Tmo/Beta_n - Tmo
        duz_mantle[j] = duz_m #.append(duz_m)
    return rift_times, moho_loc, mantle_loc, duz_moho, duz_mantle
