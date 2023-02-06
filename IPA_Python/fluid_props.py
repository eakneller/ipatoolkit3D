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
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True, cache=True)
def calc_gas_zfactor(TF, Ppsi, gas_grav, oil_api, mol_frac_CO2, mol_frac_H2S):    
    Nmain = 5
    N = 10
    z_mino = 0.001
    z_maxo = 4
    dzo = (z_maxo - z_mino)/float(N)
    # Psuedocritical Temperature R (free gas) (eq 25)
    TR_cr = 169.2 + gas_grav*(349.5-74*gas_grav)
    #Psuedocritical Pressure psia free gas (eq 24)
    Ppsi_cr = 756.8 - gas_grav*(131+3.6*gas_grav)
    # McCain epsilon ( CO2,H2Sadjustment factor)( eq 30)
    epsilon = (
                120*(
                          (mol_frac_CO2 + mol_frac_H2S)**0.9 
                        + (mol_frac_CO2 + mol_frac_H2S)**1.6
                    )
                + (mol_frac_H2S**0.5 - mol_frac_H2S**4)
            )
    # Adjusted psuedocritical Temperature R (eq 28)
    TR_cr_adj = TR_cr - epsilon
    #Adjusted psuedocritcal Pressure (eq 29)
    Ppsi_cr_adj = (
                    Ppsi_cr*TR_cr_adj/
                                (
                                      TR_cr 
                                    + mol_frac_H2S*(1 - mol_frac_H2S)*epsilon
                                )
                )
    # 1/Psuedoreduced Temperature  1/eq 23a)
    Tred_inv = TR_cr_adj/(TF + 459.6)
    # Psuedoreduced Pressure (eq 23b)
    Pred = Ppsi/Ppsi_cr_adj
    zroot = 1e39
    for mm in range(Nmain):
        if mm == 0:
            z_min = z_mino
            z_max = z_maxo
            dz = dzo
        else:
            z_min = zroot - dz
            z_max = zroot + dz
            dz = (z_max - z_min)/float(N)         
        z_new_min = 1e39
        diff_min = 1e39
        for i in range(N):
            if i == 0:
                z_old = z_min
            else:
                z_old = z_old + dz
            # rhopr (eq 22)
            rhopr = 0.27*Pred*Tred_inv/z_old
            S41 = rhopr
            Q41 = Tred_inv    
            z_new = (
                      1
                      + S41*(
                                0.3265 
                              + Q41*(
                                      - 1.07 
                                      + Q41**2*(
                                                  - 0.5339 
                                                  + Q41*(0.01569-0.05165*Q41)
                                               )
                                    )
                            )
                      + S41**2*(0.5475+Q41*(-0.7361+0.1844*Q41))
                      - 0.1056*S41**5*Q41*(-0.7361+Q41*0.1844)
                      + 0.6134*Q41**3*S41**2*(1 + 0.721*S41**2)
                                                       *math.exp(-0.721*S41**2)
                    )
            diff = z_new - z_old
            if abs(diff) < diff_min:
                diff_min = diff
                z_new_min = z_new
        zroot = z_new_min
    return zroot


def calc_fluid_props(
                    TF, fpress_psi, sgrav_gas, Oil_API, GOR_scf_bbl,
                     sGOR_scf_bbl_max, iuse_sat_gor, iuse_sGOR_cor,
                     sGOR_at_charp, charp, exp_fac, a_gas, b_gas
):
    cm3_scf = 28316.85
    cm3_bbl = 158987.30
    rho_air_surface = 0.001292 # g/cm3    
    rho_gas_surface = rho_air_surface*sgrav_gas # g/cm3
    sgrav_oil = 141.5/(131.5 + Oil_API)
    rho_oil_surface = sgrav_oil # g/cm3
    if iuse_sGOR_cor == 1:
        sGOR_cor = sGOR_at_charp*((fpress_psi/charp)**exp_fac)
    else:
        sGOR_cor = 0.0
    sGORoil_scf_bbl = (
                        sgrav_gas*(
                                    (fpress_psi/18.2 + 1.4)*10**
                                                              (
                                                                0.0125*Oil_API 
                                                                - 0.00091*TF
                                                              )
                                    )**(1.0/0.83) 
                       + sGOR_cor
                    )
    if sGORoil_scf_bbl > sGOR_scf_bbl_max:
        sGORoil_scf_bbl = sGOR_scf_bbl_max        
    sGORoil_cm3_cm3 = sGORoil_scf_bbl*cm3_scf/cm3_bbl    
    sGORoil_g_g = sGORoil_cm3_cm3*(rho_gas_surface/rho_oil_surface)    
    zfactor = calc_gas_zfactor(TF, fpress_psi, sgrav_gas, Oil_API, 0, 0)        
    # Gas FVF (bg) rbbl/scf
    if fpress_psi == 0.0:
        fpress_psi = 0.001   
    GFVF_rbbl_scf = 0.00502*zfactor*(TF + 460.0)/fpress_psi
    GFVF_rcf_scf = GFVF_rbbl_scf**5.6145833333    
    # Reservoir gas density g/cc
    rho_rgas_g_cc = 0.21870617*(0.001/GFVF_rbbl_scf)*sgrav_gas
    if iuse_sat_gor == 1:
        GOR_tmp = sGORoil_scf_bbl
    else:
        GOR_tmp = GOR_scf_bbl    
    # Liquid FVF (bo) rbbl/stb
    LFVF_rbbl_stb = (
                        0.9759 
                      + 0.00012*
                                (
                                    GOR_tmp*math.sqrt(sgrav_gas/sgrav_oil) 
                                  + 1.25*TF
                                )**1.2
                    )
    # Reservoir fluid density g/cc        
    rho_rfluid_g_cc = (sgrav_oil + 0.0002179*GOR_tmp*sgrav_gas)/LFVF_rbbl_stb        
    #**************************************************************************
    # Calculate saturation GOR for gsa
    #**************************************************************************
    sGORgas_scf_bbl = (a_gas/fpress_psi)**(1.0/b_gas)   
    if sGORgas_scf_bbl > sGOR_scf_bbl_max:
        sGORgas_scf_bbl = sGOR_scf_bbl_max    
    sGORgas_cm3_cm3 = sGORgas_scf_bbl*cm3_scf/cm3_bbl
    sGORgas_g_g = sGORgas_cm3_cm3*(rho_gas_surface/rho_oil_surface)    
    sGOR_int_scf_bbl, fpress_int = calc_intersection_sat_history(
                                            TF, Oil_API, sgrav_gas, 
                                            rho_gas_surface, rho_oil_surface, 
                                            sGOR_at_charp, charp, exp_fac, 
                                            sGOR_scf_bbl_max, cm3_scf, cm3_bbl, 
                                            a_gas, b_gas, iuse_sGOR_cor
                                            )    
    sGOR_int_g_g = (
                    sGOR_int_scf_bbl*cm3_scf/cm3_bbl
                                            *(rho_gas_surface/rho_oil_surface)
                   )    
    return (
            zfactor, sGORoil_scf_bbl, sGORoil_g_g, 
            sGORgas_scf_bbl, sGORgas_g_g, GFVF_rbbl_scf, 
            GFVF_rcf_scf, rho_rgas_g_cc, LFVF_rbbl_stb, 
            rho_rfluid_g_cc, sGOR_int_scf_bbl, sGOR_int_g_g
           )

    
@jit(nopython=True, cache=True)
def calc_liquid_and_free_gas_mass(
                                    n, moil_g, mgas_g, 
                                    sGOR_oil_g_g, sGOR_gas_g_g
):    
    mDO = 0
    mDG = 0
    for mm in range(n):
        if moil_g > 0:
            mDG_max = sGOR_oil_g_g*(moil_g-mDO)
        else:
            mDG_max = 0
        if moil_g > 0:
            if mgas_g > mDG_max:
                mDG = mDG_max
            else:
                mDG = mgas_g
        else:
            mDG = 0
        if mgas_g > 0:
            mDO_max = (1/sGOR_gas_g_g)*(mgas_g-mDG)
        else:
            mDO_max = 0
        if mgas_g > 0:
            if moil_g > mDO_max:
                mDO = mDO_max
            else:
                mDO = moil_g
        else:
            mDO = 0
    mL_g = (moil_g-mDO)+mDG
    mFG_g = (mgas_g - mDG)+mDO
    return mDO, mDG, mL_g, mFG_g


@jit(nopython=True, cache=True)
def calc_intersection_trap_charge(
                                    TC_res, TC_surf, WD_m, res_depth_smm, 
                                    rho_sea, rho_brine,
                                    Oil_API, sgrav_gas, rho_gas_surface, 
                                    rho_oil_surface, sGOR_at_charp, charp,
                                    exp_fac, sGOR_scf_bbl_max, 
                                    cm3_scf, cm3_bbl, 
                                    a_gas, b_gas, iuse_sGOR_cor
):  
    if res_depth_smm > 0:
        C_m = (TC_res - TC_surf)/(res_depth_smm)
    else:
        C_m = 25.0/1000.0
    dz = 50 #meters
    z_max_m = 12000 # meters    
    n = int(z_max_m/dz)    
    diff_min = 1e32
    sGORint = -99999
    fpress_int = -99999
    for i in range(n):      
        z_smm = dz*float(i) # submud meters
        p_Pa = rho_sea*9.8*WD_m + rho_brine*9.8*z_smm
        p_psi = p_Pa/6894.76
        TC = TC_surf + C_m*z_smm    
        if i > 0:
            fpress_psi = p_psi
            TF = 9/5*TC+32
            (
                sGOR_oil_g_g, sGOR_oil_scf_bbl, 
                sGOR_gas_g_g, sGOR_gas_scf_bbl
            ) = calc_saturation_gor(
                                    fpress_psi, Oil_API, sgrav_gas, 
                                    rho_gas_surface, rho_oil_surface,
                                    TF, sGOR_at_charp, charp, exp_fac, 
                                    sGOR_scf_bbl_max, cm3_scf, cm3_bbl,
                                    a_gas, b_gas, iuse_sGOR_cor
                                    )
            diff = abs(sGOR_oil_scf_bbl - sGOR_gas_scf_bbl)
            if diff < diff_min:
                diff_min = diff
                sGORint = sGOR_oil_scf_bbl
                fpress_int = fpress_psi
    return sGORint, fpress_int


@jit(nopython=True, cache=True)
def calc_intersection_sat_history(
                                    TF, Oil_API, sgrav_gas, 
                                    rho_gas_surface, rho_oil_surface,
                                    sGOR_at_charp, charp, exp_fac, 
                                    sGOR_scf_bbl_max, cm3_scf, cm3_bbl, 
                                    a_gas, b_gas, iuse_sGOR_cor
):
    dpress_max_psi = 15000
    dfpress = 10
    n = int(dpress_max_psi/dfpress)
    diff_min = 1e32
    sGORint = -99999
    fpress_int = -99999
    for i in range(n):
        if i > 0:
            fpress_psi = float(i)*dfpress
            (
                sGOR_oil_g_g, sGOR_oil_scf_bbl, 
                sGOR_gas_g_g, sGOR_gas_scf_bbl
            ) = calc_saturation_gor(
                                fpress_psi, Oil_API, sgrav_gas, 
                                rho_gas_surface, rho_oil_surface,
                                TF, sGOR_at_charp, charp, exp_fac, 
                                sGOR_scf_bbl_max, cm3_scf, cm3_bbl,
                                a_gas, b_gas, iuse_sGOR_cor
                                )
            diff = abs(sGOR_oil_scf_bbl - sGOR_gas_scf_bbl)
            if diff < diff_min:
                diff_min = diff
                sGORint = sGOR_oil_scf_bbl
                fpress_int = fpress_psi
    return sGORint, fpress_int


@jit(nopython=True, cache=True)        
def calc_saturation_gor(
                        fpress_psi, Oil_API, sgrav_gas,
                        rho_gas_surface, rho_oil_surface,
                        TF, sGOR_at_charp, charp, exp_fac, 
                        sGOR_scf_bbl_max, cm3_scf, cm3_bbl,
                        a_gas, b_gas, iuse_sGOR_cor
): 
    # The empirical Black Oil saturation model of McCain (1991) is used as the
    # the basis of a parameterized saturation model as a function of 
    # temperature, pressure, oil API and gas gravity. This parameterization 
    # allows for easier exploration on uncertainty. Parameterization is
    # formulated using the following terms used in the equations below:
    #
    # charp: characteristic pressure (psi)
    # sGOR_at_charp: saturation GOR at characteristic pressure
    # exp_fac: exponential factor
    # a_gas: constant a used to define an approximate reference gas saturation
    # b_gas: constant b used to define an approximate reference gas saturation
    #
    # This parameterization approach can also be used to convert from McCain91
    # to the updated model of McCain (2011) using the following values:
    #
    # sGOR_at_charp = 380.0
    # charp = 4750.0
    # exp_fac = 2.4
    if iuse_sGOR_cor == 1:
        sGOR_cor = sGOR_at_charp*((fpress_psi/charp)**exp_fac)
    else:
        sGOR_cor = 0.0
    # calculate saturation GOR for oil
    sGOR_scf_bbl = (
                    sgrav_gas*(
                               (fpress_psi/18.2+1.4)*
                                           10**(0.0125*Oil_API-0.00091*TF)
                              )**(1.0/0.83) 
                    + sGOR_cor
                   )
    if sGOR_scf_bbl > sGOR_scf_bbl_max:
        sGOR_scf_bbl = sGOR_scf_bbl_max    
    sGOR_cm3_cm3 = sGOR_scf_bbl*cm3_scf/cm3_bbl    
    sGOR_oil_g_g = sGOR_cm3_cm3*(rho_gas_surface/rho_oil_surface)
    sGOR_oil_scf_bbl = sGOR_scf_bbl    
    # calculate saturation GOR for gas
    if fpress_psi <= 0:
        fpress_psi = 0.0001
    sGOR_scf_bbl = (a_gas/fpress_psi)**(1.0/b_gas)    
    if sGOR_scf_bbl > sGOR_scf_bbl_max:
        sGOR_scf_bbl = sGOR_scf_bbl_max    
    sGOR_cm3_cm3 = sGOR_scf_bbl*cm3_scf/cm3_bbl
    sGOR_gas_g_g = sGOR_cm3_cm3*(rho_gas_surface/rho_oil_surface)
    sGOR_gas_scf_bbl = sGOR_scf_bbl    
    return sGOR_oil_g_g, sGOR_oil_scf_bbl, sGOR_gas_g_g, sGOR_gas_scf_bbl

                
@jit(nopython=True, cache=True)
def calc_bubblepoint_pressure(TF, oil_api, gas_grav, lnRsb):
    C01	= -5.48
    C11	= -0.0378
    C21	= 0.281
    C31	= -0.0206
    C02	= 1.27
    C12	= -0.0449
    C22	= 4.36E-04
    C32	= -4.76E-06
    C03	= 4.51
    C13	= -10.84
    C23	= 8.39
    C33	= -2.34
    C04	= -0.7835
    C14	= 6.23E-03
    C24	= -1.22E-05
    C34	= 1.03E-08
    z1 = C01 + C11*lnRsb + C21*(lnRsb**2) + C31*(lnRsb**3)
    z2 = C02 + C12*oil_api + C22*(oil_api**2) + C32*(oil_api**3)
    z3 = C03 + C13*gas_grav + C23*(gas_grav**2) + C33*(gas_grav**3)
    z4 = C04 + C14*TF + C24*(TF**2) + C34*(TF**3)
    z = z1 + z2 + z3 + z4
    Pb = math.exp(7.475 + 0.713*z + 0.0075*(z**2))
    return Pb


@jit(nopython=True, cache=True)
def get_solution_GOR_loop(
                            n, Rsb_o, dRsb, TF, Pres_psi, 
                            oil_api, gas_grav,
                            data_PbRsb_oil_np, data_PbRsb_gas_np
):
    noil = 0
    ngas = 0
    for i in range(n):
        Rsb_1 = Rsb_o + float(i)*dRsb
        Rsb_2 = Rsb_1 + dRsb
        if Rsb_1 <= 0:
            Rsb_1 = 0.01
        if Rsb_2 <= 0:
            Rsb_2 = 0.01
        lnRsb_1 = math.log(Rsb_1)
        lnRsb_2 = math.log(Rsb_2)
        Pb_1 = calc_bubblepoint_pressure(TF, oil_api, gas_grav, lnRsb_1)
        Pb_2 = calc_bubblepoint_pressure(TF, oil_api, gas_grav, lnRsb_2)
        dPb_dRsb = (Pb_2-Pb_1)/dRsb
        if dPb_dRsb >= 0:
            data_PbRsb_oil_np[i][0] = Pb_1
            data_PbRsb_oil_np[i][1] = Rsb_1
            noil = noil + 1
        else:
            data_PbRsb_gas_np[i][0] = Pb_1
            data_PbRsb_gas_np[i][1] = Rsb_1
            ngas = ngas + 1
    Pb_oil_min = 1e32
    Pb_oil_max = 1e-32
    for i in range(noil):
        Pb_oil = data_PbRsb_oil_np[i][0]
        if Pb_oil > Pb_oil_max:
            Pb_oil_max = Pb_oil
        if Pb_oil < Pb_oil_min:
            Pb_oil_min = Pb_oil
    Pb_gas_min = 1e32
    Pb_gas_max = 1e-32
    for i in range(ngas):
        Pb_gas = data_PbRsb_gas_np[i][0]
        if Pb_gas > Pb_gas_max:
            Pb_gas_max = Pb_gas
        if Pb_gas < Pb_gas_min:
            Pb_gas_min = Pb_gas            
    Rsb_sat_oil = math_tools.linear_interp_numba(Pres_psi, data_PbRsb_oil_np)
    Rsb_sat_gas = math_tools.linear_interp_numba(Pres_psi, data_PbRsb_gas_np)
    return noil, ngas, Rsb_sat_oil, Rsb_sat_gas

  
def get_solution_GOR(TF, Pres_psi, oil_api, gas_grav):
    Rsb_o = 0 # scf/bbl
    Rsb_f = 1e6
    dRsb = 15
    n = int((Rsb_f - Rsb_o)/dRsb)
    data_PbRsb_oil_np = np.zeros((n,2))
    data_PbRsb_gas_np = np.zeros((n,2))
    (   noil, ngas, 
        Rsb_sat_oil, Rsb_sat_gas
    ) = get_solution_GOR_loop(
                                n, Rsb_o, dRsb, TF, Pres_psi,
                                oil_api, gas_grav, 
                                data_PbRsb_oil_np, data_PbRsb_gas_np
                                )
    iplot = 0
    if iplot == 1:
        Rsb_cut = 100
        xs_oil = []
        ys_oil = []
        for i in range(noil):
            Pb = data_PbRsb_oil_np[i][0]
            Rsb = data_PbRsb_oil_np[i][1]
            if Rsb >= Rsb_cut:
                xs_oil.append(math.log10(Rsb))
                ys_oil.append(Pb)
        xs_gas = []
        ys_gas = []
        for i in range(ngas):
            Pb = data_PbRsb_gas_np[i][0]
            Rsb = data_PbRsb_gas_np[i][1]
            if Rsb >= Rsb_cut:            
                xs_gas.append(math.log10(Rsb))
                ys_gas.append(Pb)
        fig, ax = plt.subplots()
        ax.plot(xs_oil, ys_oil, 'g')
        ax.plot(xs_gas, ys_gas, 'r')
        ax.set_ylim(12000, 0)  # decreasing time
        ax.set_xlim(2, 6)  # decreasing time
        ax.set_xlabel('Log10 Solution GOR (scf/bbl)')
        ax.set_ylabel("Bubble Point Pressure (psi)")
        plt.show()
    return Rsb_sat_oil, Rsb_sat_gas