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
from numba import jit


@jit(nopython=True, cache=True)
def steady_state_nlayer_v3(
                            idebug_out, nnodes, T_top, q_bottom, 
                            zs, k_zs, Q_zs, dzi, Hxdzi, Qbi, T_base, Tc_ss
                        ):
    """ Calculate steady-state 1D temperature for n-layers of sediment
    
    Layers are defined by nnodes stratigraphic tops. Computational elements are
    defined by the mid-points of stratigraphic tops and are used to calculate
    bulk average properties that take sublayering and variable lithology into 
    account. There are nnodes computaiton elements. Temperatures are calculated 
    at the base of computational elements and interpolated to stratigraphic 
    tops. 
    
    Input Description
    -----------------
    nnodes : integer
        number of nodes describing the tops of sedimentary layers
        
    T_top : float
        Temperature in Celcius at the top of the sediment package
    
    q_bottom : float
        bottom heat-flow boundary condition in W/m^2 (negative points upward)
        
    zs : numpy float array with dimensions (nnodes) 
        This array defines the depth of stratigraphic tops in kilometers.
    
    k_zs : numpy float array with dimensions (nnodes) 
        This array defines the harmonic average of thermal conductivity for 
        elements defined by the mid-points between stratigraphic tops.
    
    Q_zs : numpy float array with dimensions (nnodes) 
        This array defines the radiogenic heat production at tops in W/m^3 and
        is used to approximate the geometric average radiogenic heat production
        of elements where elements are defined by the mid-points of 
        stratigraphic tops.
    
    dzi: numpy float array with dimensions (nnodes)
        Thsi array is initialized with zeros and filled by this function
        with the thickness of elements in meters where elements are defined
        by the mid-points of stratigraphic tops.
    
    Hxdzi: numpy float array with dimensions (nnodes)
        This array is initialized with zeros and filled by this function
        with the product of dzi and Q_zs the radiogenic heat production at top
        locations
    
    Qbi : numpy float array with dimensions (nnodes)
        This array is initialized with zeros and filled by this function
        with the sum of Hxdzi for all elements deeper than the current element
        plus the absolute value of the bottom heat flow boundary condition
        
    T_base : numpy float array with dimensions (nnodes)
        Steady-state temperature at the base of elements
        
    Tc_ss : numpy float array with dimensions (nnodes)
        Steady-state temperature interpolated at stratigraphic tops.   
        
    """
    # zs depths needs to be in km
    # all other inputs in standard SI units
    # Calculate dzi: Thickness of layers assuming harmonic 
    #                averaging of conductivities
    # Calculate heat production x layer thickness
    ilast = nnodes - 1
    for i in range(nnodes):
        if i == 0:
            zc = zs[i]
            zl = zs[i + 1]
            dz = (zl - zc)/2.0
        elif i == ilast:
            zu = zs[i - 1]
            zc = zs[i]
            dz = (zc - zu)/2.0
        else:
            zu = zs[i - 1]
            zc = zs[i]
            zl = zs[i + 1]
            dz = (zc - zu)/2.0 + (zl - zc)/2.0
        dzi[i]= dz*1000.0
        val = Q_zs[i]*dz*1000.0
        Hxdzi[i] = val
    for i in range(nnodes):
        #if i < ilast:
        sumit = 0.0
        for j, val in enumerate(Hxdzi):
            if j >= i: # j > i                   
                sumit = sumit + val
        sumit = -q_bottom + sumit
        #else:
        #    sumit = -q_bottom
        Qbi[i] = sumit
    for i in range(nnodes):
        dz = dzi[i]
        Qb = Qbi[i]
        H = Q_zs[i]
        k = k_zs[i]
        z = dz
        if k == 0.0:
            k = 2.0  
        if i == 0:
            Tn = T_top + (Qb)*z/k - H/2.0/k*z*z
        else:
            Tn = T_base[i-1] + (Qb)*z/k - H/2.0/k*z*z
        T_base[i] = Tn
    iuse_linear = 1
    for i in range(nnodes):
        zc = zs[i]
        if i == 0:
            zl = zs[i+1]
            dz = (zl-zc)/2.0
            z_elem_top = zc
            z_elem_bottom = z_elem_top + dz
        elif i == ilast:
            zu = zs[i-1]
            dz = (zc-zu)/2.0
            z_elem_top = zc - dz
            z_elem_bottom = zc
        else:
            zu = zs[i-1]
            zl = zs[i+1]
            dz = (zc-zu)/2.0 + (zl-zc)/2.0
            z_elem_top = zc - (zc-zu)/2.0
            z_elem_bottom = zc + (zl-zc)/2.0
        dz = dzi[i]
        Qb = Qbi[i]
        H = Q_zs[i]
        k = k_zs[i]
        z = dz/2.0
        if k == 0.0:
            k = 2.0               
        if i == 0:
            Tn = T_top            
        elif i == ilast:           
            z = dz                
            Tn = T_base[i-1] + (Qb)*z/k - H/2.0/k*z*z
        else:
            if iuse_linear == 0:
                Tn = T_base[i-1] + (Qb)*z/k - H/2.0/k*z*z
            else:             
                dT = T_base[i] - T_base[i-1]
                dzz = z_elem_bottom - z_elem_top
                if dzz > 0.0:
                    Tn = T_base[i-1] + (dT)/(dzz)*(zc - z_elem_top)
                else:
                    Tn = T_base[i-1]
        Tc_ss[i] = Tn
        
        
@jit(nopython=True, cache=True)
def steady_state_nlayer_TempBCs_v2(
                                    nnodes, T_top, T_bottom, 
                                    zs, k_zs, Q_zs, Tc_ss, dzi
):
    nlayers = nnodes
    # Calculate dzi: Thickness of layers assuming 
    # harmonic averaging of conductivities   
    ilast = nnodes - 1
    Ltot = abs(zs[ilast]-zs[0])*1000.0
    for i in range(nnodes):
        if i == 0:
            dz = abs(zs[1]-zs[0])/2*1000
        elif i == ilast:
            dz = abs(zs[ilast]-zs[ilast-1])/2*1000
        else:
            dz = abs(zs[i]-zs[i+1])*1000
        dzi[i]=dz
    # Calculate average conductivity using a harmonic average
    sumit = 0.0
    for i in range(nnodes):
        sumit = sumit + dzi[i]/Ltot/k_zs[i]
    kavg = 1.0/sumit
    # Calculate the constant G
    sumit1 = 0.0
    for i in range(nlayers):
        sumit2 = 0.0
        for j in range(i): 
            # Loop over all layers except current layer
            cf = k_zs[j]/k_zs[i]
            term1 = Q_zs[j]*dzi[j]*dzi[i]/k_zs[j]
            sumit2 = sumit2 + term1*cf
        Hterm = Q_zs[i]/2.0/k_zs[i]*dzi[i]*dzi[i]
        sumit1 = sumit1 + sumit2 + Hterm
#        if i == 0:
#            fff = 1.0
#        elif i == ilast:
#            fff = 1.0
#        else:
#            fff = 2.0
    G = kavg/Ltot*(T_bottom - T_top + sumit1)
    # Calculate the temperature at nodes associated with each layer 
    ilast = nlayers - 1
    f = 2.0
    for m in range (nlayers):
        if m > 0:
            sumit1 = 0.0
            sumit3 = 0.0
            nlayer_last = m+1 # 
            # Integration of B(z) from 0 to middle of layer nlayer_last-1            
            for i in range(nlayer_last): 
                # Loop over all layers up to current layer m
                if i < ilast: 
                    # If we are not at the last layer use 
                    # f to caulcate at middle of layer
                    f = 2.0
                else:
                    # If we are at the last layer use f to 
                    # caulctae integral at last node
                    f = 1.0
                sumit2 = 0.0
                for j in range(i): 
                    # i, Loop over all layers except current 
                    # layer and calculate heat production sum
                    cf = k_zs[j]/k_zs[i]
                    if i < m:
                        sumit2 = sumit2 + Q_zs[j]*dzi[j]*dzi[i]/k_zs[j]*cf
                    elif i == m:
                        sumit2 = sumit2 + Q_zs[j]*dzi[j]*dzi[i]/f/k_zs[j]*cf
                sumit1 = sumit1 + sumit2
                # Calculate the other heat production term for 
                # each layer and add it to sumit1
                if i < m: 
                    # If we are not the current layer m...
                    Hterm = Q_zs[i]/2.0/k_zs[i]*dzi[i]*dzi[i]
                    sumit1 = sumit1 + Hterm
                    sumit3 = sumit3 + dzi[i]/k_zs[i]
                elif i == m: 
                    # If we are at the current layer m...
                    Hterm = Q_zs[i]/2.0/k_zs[i]*dzi[i]/f*dzi[i]/f
                    sumit1 = sumit1 + Hterm
                    sumit3 = sumit3 + dzi[i]/f/k_zs[i]
            Tn = T_top - sumit1 + G*sumit3
        else:
            Tn = T_top
        Tc_ss[m]=Tn
    return zs


@jit(nopython=True, cache=True)
def linear_geotherm_v3(
                        L, z_moho, zs, T, nnodes, T_top, 
                        T_bottom, k_crust, k_mantle
):
    # m
    Lc = z_moho
    # m
    Lm = L-z_moho
    # W/m/m
    q_ss = k_crust*k_mantle/(k_crust*Lm+k_mantle*Lc)*(T_bottom-T_top)
    # T/km
    dT_dz_1 = q_ss/k_crust*1000.0
    # kmv
    z_moho = z_moho/1000.0
    # C
    T_moho = dT_dz_1*z_moho
    # T/km
    dT_dz_2 = q_ss/k_mantle*1000.0
    for i in range(nnodes):
        z = zs[i]
        # Initialize temperature
        if z <= z_moho:
            Tn = T_top+dT_dz_1*z
        else:
            Tn = T_moho+dT_dz_2*(z-z_moho)
        T[i]=Tn
        
        