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
import numpy as np
import scipy.stats


@jit(nopython=True, cache=True)
def linear_interp_v2(nnodes, xp, data_xy):
    ilast = nnodes-1
    imin = 0
    dist_min = 1e32
    dx_min = 0.0
    for i in range(nnodes):
        x = data_xy[i][0]
        dx = xp-x
        dist = abs(dx)
        if dist < dist_min:
            imin = i
            dist_min = dist
            dx_min = dx
    if dx_min == 0.0: 
        # xp is directly on data point imin
        yp = data_xy[imin][1]
    if dx_min < 0.0: 
        # xp is to the left of data point imin
        if imin != 0:
            x1 = data_xy[imin-1][0]
            y1 = data_xy[imin-1][1]
            x2 = data_xy[imin][0]
            y2 = data_xy[imin][1]
            yp = (y2-y1)/(x2-x1)*(xp-x1)+y1
        else: 
            # Use a constant value before first data point
            yp = data_xy[imin][1]
    if dx_min > 0.0: 
        # xp is to the right of data point imin
        if imin != ilast:
            x1 = data_xy[imin][0]
            y1 = data_xy[imin][1]
            x2 = data_xy[imin+1][0]
            y2 = data_xy[imin+1][1]
            yp = (y2-y1)/(x2-x1)*(xp-x1)+y1
        else: 
            # Use a constant value past last data point
            yp = data_xy[imin][1]
    return yp


def linear_interp_python(xp, data_x, data_y):
    data_xy = []
    for i, x in enumerate(data_x):
        y = data_y[i]
        data_xy.append([x,y])
    ilast = len(data_xy) - 1
    imin = 0
    dist_min = 1e32
    dx_min = 0.0
    for i,xy in enumerate(data_xy):
        x = xy[0]
        y = xy[1]
        dx = xp-x
        dist = abs(dx)
        if dist < dist_min:
            imin = i
            dist_min = dist
            dx_min = dx
    if dx_min == 0.0: 
        # xp is directly on data point imin
        yp = data_xy[imin][1]
    if dx_min < 0.0: 
        # xp is to the left of data point imin
        if imin != 0:
            x1 = data_xy[imin-1][0]
            y1 = data_xy[imin-1][1]
            x2 = data_xy[imin][0]
            y2 = data_xy[imin][1]
            yp = (y2-y1)/(x2-x1)*(xp-x1)+y1
        else: 
            # Use a constant value before first data point
            yp = data_xy[imin][1]
    if dx_min > 0.0: 
        # xp is to the right of data point imin
        if imin != ilast:
            x1 = data_xy[imin][0]
            y1 = data_xy[imin][1]
            x2 = data_xy[imin+1][0]
            y2 = data_xy[imin+1][1]
            yp = (y2-y1)/(x2-x1)*(xp-x1)+y1
        else: 
            # Use a constant value past last data point
            yp = data_xy[imin][1]
    return yp


@jit(nopython=True, cache=True)
def Trapezoid_integrate_xy_v2(x, y, nnodes):
    int_sum = 0.0
    for i in range(1,nnodes):
        xp_1 = x[i-1]
        xp_2 = x[i]
        dx = xp_2-xp_1
        yp_1 = y[i-1]
        yp_2 = y[i]
        # Trapezoid Method
        if yp_1 > yp_2:
            area = dx*yp_2+dx*(yp_1-yp_2)*0.5
        else:
            area = dx*yp_1+dx*(yp_2-yp_1)*0.5
        int_sum = int_sum + area
    return int_sum


@jit(nopython=True, cache=True)
def get_avg_temps_v2(Ts, zs, xc, yc, xm, ym, z_moho, L, nnodes):
    z_moho_p = z_moho
    # Initialize arrays to zero before calculating averages
    for i in range(nnodes):
        xc[i]=0.0
        yc[i]=0.0
        xm[i]=0.0
        ym[i]=0.0
    # First do the crust
    icountc = 0
    for i in range(nnodes):
        T = Ts[i]
        z = zs[i]*1000.0
        if z < z_moho_p:
            xc[icountc] = z
            yc[icountc] = T
            icountc = icountc + 1
            ilast = i
    if icountc == 1:
        icountc = icountc + 1
        To = Ts[0]
        dT = Ts[1]-Ts[0]
        dz = zs[1]-zs[0]
        dT_dz = dT/dz
        T_moho = To + dT_dz*(z_moho_p/1000.0)
    else:
        To = Ts[ilast]
        dT = Ts[ilast+1]-Ts[ilast] 
        dz = zs[ilast+1]-zs[ilast]
        dT_dz = dT/dz
        T_moho = To + dT_dz*(z_moho_p/1000.0-zs[ilast])
    # Add moho node
    xc[icountc-1] = z_moho_p
    yc[icountc-1] = T_moho    
    # Now do the mantle
    icountm = 0
    xm[icountm] = z_moho_p
    ym[icountm] = T_moho
    icountm = 1    
    for i in range(nnodes):
        T = Ts[i]
        z = zs[i]*1000.0
        if z > z_moho_p:
            xm[icountm] = z
            ym[icountm] = T
            icountm = icountm + 1
    # Add Moho nodes
    int_sum = Trapezoid_integrate_xy_v2(xc, yc, icountc)
    avg_Tc = int_sum/(xc[icountc-1]-xc[0])
    int_sum = Trapezoid_integrate_xy_v2(xm, ym, icountm)    
    avg_Tm = int_sum/(xm[icountm-1]-xm[0])
    return avg_Tc, avg_Tm


def random_values_from_discrete(p1, p2, p3, nruns):
    xk = np.zeros((3))
    xk[0] = -1
    xk[1] = 0
    xk[2] = 1
    pk = np.zeros((3))
    pk[0] = p1
    pk[1] = p2
    pk[2] = p3    
    custm = scipy.stats.rv_discrete(name='custm', values=(xk, pk))
    vec = custm.rvs(size=nruns)
    return vec
    

def random_values_from_pdf(pdf_stype, a, b, c, vmin, vmax, nruns):
    # Run# vs variance or std
    # Triangular
    # Uniform
    # Guaisian
    # Beta: got it
    # Discrete: use other function
    scale_val = vmax - vmin
    loc_val = vmin
    if pdf_stype == "beta":
        vec = scipy.stats.beta.rvs(
                                    a, b, scale=scale_val, 
                                    loc=loc_val, size=nruns
                                )
    elif pdf_stype == "triangular":
        vec = scipy.stats.triang.rvs(c, scale=scale_val, loc=loc_val)
    elif pdf_stype == "uniform":
        vec = scipy.stats.uniform.rvs(scale=scale_val, loc=loc_val)
    else:
        vec = scipy.stats.beta.rvs(
                                    a, b, scale=scale_val, 
                                    loc=loc_val, size=nruns
                                )
    return vec


@jit(nopython=True, cache=True)
def inside_outside_poly(coords_poly, py, pz):
    # poly coordinates require that the first node equals the last node
    #
    #  z
    #  ^
    #  |
    #  |
    #  |-------> y
    #
    TOL2 = 1e-3
    TOL3 = 1e-3        
    bused_exact = False
    nexact = 0    
    ymin = 1e32
    ymax = -1e32
    zmin = 1e32
    zmax = -1e32    
    too_close = False
    # loop over all edges or point pairs
    nintersect = 0
    for j in range(0,len(coords_poly)-1):            
        # point j of polygon
        yn_1 = coords_poly[j][0]
        zn_1 = coords_poly[j][1]            
        # point j+1 of polygon
        yn_2 = coords_poly[j+1][0]
        zn_2 = coords_poly[j+1][1]
        if yn_1 < ymin:
            ymin = yn_1
        if yn_1 > ymax:
            ymax = yn_1
        if yn_2 < ymin:
            ymin = yn_2
        if yn_2 > ymax:
            ymax = yn_2       
        if zn_1 < zmin:
            zmin = zn_1
        if zn_1 > zmax:
            zmax = zn_1
        if zn_2 < zmin:
            zmin = zn_2
        if zn_2 > zmax:
            zmax = zn_2                
        falls_on_node = False        
        dy1 = abs(yn_1-py)
        dy2 = abs(yn_2-py)
        dz1 = abs(zn_1-pz)
        dz2 = abs(zn_2-pz)        
        if dy1 < TOL2 and dz1 < TOL2:
            falls_on_node = True
            nexact = nexact + 1            
        elif dy2 < TOL2 and dz2 < TOL2:            
            falls_on_node = True
            nexact = nexact + 1            
        if falls_on_node == True and bused_exact == False:
            nintersect = nintersect+1
            bused_exact = True        
        if falls_on_node == False:            
            if zn_1 != zn_2:                
                if zn_1 < zn_2:
                    zn_max = zn_2
                    zn_min = zn_1
                else:
                    zn_max = zn_1
                    zn_min = zn_2                
                if pz <= zn_max and pz > zn_min:                   
                    # get y-coordinate of interval
                    y_edge = (yn_2-yn_1)/(zn_2-zn_1)*(pz-zn_1)+yn_1                    
                    diff_check = abs(py-y_edge)                    
                    if diff_check < TOL3:                       
                        too_close = True
                        break                    
                    if py <= y_edge and zn_1 != zn_2:                                    
                        nintersect = nintersect+1
                    elif zn_1 == zn_2:                        
                        if abs(py-y_edge) < TOL3:
                            nintersect = nintersect+1               
    if too_close == False:        
        if nintersect == 1 and bused_exact == True:            
            inpoly = False            
        elif nintersect % 2 == 0 or nintersect == 0:                
            inpoly = False            
        else:            
            inpoly = True
    else:        
        inpoly = False
    if pz > zmax or pz < zmin:
        inpoly = False    
    if py > ymax or py < ymin:
        inpoly = False    
    return inpoly


@jit(nopython=True, cache=True)
def linear_interp_numba(xp, data_xy_np):
    ss = data_xy_np.shape
    ndata = ss[0]
    ilast = ndata-1
    imin = 0
    dist_min = 1e32
    dx_min = 0.0
    for i in range(ndata):        
        x = data_xy_np[i][0]
        dx = xp-x
        dist = abs(dx)
        if dist < dist_min:
            imin = i
            dist_min = dist
            dx_min = dx
    if dx_min == 0.0:
        # xp is directly on data point imin
        yp = data_xy_np[imin][1]
    if dx_min < 0.0: 
        # xp is to the left of data point imin
        if imin != 0:
            x1 = data_xy_np[imin-1][0]
            y1 = data_xy_np[imin-1][1]
            x2 = data_xy_np[imin][0]
            y2 = data_xy_np[imin][1]           
            ddx = x2-x1
            if ddx > 0:
                yp = (y2-y1)/ddx*(xp-x1)+y1
            else:
                yp = y1
        else: 
            # Use a constant value before first data point
            yp = data_xy_np[imin][1]
    if dx_min > 0.0: 
        # xp is to the right of data point imin
        if imin != ilast:
            x1 = data_xy_np[imin][0]
            y1 = data_xy_np[imin][1]
            x2 = data_xy_np[imin+1][0]
            y2 = data_xy_np[imin+1][1]
            ddx = x2-x1
            if ddx > 0:            
                yp = (y2-y1)/(x2-x1)*(xp-x1)+y1
            else:
                yp = y1                
        else: 
            # Use a constant value past last data point
            yp = data_xy_np[imin][1]
    return yp

    
    