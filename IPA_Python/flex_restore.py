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
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from numba import jit


@jit(nopython=True)
def apply_taper_method2(
                        xn, yn, dx, dy, xnf, ynf, 
                        xn_exp, yn_exp, Flex_filling_AOI, 
                        Flex_q_sed, Flex_q_uc, 
                        Flex_q_lc, Flex_q_thermal, Flex_q_BL, Flex_D, 
                        q_sed, q_uc, q_lc, q_thermal, q_BL,
                        D, Dtaper
):
    xmin_s = 0.0
    ymin_s = 0.0
    xmin_m = xmin_s - dx*xnf
    ymin_m = ymin_s - dy*xnf
    # Loop over main grid
    for i in range(xn_exp):
        for j in range(yn_exp):
            xm = xmin_m + dx*i
            ym = ymin_m + dy*j
            check = Flex_filling_AOI[i,j]
            if check == 0: 
                # Skip nodes that have original values
                # Loop over small grid and find closest small grid node
                ii_c = -9999
                jj_c = -9999
                dist_c = 1e32
                for ii in range(xn):
                    for jj in range(yn):
                        xs = xmin_s + dx*ii
                        ys = ymin_s + dy*jj
                        ddx = xs - xm
                        ddy = ys - ym
                        dist = math.sqrt(ddx*ddx + ddy*ddy)
                        if dist < dist_c:
                            dist_c = dist
                            ii_c = ii
                            jj_c = jj
                # Calculate taper factor
                if dist_c < Dtaper:
                    dfac = dist_c/Dtaper
                else:
                    dfac = 1.0
                theta = math.pi*dfac
                fac = (math.cos(theta)+1.0)/2.0
                # Apply taper factor
                Flex_q_sed[i,j]=q_sed[ii_c, jj_c]*fac
                Flex_q_uc[i,j]=q_uc[ii_c, jj_c]*fac
                Flex_q_lc[i,j]=q_lc[ii_c, jj_c]*fac
                Flex_q_thermal[i,j]=q_thermal[ii_c, jj_c]*fac
                Flex_q_BL[i,j]=q_BL[ii_c, jj_c]*fac
                Flex_D[i,j]=D[ii_c, jj_c]*fac

            
@jit(nopython=True)
def get_number_gt_zero(sgrid, xn, yn):
    n = 0
    for i in range(xn):
        for j in range(yn):
            sval = sgrid[i][j]
            if sval > 0:
                n = n + 1
    return n


@jit(nopython=True)
def calc_nz_avg(vec):  
    sumit = 0.0
    nval_nz = 0
    N = len(vec)
    for i in range(N):
        if vec[i] > 0.0:
            sumit = sumit + vec[i]
            nval_nz = nval_nz + 1
    if nval_nz > 0:
        vavg = sumit/float(nval_nz)
    else:
        vavg = 0
    return vavg

@jit(nopython=True)
def calc_nz_avg_v2(vec, nlimit):
    sumit = 0.0
    nval_nz = 0
    for i in range(nlimit):
        if vec[i] > 0.0:
            sumit = sumit + vec[i]
            nval_nz = nval_nz + 1
    if nval_nz > 0:
        vavg = sumit/float(nval_nz)
    else:
        vavg = 0
    return vavg


@jit(nopython=True)
def calc_avg_2Dgrid_omitnan(sgrid, xn, yn):
    sumit = 0.0
    nval = 0
    for i in range(xn):
        for j in range(yn):
            v = sgrid[i][j]
            bcheck = np.isnan(v)
            if bcheck == 0:
                sumit = sumit + v
                nval = nval + 1
    if nval > 0:
        savg = sumit/float(nval)
    else:
        savg = 0
    return savg


@jit(nopython=True)
def replace_nan_with_avg(sgrid, xn, yn, savg):
    for i in range(xn):
        for j in range(yn):
            v = sgrid[i][j]
            bcheck = np.isnan(v)
            if bcheck == 1:
                sgrid[i][j] = savg


@jit(nopython=True)
def global_ind(i, j, yn):
    ij = yn*(i)+j
    return ij


@jit(nopython=True)
def find_neighbor(ic, jc, xn, yn, neighbor_list):
    low_i = ic-1
    if low_i < 0: low_i = 0
    high_i = ic + 1
    if high_i > xn-1: high_i = xn-1
    low_j = jc-1
    if low_j < 0: low_j = 0
    high_j = jc + 1
    if high_j > yn-1: high_j = yn-1
    nnn = 0
    k = 0
    for i in range(low_i,high_i+1):
        for j in range(low_j,high_j+1):
            check_sum = abs(i-ic) + abs(j-jc)
            if check_sum > 0:
                neighbor_list[k][0] = i
                neighbor_list[k][1] = j
                nnn = nnn + 1
                k = k + 1
    return nnn


def print_grid(sgrid, xi, xf, yi, yf):
    for i in range(xi,xf):
        for j in range(yi,yf):
            val = sgrid[i][j]
            print("i, j, val: ", i, j, val)
            

@jit(nopython=True)
def apply_taper(
                ic, jc, xn_exp, yn_exp, n_empty, 
                neighbor_list, neighbor_vals,
                temp_neighbor_list, Flex_filling_AOI,
                Flex_q_sed, Flex_q_uc, Flex_q_lc, 
                Flex_q_thermal, Flex_q_BL, Flex_rho_m,
                taper_fac_mat, AOI_val_list, 
                q_sed_val_list, q_uc_val_list,
                q_lc_val_list, q_thermal_val_list, 
                q_BL_val_list
):
    #**************************************************************************
    # Flex_filling_AOI is used to determine which external cells are filled
    # with interpolated values and which are not. The taper algorithm continues 
    # untill all external cells are filled.
    #**************************************************************************
    count = 1
    icheck = 1
    while icheck == 1:
        #********************************
        # Find neighbors of current cell.
        #********************************
        nnn =  find_neighbor(ic,jc, xn_exp, yn_exp, neighbor_list)
        for k in range(nnn):
            i = neighbor_list[k][0]
            j = neighbor_list[k][1]
            AOI_val_list[k] = Flex_filling_AOI[i][j]
            q_sed_val_list[k] = Flex_q_sed[i][j]
            q_uc_val_list[k] = Flex_q_uc[i][j]
            q_lc_val_list[k] = Flex_q_lc[i][j]
            q_thermal_val_list[k] = Flex_q_thermal[i][j]
            q_BL_val_list[k] = Flex_q_BL[i][j]
        vavg1 = calc_nz_avg_v2(q_sed_val_list, nnn)
        Flex_q_sed[ic,jc] = taper_fac_mat[ic,jc]*vavg1
        vavg2 = calc_nz_avg_v2(q_uc_val_list, nnn)
        Flex_q_uc[ic,jc] = taper_fac_mat[ic,jc]*vavg2
        vavg2B = calc_nz_avg_v2(q_lc_val_list, nnn)
        Flex_q_lc[ic,jc] = taper_fac_mat[ic,jc]*vavg2B
        vavg3 = calc_nz_avg_v2(q_thermal_val_list, nnn)
        Flex_q_thermal[ic,jc] = taper_fac_mat[ic,jc]*vavg3
        vavg4 = calc_nz_avg_v2(q_BL_val_list, nnn)
        Flex_q_BL[ic,jc] = taper_fac_mat[ic,jc]*vavg4        
        val =  0.5*(n_empty-count)/n_empty + 0.25
        Flex_filling_AOI[ic,jc] = val        
        count = count + 1
        #**********************************************************************
        # Find empty neighbor cell with most borders of values.
        #**********************************************************************
        nlist = nnn
        for k in range(8):
            neighbor_vals[k] = 0.0
        for k in range(nnn): # k = 1:length(neighbor_list(:,1))
            if AOI_val_list[k] == 0:
                i = neighbor_list[k][0] 
                j = neighbor_list[k][1]
                nnn_tmp =  find_neighbor(
                                            i, j, xn_exp, yn_exp, 
                                            temp_neighbor_list
                                        )
                neighbor_sum = 0.0
                for l in range(nnn_tmp):
                    inn = temp_neighbor_list[l][0]
                    jnn = temp_neighbor_list[l][1]
                    neighbor_sum = neighbor_sum + Flex_filling_AOI[inn][jnn]
                neighbor_vals[k] = neighbor_sum
        vmax = -1e32
        ind_max = -9999
        for k in range(nlist):
            val = neighbor_vals[k]
            if val > vmax:
                vmax = val
                ind_max = k
        nn = ind_max
        sum_check = abs(ic-neighbor_list[nn][0])+abs(jc-neighbor_list[nn][1]) 
        sum_check2 = 0
        for k in range(nlist): 
            sum_check2 = sum_check2 + neighbor_vals[k]
        # if no more empty cell neighbors, 
        # scan through matrix until you find one.
        if sum_check > 0 and sum_check2 > 0:
            ic = neighbor_list[nn][0]
            jc = neighbor_list[nn][1]
        else:
            flag = 0
            for i in range(xn_exp):
                for j in range(yn_exp):
                    if Flex_filling_AOI[i][j] > 0 and flag == 0:
                        if i > 0:
                            if Flex_filling_AOI[i-1][j] == 0:    
                                ic = i-1
                                jc = j
                                flag = 1
                            elif i < xn_exp-1:
                                if Flex_filling_AOI[i+1][j] == 0:
                                    ic = i+1
                                    jc = j
                                    flag = 1
                        elif j > 0:
                            if Flex_filling_AOI[i][j-1] == 0:
                                ic = i
                                jc = j-1
                                flag = 1
                            elif j < yn_exp-1:
                                if Flex_filling_AOI[i][j+1] == 0:
                                    ic = i
                                    jc = j+1
                                    flag = 1
                    if flag == 1:
                        break
                if flag == 1:
                    break
            if flag == 0 and i != xn_exp-1 and j != yn_exp-1:
                break
        n_gt_zero = get_number_gt_zero(Flex_filling_AOI, xn_exp, yn_exp)
        if n_gt_zero < xn_exp*yn_exp:
            icheck = 1
        else:
            icheck = 0


@jit(nopython=True)
def build_system_full(
                        xn, yn, global_index, D, 
                        dx, dy, pr, g, q_sed, q_uc,
                        q_lc, q_thermal, q_BL, rho_m,
                        rho_c, rho_w, A, b
):
    k = 0
    for i in range(xn):
        for j in range(yn):
            i_j = global_ind(i,j, yn)
            im2_j = global_ind(i-2,j, yn)
            im1_j = global_ind(i-1,j, yn)
            i_jm2 = global_ind(i,j-2, yn)
            i_jm1 = global_ind(i,j-1, yn)            
            ip2_j = global_ind(i+2,j, yn)
            ip1_j = global_ind(i+1,j, yn)
            i_jp2 = global_ind(i,j+2, yn)
            i_jp1 = global_ind(i,j+1, yn)            
            im1_jm1 = global_ind(i-1,j-1, yn)
            im1_jp1 = global_ind(i-1,j+1, yn)
            ip1_jm1 = global_ind(i+1,j-1, yn)
            ip1_jp1 = global_ind(i+1,j+1, yn)            
            global_index[k,0] = i
            global_index[k,1] = j
            # coefficients for A matrix.
            Ax = D[i,j]/(dx**4)
            Ay = D[i,j]/(dy**4)
            B = (2*D[i,j])/((dx**2)*(dy**2))
            # Cx, Dx, Ex, and Fx need x boundaries.
            if i > 0 and i < xn-1:
                Cx = (D[i+1,j]-D[i-1,j])/(2*(dx**4))
                Dx = (D[i+1,j]-(2*D[i,j])+D[i-1,j])/(dx**4)
                Ex = pr*(D[i+1,j]-(2*D[i,j])+D[i-1,j])/((dx**2)*(dy**2))
                Fx = (D[i+1,j]-D[i-1,j])/(2*(dx**2)*(dy**2))
            else:
                Cx = 0
                Dx = 0
                Ex = 0
                Fx = 0
            # Cy, Dy, Ey, and Fy need y boundaries.
            if j > 0 and j < yn-1:
                Cy = (D[i,j+1]-D[i,j-1])/(2*(dy**4))
                Dy = (D[i,j+1]-(2*D[i,j])+D[i,j-1])/(dy**4)
                Ey = pr*(D[i,j+1]-(2*D[i,j])+D[i,j-1])/((dx**2)*(dy**2))
                Fy = (D[i,j+1]-D[i,j-1])/(2*(dx**2)*(dy**2))
            else:
                Cy = 0
                Dy = 0
                Ey = 0
                Fy = 0
            # Finally G needs all the corners...
            if i > 0 and i < xn-1 and j > 0 and j < yn-1:
                G = (
                        (1-pr)*(
                                     D[i+1,j+1]
                                    -D[i-1,j+1]
                                    -D[i+1,j-1]
                                    +D[i-1,j-1]
                                )/(8*(dx**2)*(dy**2))
                    )
            else:
                G = 0
            fdcoeff_ij = (6*Ax)+(6*Ay)+(4*B)-(2*Dx)-(2*Dy)-(2*Ey)-(2*Ex)
            fdcoeff_ip2_j = Ax + Cx
            fdcoeff_im2_j = Ax - Cx
            fdcoeff_i_jp2 = Ay + Cy
            fdcoeff_i_jm2 = Ay - Cy
            fdcoeff_ip1_j = -(4*Ax)-(2*B)-(2*Cx) + Dx + Ey - (2*Fx)
            fdcoeff_im1_j = -(4*Ax)-(2*B)+(2*Cx) + Dx + Ey + (2*Fx)
            fdcoeff_i_jp1 = -(4*Ay)-(2*B)-(2*Cy) + Dy + Ex - (2*Fy)
            fdcoeff_i_jm1 = -(4*Ay)-(2*B)+(2*Cy) + Dy + Ex + (2*Fy)
            fdcoeff_ip1_jp1 = B + Fx + Fy + G
            fdcoeff_ip1_jm1 = B + Fx - Fy - G
            fdcoeff_im1_jp1 = B - Fx + Fy - G
            fdcoeff_im1_jm1 = B - Fx - Fy + G            
            # Corrected formulation (Kneller, 2021)
            # b[k] = (rho_c-rho_w)*g*delta_uc[i,j]
            # -(rho_m[i,j]-rho_c)*g*delta_lc[i,j]
            # -(rho_s[i,j]-rho_w)*g*h_s[i,j]+q_thermal[i,j]+dBL*rho_w*g
            b[k] = q_uc[i,j]-q_lc[i,j]-q_sed[i,j]+q_thermal[i,j]+q_BL[i,j]
            A[k, i_j] = fdcoeff_ij + ((rho_m[i,j]-rho_w)*g)
            if i > 2-1:
                A[k, im2_j] = fdcoeff_im2_j
            if i < xn-1-1:
                A[k, ip2_j] = fdcoeff_ip2_j
            if i > 1-1:
                A[k, im1_j] = fdcoeff_im1_j
            if i < xn-1:
                A[k, ip1_j] = fdcoeff_ip1_j
            if j > 2-1:
                A[k, i_jm2] = fdcoeff_i_jm2
            if j < yn-1-1:
                A[k, i_jp2] = fdcoeff_i_jp2
            if j > 1-1:
                A[k, i_jm1] = fdcoeff_i_jm1
            if j < yn-1:
                A[k, i_jp1] = fdcoeff_i_jp1
            if i > 1-1 and j > 1-1:
                A[k, im1_jm1] = fdcoeff_im1_jm1
            if i < xn-1 and j > 1-1:
                A[k, ip1_jm1] = fdcoeff_ip1_jm1
            if i > 1-1 and j < yn-1:
                A[k, im1_jp1] = fdcoeff_im1_jp1
            if i < xn-1 and j < yn-1:
                A[k, ip1_jp1] = fdcoeff_ip1_jp1
            k = k + 1


@jit(nopython=True)
def build_system_compact(
                    xn, yn, global_index, 
                    D, dx, dy, pr, g, q_sed, q_uc,
                    q_lc, q_thermal, q_BL, rho_m, 
                    rho_c, rho_w, Lii, Ljj, Li, Lj, Lv, b
):
    inz = 0
    k = 0
    for i in range(xn):
        for j in range(yn):
            i_j = global_ind(i,j, yn)
            im2_j = global_ind(i-2,j, yn)
            im1_j = global_ind(i-1,j, yn)
            i_jm2 = global_ind(i,j-2, yn)
            i_jm1 = global_ind(i,j-1, yn)
            ip2_j = global_ind(i+2,j, yn)
            ip1_j = global_ind(i+1,j, yn)
            i_jp2 = global_ind(i,j+2, yn)
            i_jp1 = global_ind(i,j+1, yn)         
            im1_jm1 = global_ind(i-1,j-1, yn)
            im1_jp1 = global_ind(i-1,j+1, yn)
            ip1_jm1 = global_ind(i+1,j-1, yn)
            ip1_jp1 = global_ind(i+1,j+1, yn)
            global_index[k,0] = i
            global_index[k,1] = j
            # coefficients for A matrix.
            Ax = D[i,j]/(dx**4)
            Ay = D[i,j]/(dy**4)
            B = (2*D[i,j])/((dx**2)*(dy**2))
            # Cx, Dx, Ex, and Fx need x boundaries.
            if i > 0 and i < xn-1:
                Cx = (D[i+1,j]-D[i-1,j])/(2*(dx**4))
                Dx = (D[i+1,j]-(2*D[i,j])+D[i-1,j])/(dx**4)
                Ex = pr*(D[i+1,j]-(2*D[i,j])+D[i-1,j])/((dx**2)*(dy**2))
                Fx = (D[i+1,j]-D[i-1,j])/(2*(dx**2)*(dy**2))
            else:
                Cx = 0
                Dx = 0
                Ex = 0
                Fx = 0
            # Cy, Dy, Ey, and Fy need y boundaries.
            if j > 0 and j < yn-1:
                Cy = (D[i,j+1]-D[i,j-1])/(2*(dy**4))
                Dy = (D[i,j+1]-(2*D[i,j])+D[i,j-1])/(dy**4)
                Ey = pr*(D[i,j+1]-(2*D[i,j])+D[i,j-1])/((dx**2)*(dy**2))
                Fy = (D[i,j+1]-D[i,j-1])/(2*(dx**2)*(dy**2))
            else:
                Cy = 0
                Dy = 0
                Ey = 0
                Fy = 0
            # Finally G needs all the corners...
            if i > 0 and i < xn-1 and j > 0 and j < yn-1:
                G = (
                        (1-pr)*(
                                  D[i+1,j+1]
                                  -D[i-1,j+1]
                                  -D[i+1,j-1]
                                  +D[i-1,j-1]
                                )/(8*(dx**2)*(dy**2))
                    ) 
            else:
                G = 0
            fdcoeff_ij = (6*Ax)+(6*Ay)+(4*B)-(2*Dx)-(2*Dy)-(2*Ey)-(2*Ex)
            fdcoeff_ip2_j = Ax + Cx
            fdcoeff_im2_j = Ax - Cx
            fdcoeff_i_jp2 = Ay + Cy
            fdcoeff_i_jm2 = Ay - Cy
            fdcoeff_ip1_j = -(4*Ax)-(2*B)-(2*Cx) + Dx + Ey - (2*Fx)
            fdcoeff_im1_j = -(4*Ax)-(2*B)+(2*Cx) + Dx + Ey + (2*Fx)
            fdcoeff_i_jp1 = -(4*Ay)-(2*B)-(2*Cy) + Dy + Ex - (2*Fy)
            fdcoeff_i_jm1 = -(4*Ay)-(2*B)+(2*Cy) + Dy + Ex + (2*Fy)
            fdcoeff_ip1_jp1 = B + Fx + Fy + G
            fdcoeff_ip1_jm1 = B + Fx - Fy - G
            fdcoeff_im1_jp1 = B - Fx + Fy - G
            fdcoeff_im1_jm1 = B - Fx - Fy + G
            b[k] = q_uc[i,j]-q_lc[i,j]-q_sed[i,j]+q_thermal[i,j]+q_BL[i,j]
            Lv[inz]=fdcoeff_ij + ((rho_m[i,j]-rho_w)*g)
            Li[inz]=k
            Lj[inz]=i_j
            Lii[inz]=i
            Ljj[inz]=j                     
            inz=inz+1
            if i > 2-1:
                Lv[inz]=fdcoeff_im2_j
                Li[inz]=k
                Lj[inz]=im2_j
                Lii[inz]=i
                Ljj[inz]=j                     
                inz=inz+1
            if i < xn-1-1:
                Lv[inz]=fdcoeff_ip2_j
                Li[inz]=k
                Lj[inz]=ip2_j
                Lii[inz]=i
                Ljj[inz]=j                     
                inz=inz+1
            if i > 1-1:
                Lv[inz]=fdcoeff_im1_j
                Li[inz]=k
                Lj[inz]=im1_j
                Lii[inz]=i
                Ljj[inz]=j                     
                inz=inz+1
            if i < xn-1:
                Lv[inz]=fdcoeff_ip1_j
                Li[inz]=k
                Lj[inz]=ip1_j
                Lii[inz]=i
                Ljj[inz]=j                     
                inz=inz+1
            if j > 2-1:
                Lv[inz]=fdcoeff_i_jm2
                Li[inz]=k
                Lj[inz]=i_jm2
                Lii[inz]=i
                Ljj[inz]=j                     
                inz=inz+1
            if j < yn-1-1:
                Lv[inz]=fdcoeff_i_jp2
                Li[inz]=k
                Lj[inz]=i_jp2
                Lii[inz]=i
                Ljj[inz]=j                     
                inz=inz+1
            if j > 1-1:
                Lv[inz]= fdcoeff_i_jm1
                Li[inz]=k
                Lj[inz]=i_jm1
                Lii[inz]=i
                Ljj[inz]=j                     
                inz=inz+1
            if j < yn-1:
                Lv[inz]=fdcoeff_i_jp1
                Li[inz]=k
                Lj[inz]=i_jp1
                Lii[inz]=i
                Ljj[inz]=j                     
                inz=inz+1
            if i > 1-1 and j > 1-1:
                Lv[inz]=fdcoeff_im1_jm1
                Li[inz]=k
                Lj[inz]=im1_jm1
                Lii[inz]=i
                Ljj[inz]=j                     
                inz=inz+1
            if i < xn-1 and j > 1-1:
                Lv[inz]=fdcoeff_ip1_jm1
                Li[inz]=k
                Lj[inz]=ip1_jm1
                Lii[inz]=i
                Ljj[inz]=j                     
                inz=inz+1
            if i > 1-1 and j < yn-1:
                Lv[inz]=fdcoeff_im1_jp1
                Li[inz]=k
                Lj[inz]=im1_jp1
                Lii[inz]=i
                Ljj[inz]=j                     
                inz=inz+1
            if i < xn-1 and j < yn-1:
                Lv[inz]=fdcoeff_ip1_jp1
                Li[inz]=k
                Lj[inz]=ip1_jp1
                Lii[inz]=i
                Ljj[inz]=j                     
                inz=inz+1
            k = k + 1
    return inz


@jit(nopython=True)
def clean_non_zero_arrays(
                            nnz, Lii_tmp, Ljj_tmp, Li_tmp, 
                            Lj_tmp, Lv_tmp, Lii, Ljj, Li, Lj, Lv
):
    for i in range(nnz):
        Lii[i]=Lii_tmp[i]
        Ljj[i]=Ljj_tmp[i]
        Li[i]=Li_tmp[i]
        Lj[i]=Lj_tmp[i]
        Lv[i]=Lv_tmp[i]


@jit(nopython=True)
def reassemble_matrix(nw, global_index, w_vec, w):
    for k in range(nw):
        i = global_index[k,0]
        j = global_index[k,1]
        w[i,j] = w_vec[k]
        
        
def flex_solver_varD(
                        xn, yn, dx, dy, 
                        rho_c, rho_w, rho_m,
                        q_sed, q_uc, q_lc, 
                        q_thermal, q_BL, D, pr
):
    # 0 = build full matrix; 1 = only define non-zero elements
    ibuild_type = 1
    # gravity, m/s**2
    g = 9.81
    # Need to assemble the matrix. 
    # 2D maps require a global indexing for subscripts. Create an index vector
    # to help. Global index starts at x = 1 and counts each y-node. Need 12
    # neighbors for node for the finite difference calculation.
    global_index = np.zeros((xn*yn, 2),dtype=int)
    b = np.zeros((yn*xn,1))
    if ibuild_type == 0:
        A = np.zeros((yn*xn,yn*xn))
        build_system_full(
                            xn, yn, global_index, D, 
                            dx, dy, pr, g, q_sed, q_uc,
                            q_lc, q_thermal, q_BL, 
                            rho_m, rho_c, rho_w, A, b
                        )
        As = sps.csr_matrix(A)
        w_vec = linsolve.spsolve(As, b, use_umfpack=True)
        nw = w_vec.size
        # Reassemble the matrix.
        w = np.zeros((xn,yn))
        reassemble_matrix(nw, global_index, w_vec, w)
#        for k in range(nw):
#            i = global_index[k,0]
#            j = global_index[k,1]
#            w[i,j] = w_vec[k]
    elif ibuild_type == 1:
        ss = 13
        Lii_tmp = np.zeros((xn*yn*ss))
        Ljj_tmp = np.zeros((xn*yn*ss))
        Li_tmp = np.zeros((xn*yn*ss))
        Lj_tmp = np.zeros((xn*yn*ss))
        Lv_tmp = np.zeros((xn*yn*ss))
        nnz = build_system_compact(
                                xn, yn, global_index, D, 
                                dx, dy, pr, g, q_sed,
                                q_uc, q_lc, q_thermal, 
                                q_BL, rho_m, rho_c, rho_w, Lii_tmp,
                                Ljj_tmp, Li_tmp, Lj_tmp, Lv_tmp, b
                            )
        Lii = np.zeros((nnz), dtype=np.int)
        Ljj = np.zeros((nnz), dtype=np.int)
        Li = np.zeros((nnz), dtype=np.int)
        Lj = np.zeros((nnz), dtype=np.int)
        Lv = np.zeros((nnz))
        clean_non_zero_arrays(
                                nnz, Lii_tmp, Ljj_tmp, Li_tmp, 
                                Lj_tmp, Lv_tmp, Lii, Ljj, 
                                Li, Lj, Lv
                            )
        N = xn*yn
        Ls = sps.csr_matrix((Lv, (Li, Lj)), shape=(N, N), dtype=np.float64)
        # Solve matrix
        w_vec = linsolve.spsolve(Ls, b, use_umfpack=True)
        nw = w_vec.size
        # Reassemble the matrix.
        w = np.zeros((xn,yn))
        reassemble_matrix(nw, global_index, w_vec, w)
#        for k in range(nw):
#            i = global_index[k,0]
#            j = global_index[k,1]
#            w[i,j] = w_vec[k]
    return w


@jit(nopython=True)
def clean_load_boundaries(xn, yn, q_load):
    imax = xn-1
    jmax = yn-1
    for i in range(xn):
        for j in range(yn):
            if i == 0 and j == 0: # lower left corner
                q_load[i, j] = q_load[i+1, j+1]
            elif i == 0 and j == jmax: # upper left corner
                q_load[i, j] = q_load[i+1, j-1]
            elif i == imax and j == jmax: # upper right
                q_load[i, j] = q_load[i-1, j-1]
            elif i == imax and j == 0: # lower right
                q_load[i, j] = q_load[i-1, j+1]
            elif i == 0: # left 
                q_load[i, j] = q_load[i+1, j]
            elif i == imax: # right
                q_load[i, j] = q_load[i-1, j]
            elif j == 0: # bottom
                q_load[i, j] = q_load[i, j+1]
            elif j == jmax: # top
                q_load[i, j] = q_load[i, j-1]                


@jit(nopython=True)
def calc_rigidity(xn, yn, Te, pr, E, D):
    for i in range(xn):
        for j in range(yn):
            Te_tmp = Te[i][j]
            D[i][j] = (E*Te_tmp**3)/(12*(1-(pr**2)))

            
def calc_flex_sub(
                        rho_c, rho_w, rho_m, q_sed, 
                        q_uc, q_lc, q_thermal,
                        q_BL, dx, dy, AOI, n_AOIs, 
                        Te, pr, E, dist_taper, event_ID
):    
    # rho_c             density of the crust (scalar)
    # rho_m             mean mantle density (array)
    # q_thermal         Calculated thermal load N/m2 (array)
    # dx, dy            cell dimensions of the grid
    # AOI               matrix storing AOI values; 0 if defined, nan if not
    # n_AOIs            number of defined values 
    # Te_km             matrix of effective elastic thickness (km)
    # pr                poisson's ratio
    # E                 young's modulus
    xn = np.size(q_uc,0)
    yn = np.size(q_uc,1)
    meanTe = calc_avg_2Dgrid_omitnan(Te, xn, yn)
    avg_rho_m = calc_avg_2Dgrid_omitnan(rho_m, xn, yn)
    # Calculate the appropriate taper width. Use relationship for radius as 
    # described pn page 67 of Watt's textbook.
    if meanTe == 0.0:
        meanTe = 1.0
    xnf = 2*math.ceil(dist_taper/dx)
    ynf = 2*math.ceil(dist_taper/dy)
    xn_exp = xn+2*xnf
    yn_exp = yn+2*ynf
    Wtaper = (dx+dy)/2*xnf
    frac_taper = 0.75
    Dtaper = Wtaper*frac_taper
    #**************************************************************************
    # Determine rigidity.
    #**************************************************************************
    # Rigidity of the lithosphere supporting sediment load (Nm)
    D = np.zeros((xn,yn))
    calc_rigidity(xn, yn, Te, pr, E, D)
#    for i in range(xn):
#        for j in range(yn):
#            Te_tmp = Te[i][j]
#            D[i][j] = (E*Te_tmp**3)/(12*(1-(pr**2))) 
    meanD = calc_avg_2Dgrid_omitnan(D, xn, yn)
    # Fill large grid will average mantle density and mean rigidity
    Flex_rho_m = np.ones((xn_exp,yn_exp))*avg_rho_m  # asthenosphere density    
    Flex_D = np.ones((xn_exp,yn_exp))*meanD   # Rigidity    
    # Initialize large main
    Flex_q_sed = np.zeros((xn_exp,yn_exp))
    Flex_q_uc = np.zeros((xn_exp,yn_exp))
    Flex_q_lc = np.zeros((xn_exp,yn_exp))
    Flex_q_thermal = np.zeros((xn_exp,yn_exp))
    Flex_q_BL = np.zeros((xn_exp,yn_exp))
    # Fill large main grid 
    Flex_q_sed[xnf+1-1:xn+xnf,ynf+1-1:yn+ynf] = q_sed
    Flex_q_uc[xnf+1-1:xn+xnf,ynf+1-1:yn+ynf] = q_uc
    Flex_q_lc[xnf+1-1:xn+xnf,ynf+1-1:yn+ynf] = q_lc
    Flex_q_thermal[xnf+1-1:xn+xnf,ynf+1-1:yn+ynf] = q_thermal
    Flex_q_BL[xnf+1-1:xn+xnf,ynf+1-1:yn+ynf] = q_BL
    replace_nan_with_avg(rho_m, xn, yn, avg_rho_m)
    Flex_rho_m[xnf+1-1:xn+xnf,ynf+1-1:yn+ynf] = rho_m
    replace_nan_with_avg(D, xn, yn, meanD)
    Flex_D[xnf+1-1:xn+xnf,ynf+1-1:yn+ynf] = D
    #**************************************************************************
    # Flex_filling_AOI is used to determine which external cells are filled
    # with interpolated values and which are not. The taper algorithm continues 
    # untill all external cells are filled.
    #**************************************************************************
    Flex_filling_AOI = np.zeros((xn_exp,yn_exp))
    Flex_filling_AOI[xnf+1-1:xn+xnf,ynf+1-1:yn+ynf] = AOI
    #**************************************************************
    # Initialize list of information associated with neighbor cells
    #**************************************************************
    apply_taper_method2(
                            xn, yn, dx, dy, 
                            xnf, ynf, xn_exp, yn_exp,
                            Flex_filling_AOI, Flex_q_sed, 
                            Flex_q_uc, Flex_q_lc, Flex_q_thermal,
                            Flex_q_BL, Flex_D, q_sed, q_uc, q_lc, 
                            q_thermal, q_BL, D, Dtaper
                        )
    Flex_w = flex_solver_varD(
                            xn_exp, yn_exp, dx, dy, 
                            rho_c, rho_w, Flex_rho_m,
                            Flex_q_sed, Flex_q_uc, Flex_q_lc, 
                            Flex_q_thermal, Flex_q_BL, Flex_D, pr
                        )
    #w = Flex_w[xnf + 1 -1:xn + xnf, ynf + 1 - 1:yn + ynf]
    w = Flex_w[xnf:xn + xnf, ynf:yn + ynf]
    return w, Flex_w, xn_exp, yn_exp