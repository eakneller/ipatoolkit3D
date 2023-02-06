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
import math


@jit(nopython=True, cache=True)
def bulk_porosity_layer(phi_o, c, z1, z2):
    if z2-z1 > 0.0:
        phi_bulk = (phi_o/c)*(math.exp(-c*z1)-math.exp(-c*z2))/(z2-z1)
    else:
        phi_bulk = 0.0
    return phi_bulk


@jit(nopython=True, cache=True)
def bulk_rho_layer(phi_bulk, rho_g, rho_w):
    rho_bulk = phi_bulk*rho_w + (1-phi_bulk)*rho_g
    return rho_bulk


def get_total_thickness(thicks):
    total_thick = 0.0
    for thick in thicks:
        total_thick = total_thick + thick
    return total_thick


def bulk_rho_column(brho_layers, z_tops, total_thick):
    sumit = 0.0
    for i, brho in enumerate(brho_layers):
        z1 = z_tops[i]
        z2 = z_tops[i+1]
        dz = z2-z1
        sumit = sumit + brho*dz
    if total_thick > 0.0:
        bulk_rho_col = sumit/total_thick
    else:
        bulk_rho_col = 0.0
    return bulk_rho_col
