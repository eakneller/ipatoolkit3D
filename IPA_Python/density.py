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
def calc_rho(
                ntime, rho_c_t, rho_m_t, z_moho_t, 
                L, alpha_c, alpha_m, rho_crust, 
                rho_mantle, avg_Tc_t, avg_Tm_t
):
    for i in range(ntime):
        Tc_avg = avg_Tc_t[i]
        rho_c = rho_crust*(1.0-alpha_c*Tc_avg)
        rho_c_t[i] = rho_c
        Tm_avg = avg_Tm_t[i]
        rho_m = rho_mantle*(1.0-alpha_m*Tm_avg)
        rho_m_t[i] = rho_m
