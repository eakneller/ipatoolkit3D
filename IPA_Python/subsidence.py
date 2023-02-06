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
def calc_sub(
                ntime, sub_t, z_moho_t, L, rho_w, rho_c_t, rho_m_t, alpha,
                T_bottom, iup_in_air, rho_crust, rho_mantle, itype_rho_a,
                L_crust_ref, L_lith_ref, rho_crust_ref, rho_mantle_ref
):
    if itype_rho_a == 0:
        rho_a = rho_mantle*(1.0-alpha*T_bottom)
    else:
        rho_a = rho_mantle
    alith = L_lith_ref
    Lco = L_crust_ref
    Lmo = alith-Lco
    tempmoho = T_bottom/alith*Lco
    rho_co = rho_crust_ref*(1.0 - alpha*tempmoho/2.0)
    rho_mo = rho_mantle_ref*(1.0 - alpha*(tempmoho+T_bottom)/2.0)
    rho_w_ini = rho_w
    for i in range(ntime):
        Lc = z_moho_t[i]
        Lm = L-Lc
        rho_c = rho_c_t[i]
        rho_m = rho_m_t[i]
        icalc_again = 1
        rho_w = rho_w_ini
        for j in range(2):
            if icalc_again == 1:
                sub = (
                        (
                              Lco*(rho_a-rho_co) 
                            + Lmo*(rho_a-rho_mo) 
                            - Lc*(rho_a-rho_c) 
                            - Lm*(rho_a-rho_m)
                        )/(rho_a-rho_w)
                    )
            if iup_in_air == 1:
                if sub < 0.0:
                    if rho_w > 0.0:
                        icalc_again = 1
                        rho_w = 0.0
                    else:
                        icalc_again = 0
            else:
                icalc_again = 0
        sub_t[i] = sub