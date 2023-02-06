# -*- coding: utf-8 -*-
import numba
    
    
def manage_parallel(func, itype3D):
    iforce_serial = 1
    if itype3D == 0 or iforce_serial == 1:
        (
            func_active
        ) = numba.jit(nopython=True, cache=True)(func)
    else:
        (
            func_active
        ) = numba.jit(nopython=True, parallel=True)(func)
    return func_active