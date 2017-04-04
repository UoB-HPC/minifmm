#pragma once

#ifdef USE_FLOAT
    #error single precision wont work until spherical harmonics implemented for floats
    #include "type_float.h"
#else
    #include "type_double.h"
#endif
