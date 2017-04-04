#pragma once

#include <stdlib.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#if defined(_MSC_VER) || defined(__MINGW32__)
    #define __PRINTF_ZU     "%Iu"
#elif defined(__GNUC__)
    #define __PRINTF_ZU     "%zu"        
#endif

double change_range(double v, double min_old, double max_old, double min_new, double max_new);