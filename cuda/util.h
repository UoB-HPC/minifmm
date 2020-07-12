#pragma once

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// 20 characters long
#define SEPERATOR "--------------------\n"
    
    static inline
void ipow(int n, TYPE* real, TYPE* imag)
{
    //i = (n & 0x1) ? I : TYPE_ONE;
    *real = (n & 0x01) ? TYPE_ZERO : TYPE_ONE;
    *imag = (n & 0x01) ? TYPE_ONE : TYPE_ZERO;
    //i *= (n & 0x2) ? -TYPE_ONE : TYPE_ONE;
    *real *= (n & 0x02) ? -TYPE_ONE : TYPE_ONE;
    *imag *= (n & 0x02) ? -TYPE_ONE : TYPE_ONE;
    //TYPE_COMPLEX i;
}

static inline
TYPE neg_pow_n(int n)
{
    return (TYPE)(1 + ((n & 0x01)*-2));
}

