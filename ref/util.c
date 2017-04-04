#include <stdio.h>
#include <math.h>

#include "util.h"

double change_range(double v, double min_old, double max_old, double min_new, double max_new)
{
	return ((max_new - min_new)/(max_old - min_old)) * (v - min_old) + min_new;
}

