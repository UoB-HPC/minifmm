#pragma once

#include <sys/time.h>
#include <stdio.h>

typedef struct
{
    double tick;
    double tock;
    double elapsed;
}t_timer;

double wtime();

static inline void start(t_timer* timer)
{
    timer->tick = wtime();
}

static inline void stop(t_timer* timer)
{
    timer->tock = wtime();
    timer->elapsed = timer->tock - timer->tick;
}

static inline double timer_seconds(t_timer* timer)
{
    return timer->elapsed;
}

static inline void timer_print(t_timer* timer, const char* timer_string)
{
    if (timer_string != NULL) printf("----- %s -----\n", timer_string);
    else printf("----------\n");
    printf("Total elapsed time = %f\n", timer_seconds(timer));
    printf("----------\n");
}
