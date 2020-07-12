#pragma once

#include <sys/time.h>

class Timer {
private:
  double wtime()
  {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1.0E-6;
  }

public:
  inline void start() { this->tick = wtime(); }
  inline void stop()
  {
    this->tock = wtime();
    this->elaps = this->tock - this->tick;
  }
  double elapsed() { return this->elaps; }

private:
  double tick, tock, elaps;
};
