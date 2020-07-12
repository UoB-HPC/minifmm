MiniFMM
=======

Building
--------

```
make COMPILER=<Compiler> ARCH=<Architecture> MODEL=<Programming Model>
```

The programming model must be the same name as the directory (i.e. `omp-tasks`).

Running
-------

./fmm.<Programming Model> <Arguments>

Valid arguments

- `n`: no. of input particles
- `c`: max. no. of particles per tree node
- `t`: no. of multipole terms
- `e`: theta (as in Barnes-Hut)
- `m`: no. samples used to validate
- `p`: use a Plummer input distribution
- `u`: use a uniform input distribution (cube)
- `i`: input file

Some sample input files are given in `input/`. 

