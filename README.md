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

```
./fmm.<Programming Model> <Arguments>
```

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

Information
-----------

Please cite using:
```
Atkinson, P., & McIntosh-Smith, S. (2017). On the Performance of Parallel Tasking Runtimes for an Irregular Fast Multipole Method Application. In Scaling OpenMP for Exascale Performance and Portability (pp. 92–106). Springer International Publishing. https://doi.org/10.1007/978-3-319-65578-9_7
```

For further information, please refer to: https://patrickatkinson.co.uk/phd-thesis.pdf

