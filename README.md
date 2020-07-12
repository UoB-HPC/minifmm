# MiniFMM

This mini-app implements the Fast Multipole Method to solve the Laplace equation in 3D using spherical coordinates. The method outputs the gravitational acceleration and potential for all particles in the input.

It was originally designed as a benchmark for task-parallel frameworks/implementations. The mini-app is suited to task-parallelism as it uses the Dual Tree-Traversal method to either approximate or calculate directly the particle forces in a system.  

Current implementations:

- OpenMP (using locks, atomics, and task dependencies)
- TBB
- CILK
- CUDA
- Kokkos (CPU and GPU)

## Usage

Build using Makefile in each directory. See READMEs in each directory for additional details.

The Makefiles will first use all the source files in the current directory, then any missing source files will be copied from the ``ref`` directory. (Except for compiling ``ref``)

Set the COMPILER environment variable to either GNU, Intel, Cray, or Clang to enable the appropriate compiler flags.

Other enviroment variables:

- ``ARCH`` used to set the CPU architecture if needed (default native)
- ``TYPE`` used to set data type, can be either DOUBLE or FLOAT

**As this is an approximate method, the last two lines print out the error of the gravational potential and acceleration. These values should typically be in the order of 10^-3 to 10^-7.**

## Input

Run with ``-i`` flag, located in ``input`` folder.

Measured on 22-core dual socket Intel Skylake /w HT (44 cores, 88 threads). Intel 2018 compiler and ``omp`` implementation.

| Name      | Typical Error Values (force, potential) | Typical OpenMP runtime (s) |
| --------- | --------------------------------------- | -------------------------- |  
| small.in  | 2.32e-04, 1.02e-05                      | 0.16                       |
| medium.in | 7.06e-08, 2.17e-09                      | 0.77                       |
| large.in  | 6.48e-08, 1.96e-09                      | 2.42                       |
| huge.in   | 1.02e-05, 5.25e-07                      | 13.04                      |

## Parameters

| Param | Description                                                                                          |
| ----- | ---------------------------------------------------------------------------------------------------- |
| h     | Print Help                                                                                           |
| n     | Number of particles, positions will be randomly generated between range 0.0 - 1.0 for all dimensions |
| c     | Maximum number of particles per node                                                                 |
| t     | Number of terms in multipole expansion                                                               |
| e     | Theta (as in Barnes-Hut)                                                                             |
| m     | Number of samples used to verify solution                                                            |
| i     | Input file                                                                                           |

Setting the ``c`` value too low will reduce the amount of work per task but will decrease the overall size of nodes (cells) which allows for more approximations of forces. 

Theta (``e``) is the ratio of the current node's (cell's) size and the distance to the source node under consideration, increasing this will decrease accuracy and allow for more approximations.

The number of terms (``t``) determines time complexity of approximating forces.

The time complexity of computing forces between nodes directly is ``O(n^2)``, where n is number of particles in a node.
The time complexity of approximating forces between nodes is ``O(t^4)``, where t is the number of terms in the multipole expansions.

## Issues

- Currently uses a fixed seed, can use different seed by changing argument to ``seed_rng`` in ``initialise.c``

- ``omp-atomics`` has problems with atomics on complex types - see README in directory for more details

- Sampling to verify solution only uses the first m points, rather than random sampling

## Citation

To cite please use this reference:

> Atkinson, Patrick and McIntosh-Smith, Simon. On the performance of parallel tasking runtimes for an irregular fast multipole method application. [Paper](https://link.springer.com/chapter/10.1007/978-3-319-65578-9_7)
