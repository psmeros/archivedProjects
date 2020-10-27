## PHPC-2020 FINAL PROJECT
### Conjugate Gradient Method for solving Ax = b


#### Requirements

> - gcc V7.5.0
> - mpicc V3.3a2
> - nvcc V9.1.85
> - OpenBLAS V0.2.20
```
module load gcc openmpi cuda openblas
```


#### Compilation

> Create executables for the three implementations: 
> - conjugategradient_seq (Sequential Implementation)
> - conjugategradient_mpi (MPI Implementation)
> - conjugategradient_cuda (CUDA Implementation)
```
$ make
```

#### Experiments

> Run strong and weak scaling experiment
```
$ make run_strong_scaling
$ make run_weak_scaling
```


#### Results

> Create the reported plots for strong and weak scaling
```
$ python results/plots.py
```
