#!bin/bash
apptainer exec -B /sda2:/sda2 --no-home jax.sif mlflow server --backend-store-uri sqlite:///l2d_pop.db --port 5000