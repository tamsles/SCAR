#!/bin/bash

# Wisteria project code passed to `pjsub -g`.
PROJECT_GROUP="gu15"

# Verified on the Wisteria login node with `show_module` on 2026-07-14.
PYTORCH_MODULE="pytorch/1.8.1"

# Set this to an absolute virtual-environment interpreter when not using the
# system PyTorch module. In that case, set PYTORCH_MODULE="" and change its
# CUDA dependency through EXTRA_MODULES.
PYTHON_BIN="/work/opt/local/x86_64/apps/cuda/11.1/pytorch/1.8.1/bin/python3"
EXTRA_MODULES="cuda/11.1"

# Unit tests are quick and catch environment/package incompatibilities before
# a 200-epoch allocation is consumed.
RUN_TESTS=1
