#!/bin/bash
# This script sets up environment for Habitat-Sim CPU rendering

# Set OSMesa library path
export OSMESA_LIBRARY=/usr/lib/x86_64-linux-gnu/libOSMesa.so

# Force software rendering
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330

# Disable GPU completely
export CUDA_VISIBLE_DEVICES=""

# Set Habitat-Sim to use CPU
export HABITAT_SIM_GPU_IDS=""  # No GPUs
export MAGNUM_GPU_VALIDATION=OFF

# Run the command
"$@"