#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f pyqt_app.yml
conda env create -f habitat_env.yml