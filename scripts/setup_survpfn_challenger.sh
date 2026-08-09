#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${1:-/mnt/d/My-Work/Esophageal/New}"
ENV_PREFIX="${SURVPFN_ENV_PREFIX:-/home/wiky/miniforge3/envs/recast-survpfn}"
CONDA="${CONDA_EXE:-/home/wiky/miniforge3/bin/conda}"
PYTHON="$ENV_PREFIX/bin/python"

cd "$WORKSPACE"
if [[ ! -x "$PYTHON" ]]; then
  "$CONDA" create -y -p "$ENV_PREFIX" python=3.12 pip
fi

"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0
"$PYTHON" -m pip install \
  numpy==2.4.4 scipy==1.17.1 pandas==2.3.3 pyarrow==25.0.0 \
  scikit-learn==1.8.0 scikit-survival==0.27.0 huggingface-hub==1.3.7 \
  pfns==0.3.0 h5py==3.16.0 joblib==1.5.3 lifelines==0.30.3 \
  click==8.4.2 PyYAML==6.0.3 matplotlib==3.11.1

if [[ ! -d third_party/SurvPFN/.git ]]; then
  git clone https://github.com/genepi-freiburg/SurvPFN.git third_party/SurvPFN
fi
if [[ ! -d third_party/TFM-Playground/.git ]]; then
  git clone https://github.com/automl/TFM-Playground.git third_party/TFM-Playground
fi
git -C third_party/SurvPFN checkout --detach 4f26fe31efd811f546a6a3a12acd28293828ca7e
git -C third_party/TFM-Playground checkout --detach 4afd33e1c60928c8425ac979b9e1fd092ad0f216

"$PYTHON" -m pip install -e third_party/TFM-Playground --no-deps
"$PYTHON" -m pip install -e third_party/SurvPFN --no-deps
"$PYTHON" -m pip install -e . --no-deps
"$PYTHON" -c 'from SurvPFN import SurvPFN; import torch; print("SurvPFN import passed; torch", torch.__version__)'
