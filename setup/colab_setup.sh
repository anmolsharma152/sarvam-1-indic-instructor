#!/bin/bash
# setup/colab_setup.sh
# Run in Colab: !bash setup/colab_setup.sh
set -e

echo "=== [1/2] Install Unsloth (manages torch/xformers/bitsandbytes/trl for Colab's CUDA) ==="
pip install unsloth

echo "=== [2/2] Install remaining project deps (no GPU conflicts) ==="
pip install -r requirements-colab.txt

echo "=== Setup complete. Verify versions ==="
python -c "import trl; print(f'trl {trl.__version__}')"
python -c "import torch; print(f'torch {torch.__version__}')"
echo "Ready to train!"
