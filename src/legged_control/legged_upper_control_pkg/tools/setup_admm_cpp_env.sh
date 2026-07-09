#!/bin/bash
# One-time environment setup for the ADMM C++ core (run INSIDE the SIL container, as root):
#   docker exec LeggedControl_SIL bash /root/LeggedControl_ws/src/legged_control/legged_upper_control_pkg/tools/setup_admm_cpp_env.sh
#
# Pins OSQP C to v0.6.3 to match osqp-python 0.6.3 — same solver code on both sides is
# the basis for bit-identical Python/C++ ADMM results (golden-reference verification).
set -euo pipefail

# pybind11 headers + CMake config (find_package'able from /usr/local)
python3 -c "import pybind11" 2>/dev/null || pip3 install "pybind11[global]==2.13.6"

# OSQP C library v0.6.3 (DLONG=ON / DFLOAT=OFF defaults — matches osqp-python's build)
if [ ! -f /usr/local/include/osqp/osqp.h ]; then
    rm -rf /tmp/osqp-src
    git clone --depth 1 --recursive -b v0.6.3 https://github.com/osqp/osqp.git /tmp/osqp-src
    cmake -S /tmp/osqp-src -B /tmp/osqp-src/build -DCMAKE_BUILD_TYPE=Release
    cmake --build /tmp/osqp-src/build -j"$(nproc)"
    cmake --install /tmp/osqp-src/build
    ldconfig
    rm -rf /tmp/osqp-src
fi

echo "[setup_admm_cpp_env] OK — pybind11 $(python3 -c 'import pybind11; print(pybind11.__version__)'), OSQP headers at /usr/local/include/osqp"
