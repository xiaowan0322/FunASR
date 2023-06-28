TORCH_DIR=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
BLADE_DIR=$(python3 -c "import torch_blade; import os; print(os.path.dirname(torch_blade.__file__))")

_LD_LIBRARY_PATH=/usr/local/lib:${TORCH_DIR}/lib:${BLADE_DIR}:$LD_LIBRARY_PATH
# : '
LD_LIBRARY_PATH=$_LD_LIBRARY_PATH g++ torch_app.cc \
    -std=c++14 \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I ${TORCH_DIR}/include \
    -I ${TORCH_DIR}/include/torch/csrc/api/include \
    -Wl,--no-as-needed \
    -L /usr/local/lib \
    -L ${TORCH_DIR}/lib \
    -L ${BLADE_DIR} \
    -l torch -l torch_cuda -l torch_cpu -l c10 -l c10_cuda \
    -l torch_blade -l ral_base_context \
    -o torch_app
# '

LD_LIBRARY_PATH=$_LD_LIBRARY_PATH ./torch_app model.fixfp16.blade.fp16.pt asr_data.pth
LD_LIBRARY_PATH=$_LD_LIBRARY_PATH ./torch_app model.gpu.torchscripts asr_data.pth
LD_LIBRARY_PATH=$_LD_LIBRARY_PATH ./torch_app model.fixfp16.fp16.pt asr_data.pth
