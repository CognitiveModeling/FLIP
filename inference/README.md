# FLIP Inference

Inference pipeline for FLIP models, providing ONNX export and WebAssembly compilation for efficient deployment.

## Features

- **ONNX Export**: Convert trained PyTorch models to ONNX format with KV caching optimization
- **WebAssembly Support**: Compile C extensions to WASM for browser-based inference  
- **Optimized C Extensions**: High-performance patch sampling and Gaussian operations
- **Evaluation Tools**: Benchmark model performance on HDF5 datasets

## Setup

### 1. Create Environment

```bash
conda env create -f environment.yml
conda activate flip-inference
```

**Note**: Default environment includes GPU support. For CPU-only, modify `environment.yml`:
- Replace `pytorch` with `pytorch-cpu`
- Replace `onnxruntime-gpu` with `onnxruntime`

### 2. Build C Extensions

```bash
cd ext
python setup.py build install
cd ..
```

### 3. (Optional) WebAssembly Compilation

For browser deployment, first install Emscripten:

```bash
# Install Emscripten
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
cd ..

# Compile to WebAssembly
cd ext
chmod +x compile_wasm.sh
./compile_wasm.sh
cd ..
```

## Quick Start

1. **Export trained model to ONNX**:
   ```bash
   python export_to_onnx.py \
       --config configs/flip-large.json \
       --checkpoint chceckpoints/flip-large.ckpt \
       --encoder-output flip-encoder-large.onnx \
       --predictor-output flip-predictor-large.onnx
   ```

2. **Run evaluation**:
   ```bash
   python -m model.scripts.evaluate_single_hdf5_onnx \
       --dataset_path your_dataset.hdf5 \
       --encoder_path flip-encoder-large.onnx \
       --predictor_path flip-predictor-large.onnx \
       --config configs/flip-large.json \
   ```

