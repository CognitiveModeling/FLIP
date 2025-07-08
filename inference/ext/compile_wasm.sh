#!/bin/bash

# compile_wasm.sh - Script to compile the C code to WASM using Emscripten

# Make sure emcc is installed
if ! command -v emcc &> /dev/null; then
    echo "Error: Emscripten (emcc) not found. Please install Emscripten first."
    echo "Visit: https://emscripten.org/docs/getting_started/downloads.html"
    exit 1
fi

echo "Compiling WASM module..."

# Compile the C files to WASM
emcc -O3 \
    wasm_wrapper.c \
    gaussian.c \
    position.c \
    bbox.c \
    ziggurat_inline.c \
    -I./ \
    -s WASM=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_FUNCTIONS='["_wasm_initialize_random", "_wasm_sample_continuous_patches", "_wasm_get_patch_count", "_wasm_get_patches", "_wasm_get_coordinates", "_wasm_cleanup", "_wasm_compute_position_rot_from_rho", "_malloc", "_free"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap", "getValue", "setValue", "HEAP8", "HEAP16", "HEAP32", "HEAPF32", "HEAPU8"]' \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="FlipWASM" \
    -o flip_wasm.js

if [ $? -eq 0 ]; then
    echo "Successfully compiled to flip_wasm.js and flip_wasm.wasm"
else
    echo "Compilation failed!"
    exit 1
fi
