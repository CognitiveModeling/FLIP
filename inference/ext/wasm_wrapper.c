// wasm_wrapper.c
#include <emscripten/emscripten.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Include the existing C headers
#include "gaussian.h"
#include "position.h"
#include "bbox.h"
#include "ziggurat_inline.h"

// Keep track of allocated memory for cleanup
typedef struct {
    uint8_t **patch_buffers;
    int *patch_counts;
    float **coordinates;
    int **target_indices;
    int num_patch_sizes;
} SampleResult;

static SampleResult last_result = {NULL, NULL, NULL, NULL, 0};

// Initialize the random number generator
EMSCRIPTEN_KEEPALIVE
void wasm_initialize_random(int seed) {
    initialize_ziggurat(seed);
}

// Free previously allocated memory
void cleanup_last_result() {
    if (last_result.patch_buffers) {
        for (int i = 0; i < last_result.num_patch_sizes; i++) {
            if (last_result.patch_buffers[i]) {
                free(last_result.patch_buffers[i]);
            }
            if (last_result.coordinates && last_result.coordinates[i]) {
                free(last_result.coordinates[i]);
            }
            if (last_result.target_indices && last_result.target_indices[i]) {
                free(last_result.target_indices[i]);
            }
        }
        free(last_result.patch_buffers);
    }
    if (last_result.patch_counts) {
        free(last_result.patch_counts);
    }
    if (last_result.coordinates) {
        free(last_result.coordinates);
    }
    if (last_result.target_indices) {
        free(last_result.target_indices);
    }
        
    last_result.patch_buffers = NULL;
    last_result.patch_counts = NULL;
    last_result.coordinates = NULL;
    last_result.target_indices = NULL;
    last_result.num_patch_sizes = 0;
}


// Sample continuous patches
EMSCRIPTEN_KEEPALIVE
int wasm_sample_continuous_patches(
    uint8_t* image_data, int H, int W, int C,
    float mu_x, float mu_y, float sigma_x, float sigma_y,
    float rot_a, float rot_b,
    int target_num_patches,
    float *patch_sizes, int num_patch_sizes,
    float max_overlap_threshold,
    float coverage
) {
    // Apply sqrt(2)x sigma for sampling (as in original sample_patches)
    sigma_x *= sqrtf(2);
    sigma_y *= sqrtf(2);
    
    // Convert from normalized coordinates (-1 to 1) to image coordinates
    mu_x = (mu_x + 1) * W / 2;
    mu_y = (mu_y + 1) * H / 2;
    sigma_x = sigma_x * W / 2;
    sigma_y = sigma_y * H / 2;
    
    // Flip rotation for sampling (as in the original code)
    rot_a = -rot_a;
    
    // Use provided parameters or generate random ones
    if (max_overlap_threshold < 0) {
        max_overlap_threshold = r4_uni_value() * 4.0;
    }
    if (coverage < 0) {
        coverage = 0.1 + r4_uni_value() * 1.9;
    }
    
    // Call the gaussian C extension for continuous patch sampling
    uint8_t **patch_buffers = NULL;
    int *patch_counts = NULL;
    float **coordinates = NULL;
    int **target_indices = NULL;
    
    int result = extract_continuous_patches_with_indices_native(
        image_data, H, W, C,
        mu_x, mu_y, sigma_x, sigma_y, rot_a, rot_b,
        target_num_patches, patch_sizes, num_patch_sizes,
        max_overlap_threshold, coverage,
        &patch_buffers, &patch_counts,
        &coordinates, &target_indices
    );
    
    if (result < 0) {
        return -1;
    }
    
    // Store the result for later access
    last_result.patch_buffers = patch_buffers;
    last_result.patch_counts = patch_counts;
    last_result.coordinates = coordinates;
    last_result.target_indices = target_indices;
    last_result.num_patch_sizes = num_patch_sizes;
    
    return 0;
}

// Get the number of patches for a specific resolution
EMSCRIPTEN_KEEPALIVE
int wasm_get_patch_count(int resolution_index) {
    if (resolution_index < 0 || resolution_index >= last_result.num_patch_sizes) {
        return 0;
    }
    return last_result.patch_counts[resolution_index];
}

// Get patch data for a specific resolution
EMSCRIPTEN_KEEPALIVE
uint8_t* wasm_get_patches(int resolution_index) {
    if (resolution_index < 0 || resolution_index >= last_result.num_patch_sizes) {
        return NULL;
    }
    return last_result.patch_buffers[resolution_index];
}

// Get coordinate data for a specific resolution
EMSCRIPTEN_KEEPALIVE
float* wasm_get_coordinates(int resolution_index) {
    if (resolution_index < 0 || resolution_index >= last_result.num_patch_sizes) {
        return NULL;
    }
    return last_result.coordinates[resolution_index];
}

// Clean up allocated memory
EMSCRIPTEN_KEEPALIVE
void wasm_cleanup() {
    cleanup_last_result();
}

// Compute position rotation from rho (for completeness)
EMSCRIPTEN_KEEPALIVE
void wasm_compute_position_rot_from_rho(float *position_rho, float *output_position) {
    compute_position_rot_from_rho(position_rho, output_position);
}
