// gaussian_batch.h
// Highly optimized batch-based 2D Gaussian sampling with fixed parameters

#ifndef GAUSSIAN_BATCH_H
#define GAUSSIAN_BATCH_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ziggurat_inline.h"

// Buffer size - adjust based on profiling
#define GAUS_BUFFER_SIZE 256
#define GAUS_BUFFER_REFILL_THRESHOLD 16


// Gaussian sampler state with fixed distribution parameters
typedef struct {
    // Sample buffer
    float x_buffer[GAUS_BUFFER_SIZE];
    float y_buffer[GAUS_BUFFER_SIZE];
    int buffer_position;
    int buffer_count;
    
    // Fixed Gaussian parameters
    float mu_x;
    float mu_y;
    float sigma_x;
    float sigma_y;
    float rot_a;
    float rot_b;
} GaussianSamplerState;

/**
 * Initialize a Gaussian sampler state with fixed distribution parameters
 * 
 * @param state: Pointer to a GaussianSamplerState to initialize
 * @param mu_x, mu_y: Mean of the Gaussian
 * @param sigma_x, sigma_y: Standard deviations
 * @param rot_a, rot_b: Rotation components (will be normalized)
 */
static inline void init_gaussian_sampler(
    GaussianSamplerState* state,
    float mu_x, float mu_y,
    float sigma_x, float sigma_y,
    float rot_a, float rot_b) {
    
    if (!state) return;
    
    // Clear buffer state
    state->buffer_position = 0;
    state->buffer_count = 0;
    
    // Store the fixed parameters
    state->mu_x = mu_x;
    state->mu_y = mu_y;
    state->sigma_x = sigma_x;
    state->sigma_y = sigma_y;
    state->rot_a = rot_a;
    state->rot_b = rot_b;
    
    // Pre-fill the buffer with initial samples
    for (int i = 0; i < GAUS_BUFFER_SIZE; i++) {
        // Generate standard Gaussian samples using Ziggurat
        float z_x = (float)r4_nor_value();
        float z_y = (float)r4_nor_value();
        
        // Apply scaling
        z_x *= sigma_x;
        z_y *= sigma_y;
        
        // Apply rotation
        float x_rot = rot_a * z_x - rot_b * z_y;
        float y_rot = rot_b * z_x + rot_a * z_y;
        
        // Translate by mean
        state->x_buffer[i] = x_rot + mu_x;
        state->y_buffer[i] = y_rot + mu_y;
    }
    
    state->buffer_count = GAUS_BUFFER_SIZE;
}

/**
 * Refill the Gaussian sample buffer using fixed parameters
 * 
 * @param state: Pointer to the sampler state to refill
 */
static inline void refill_gaussian_buffer(GaussianSamplerState* state) {
    // Reset buffer position
    state->buffer_position = 0;
    
    // Use stored parameters
    float mu_x = state->mu_x;
    float mu_y = state->mu_y;
    float sigma_x = state->sigma_x;
    float sigma_y = state->sigma_y;
    float rot_a = state->rot_a;
    float rot_b = state->rot_b;
    
    // Generate new batch of samples
    for (int i = 0; i < GAUS_BUFFER_SIZE; i++) {
        // Generate standard Gaussian samples using Ziggurat
        float z_x = (float)r4_nor_value();
        float z_y = (float)r4_nor_value();
        
        // Apply scaling
        z_x *= sigma_x;
        z_y *= sigma_y;
        
        // Apply rotation
        float x_rot = rot_a * z_x - rot_b * z_y;
        float y_rot = rot_b * z_x + rot_a * z_y;
        
        // Translate by mean
        state->x_buffer[i] = x_rot + mu_x;
        state->y_buffer[i] = y_rot + mu_y;
    }
    
    state->buffer_count = GAUS_BUFFER_SIZE;
}

/**
 * Get a single 2D Gaussian sample from the pre-configured distribution.
 * Ultra-fast with minimal overhead.
 * 
 * @param state: Pointer to the sampler state
 * @param out_x, out_y: Output coordinates
 */
static inline void sample_2d_gaus_continuous(
    GaussianSamplerState* state,
    float* out_x, float* out_y) {
    
    // Check if we need to refill the buffer
    if (state->buffer_position >= state->buffer_count - GAUS_BUFFER_REFILL_THRESHOLD) {
        refill_gaussian_buffer(state);
    }
    
    // Return the next sample from the buffer
    *out_x = state->x_buffer[state->buffer_position];
    *out_y = state->y_buffer[state->buffer_position];
    state->buffer_position++;
}


#endif /* GAUSSIAN_BATCH_H */
