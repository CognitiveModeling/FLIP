#ifndef __GAUSSIAN_H__
#define __GAUSSIAN_H__
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "ziggurat_inline.h"
#include "bbox.h" 
#include "gaussian_batch.h"

// epsilon for rounding
#define EPSILON 1e-10f

// Map Ziggurat function to a more descriptive name
#define randn r4_nor_value

// Map Ziggurat random integer function to a more descriptive name
#define randi shr3_value

// Map Ziggurat random float function to a more descriptive name
#define rand r4_uni_value

// inline relu function
static inline float relu(float x) {
    return x > 0 ? x : 0;
}

// inline max function
static inline float max(float x, float y) {
    return x > y ? x : y;
}

// inline min function
static inline float min(float x, float y) {
    return x < y ? x : y;
}

/**
 * Initializes the Ziggurat random number generator.
 * If seed == -1, it uses time + clock to generate a random seed.
 */
static inline void initialize_ziggurat(int seed) {
    //printf("Initializing Ziggurat with seed: %d\n", seed);
    uint32_t seed_jsr, seed_jcong, seed_w, seed_z;

    if (seed == -1) {
        // Use time + clock for a more randomized seed
        uint32_t t = (uint32_t) time(NULL);
        uint32_t c = (uint32_t) clock();

        seed_jsr   = t ^ (c << 16) ^ (c >> 16);
        seed_jcong = (t * 69069) + (c * 1234567);
        seed_w     = (t + c) * 1664525 + 1013904223;
        seed_z     = (t ^ c) * 18000 + 987654321;
    } else {
        // Use user-provided seed and mix it a bit
        seed_jsr   = (uint32_t) seed ^ 0xCAFEBABE;
        seed_jcong = (uint32_t) (seed * 69069) + 1234567;
        seed_w     = (uint32_t) (seed * 1664525) + 1013904223;
        seed_z     = (uint32_t) (seed * 18000) + 987654321;
    }

    // Seed the random number generator
    zigset(seed_jsr, seed_jcong, seed_w, seed_z);
    r4_nor_setup();
}


// Function to compute the integral of a 2D Gaussian
static inline float compute_2d_gaussian_integral(float sigma_x, float sigma_y) {
    return (2.0f * M_PI * sigma_x * sigma_y);
}

// Function to compute the num patches per resolution given a desired target num patches
int compute_num_patches_per_resolution(
    float sigma_x, 
    float sigma_y, 
    int target_num_patches, 
    float *patch_sizes, 
    int num_patches_sizes, 
    int **out_num_patches_per_resolution, 
    float coverage
);

/**
 * Computes the L1 (Manhattan) distance to the nearest boundary in a binary mask.
 * Each pixel in the output distance map contains the minimum grid distance to a pixel
 * with a different value in the input mask.
 *
 * @param mask_data: Binary mask (uint8_t array)
 * @param H: Height of the mask
 * @param W: Width of the mask
 * @param out_distance_map: Output distance map (int array, will be allocated)
 *
 * @return: 0 on success, -1 on failure
 */
int compute_mask_boundary_distance_map(
    uint8_t* mask_data,
    int H,
    int W,
    int** out_distance_map
);

/**
 * Creates arrays of coordinates grouped by their L1 distance from mask boundaries.
 * This function first computes the distance map, then groups coordinates into specified 
 * distance ranges: 0-1, 2-4, 4-8, 8-16, 16-32, 32-64, and beyond 64 pixels.
 *
 * @param mask_data: Binary mask (uint8_t array)
 * @param H: Height of the mask
 * @param W: Width of the mask
 * @param out_inside_coord_arrays: Output array of 7 float* arrays for inside mask coordinates
 * @param out_outside_coord_arrays: Output array of 7 float* arrays for outside mask coordinates
 * @param out_inside_counts: Output array of 7 integers giving the count of inside coordinates in each array
 * @param out_outside_counts: Output array of 7 integers giving the count of outside coordinates in each array
 *
 * @return: 0 on success, -1 on failure
 */
int group_coordinates_by_boundary_distance(
    uint8_t* mask_data,
    int H,
    int W,
    float*** out_inside_coord_arrays,
    float*** out_outside_coord_arrays,
    int** out_inside_counts,
    int** out_outside_counts
);

/**
 * Sample pixels from a mask using efficient cropping based on rotated Gaussian parameters,
 * then sample using boundary distance information from the cropped mask.
 *
 * @param mask_data: Binary mask (uint8_t array)
 * @param H: Height of the mask
 * @param W: Width of the mask
 * @param mu_x, mu_y: Mean of the Gaussian distribution
 * @param sigma_x, sigma_y: Standard deviations of the Gaussian
 * @param rot_a, rot_b: Rotation parameters (normalized direction vector)
 * @param samples_per_group: Array of 7 integers specifying total samples to take from each boundary distance group
 * @param out_pixel_values: Output array of sampled pixel values (uint8_t)
 * @param out_coordinates: Output array of normalized coordinates (float)
 * @param out_count: Output total number of sampled pixels
 *
 * @return: 0 on success, -1 on failure
 */
int sample_mask_with_cropping(
    uint8_t* mask_data,
    int H, int W,
    float mu_x, float mu_y,
    float sigma_x, float sigma_y,
    float rot_a, float rot_b,
    int* samples_per_group,
    /* outputs: */
    uint8_t** out_pixel_values,
    float** out_coordinates,
    int* out_count
);

// Modified extract_continuous_patches_with_indices_native function
int extract_continuous_patches_with_indices_native(
    uint8_t* image_data, int H, int W, int C,
    float mu_x, float mu_y, float sigma_x, float sigma_y,
    float rot_a, float rot_b,
    int target_num_patches,
    float *patch_sizes, int num_patch_sizes, 
    float max_overlap_threshold,  
    float coverage,
    /* outputs: */
    uint8_t ***out_patches,      // [num_patch_sizes]: buffers for patches
    int **out_patch_counts,      // [num_patch_sizes]: number of patches per resolution
    float ***out_coordinates,   // [num_patch_sizes]: float arrays of shape (count,2)
    int ***out_target_indices    // [num_patch_sizes]: int arrays of shape (count,)
);
#endif // __GAUSSIAN_H__
