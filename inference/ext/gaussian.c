#include "gaussian.h"


// Function to compuet the averagte number of patches in a 2D Gaussian
static inline int compute_avg_2d_gaussian_patches(float sigma_x, float sigma_y, float *patch_sizes, int num_patches_sizes, int *out_avg_num_patches, float coverage) {

    // compute the integral of the 2D Gaussian
    float gaussian_integral = compute_2d_gaussian_integral(sigma_x, sigma_y);

    // compute the average number of patches for each patch size
    for (int i = 0; i < num_patches_sizes; i++) {
        float patch_size = patch_sizes[i];
        float patch_area = patch_size * patch_size;
        out_avg_num_patches[i] = (int) (gaussian_integral * coverage / patch_area);
    }

    return 0;
}

// Function to compute the num patches per resolution given a desired target num patches
int compute_num_patches_per_resolution(float sigma_x, float sigma_y, int target_num_patches, float *patch_sizes, int num_patches_sizes, int **out_num_patches_per_resolution, float coverage) {

    // compute the average number of patches
    int *avg_num_patches;

    // allocate memory for the output
    avg_num_patches = (int*)malloc(num_patches_sizes * sizeof(int));

    if (!avg_num_patches) {
        return -1;
    }
    
    if (compute_avg_2d_gaussian_patches(sigma_x, sigma_y, patch_sizes, num_patches_sizes, avg_num_patches, coverage) < 0) {
        return -1;
    }

    // allocate memory for the output
    int *num_patches_per_resolution = (int*)malloc(num_patches_sizes * sizeof(int));

    if (!out_num_patches_per_resolution) {
        return -1;
    }
    
    // go from highes to lowest resolution (reverse order)
    for (int i = num_patches_sizes - 1; i >= 0; i--) {
        num_patches_per_resolution[i] = relu(min(target_num_patches, avg_num_patches[i]));
        target_num_patches -= num_patches_per_resolution[i];
    }

    free(avg_num_patches);
    *out_num_patches_per_resolution = num_patches_per_resolution;

    return 0;
}

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
) {
    // Normalize rotation
    float scale = sqrt(rot_a * rot_a + rot_b * rot_b);
    if (scale > 1e-16) {
        rot_a /= scale;
        rot_b /= scale;
    }
    
    // Compute the number of patches to sample at each resolution
    int *num_patches_per_res = NULL;
    if (compute_num_patches_per_resolution(sigma_x, sigma_y, target_num_patches, patch_sizes, num_patch_sizes, &num_patches_per_res, coverage) < 0) {
        printf("Error: compute_num_patches_per_resolution\n");
        return -1;
    }
    
    // Allocate memory for output arrays
    uint8_t **patch_buffers = (uint8_t**)malloc(num_patch_sizes * sizeof(uint8_t*));
    int *valid_counts = (int*)malloc(num_patch_sizes * sizeof(int));
    float **norm_coords = (float**)malloc(num_patch_sizes * sizeof(float*));
    int **target_inds = (int**)malloc(num_patch_sizes * sizeof(int*));
    
    if (!patch_buffers || !valid_counts || !norm_coords || !target_inds) {
        printf("Error: Memory allocation failed\n");
        return -1;
    }
    
    // Initialize counts
    for (int i = 0; i < num_patch_sizes; i++) {
        valid_counts[i] = 0;
    }
    
    // Create a spatial hash grid using the API
    int max_overlaps = ceilf(max_overlap_threshold);
    SpatialHashGrid* grid = spatial_hash_init(patch_sizes, num_patch_sizes, num_patches_per_res, max_overlaps);
    if (!grid) {
        printf("Error: Failed to initialize spatial hash grid\n");
        return -1;
    }
    
    // Pre-allocate max possible space for each resolution
    for (int i = 0; i < num_patch_sizes; i++) {
        int p_int = (int)patch_sizes[i];
        int num_patches = num_patches_per_res[i];
        
        patch_buffers[i] = (uint8_t*)malloc(num_patches * p_int * p_int * C * sizeof(uint8_t));
        norm_coords[i] = (float*)malloc(num_patches * 2 * sizeof(float));
        target_inds[i] = (int*)malloc(num_patches * sizeof(int));
        
        if (!patch_buffers[i] || !norm_coords[i] || !target_inds[i]) {
            printf("Error: Memory allocation failed for resolution %d\n", i);
            return -1;
        }
    }

    // Initialize our Gaussian sampler 
    GaussianSamplerState gaus_state;
    init_gaussian_sampler(&gaus_state, mu_x, mu_y, sigma_x, sigma_y, rot_a, rot_b);
    
    // Process each resolution
    for (int res_idx = 0; res_idx < num_patch_sizes; res_idx++) {
        int p_int = (int)patch_sizes[res_idx];
        float half_p = p_int / 2.0f;
        
        // Try to sample the target number of patches
        int attempts = 0;
        int max_attempts = num_patches_per_res[res_idx] * 10; // Allow up to 10x attempts
        
        while (valid_counts[res_idx] < num_patches_per_res[res_idx] && attempts < max_attempts) {
            attempts++;
            
            // Sample from 2D Gaussian (continuous coordinates)
            float center_x;
            float center_y;
            sample_2d_gaus_continuous(&gaus_state, &center_x, &center_y);
            
            // Check if patch would be fully within image bounds
            if (center_x - half_p < 0 || center_x + half_p >= W - 1 || 
                center_y - half_p < 0 || center_y + half_p >= H - 1) {
                continue; // Skip if patch goes outside image bounds
            }
            
            // Check overlap percentage against threshold
            if (spatial_hash_has_significant_overlap(grid, center_x, center_y, p_int, max_overlap_threshold)) {
                continue; // Skip if overlap percentage exceeds threshold
            }
            
            // Add this patch's bounding box to the spatial hash grid using the new API
            spatial_hash_add_box(grid, center_x, center_y, p_int);
            
            // Extract patch with bilinear interpolation
            uint8_t *patch_ptr = patch_buffers[res_idx] + (valid_counts[res_idx] * p_int * p_int * C);
            
            // Iterate over the patch grid
            for (int y_idx = 0; y_idx < p_int; y_idx++) {
                for (int x_idx = 0; x_idx < p_int; x_idx++) {
                    // Calculate pixel coordinates in the image
                    float y_offset = y_idx - half_p + 0.5f; 
                    float x_offset = x_idx - half_p + 0.5f;
                    
                    // Calculate the exact pixel coordinates in the image
                    float pixel_y = center_y + y_offset;
                    float pixel_x = center_x + x_offset;
                    
                    // Get the four surrounding integer coordinates
                    int x0 = (int)pixel_x;
                    int y0 = (int)pixel_y;
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    
                    // Calculate the interpolation weights
                    float dx = pixel_x - x0;
                    float dy = pixel_y - y0;
                    
                    // Get patch buffer destination index
                    uint8_t *dest = patch_ptr + (y_idx * p_int * C) + (x_idx * C);
                    
                    // Perform bilinear interpolation for each channel
                    for (int c = 0; c < C; c++) {
                        uint8_t f00 = image_data[(y0 * W + x0) * C + c];
                        uint8_t f01 = image_data[(y0 * W + x1) * C + c];
                        uint8_t f10 = image_data[(y1 * W + x0) * C + c];
                        uint8_t f11 = image_data[(y1 * W + x1) * C + c];
                        
                        float value = f00 * (1 - dx) * (1 - dy) +
                                       f01 * dx * (1 - dy) +
                                       f10 * (1 - dx) * dy +
                                       f11 * dx * dy;
                        
                        dest[c] = (uint8_t)roundf(value);
                    }
                }
            }
            
            // Save normalized coordinates
            norm_coords[res_idx][2 * valid_counts[res_idx]] = center_x / 128.0f - ((float)W) / 256.0f;
            norm_coords[res_idx][2 * valid_counts[res_idx] + 1] = center_y / 128.0f - ((float)H) / 256.0f;
            
            valid_counts[res_idx]++;
        }
    }
    
    // Step 5: Compute target indices as cumulative offsets
    int cumulative = 0;
    for (int i = 0; i < num_patch_sizes; i++) {
        for (int j = 0; j < valid_counts[i]; j++) {
            target_inds[i][j] = cumulative + j;
        }
        cumulative += valid_counts[i];
    }
    
    // Step 6: Resize output arrays to actual count (if we sampled fewer than allocated)
    for (int i = 0; i < num_patch_sizes; i++) {
        int p_int = (int)patch_sizes[i];
        int count = valid_counts[i];

        if (count == 0) {
            free(patch_buffers[i]);
            free(norm_coords[i]);
            free(target_inds[i]);
            patch_buffers[i] = NULL;
            norm_coords[i] = NULL;
            target_inds[i] = NULL;
        } else {
            if (count < num_patches_per_res[i]) {
                uint8_t *resized_patches = (uint8_t*)realloc(patch_buffers[i], count * p_int * p_int * C * sizeof(uint8_t));
                float *resized_coords = (float*)realloc(norm_coords[i], count * 2 * sizeof(float));
                int *resized_inds = (int*)realloc(target_inds[i], count * sizeof(int));
                
                if (resized_patches) patch_buffers[i] = resized_patches;
                if (resized_coords) norm_coords[i] = resized_coords;
                if (resized_inds) target_inds[i] = resized_inds;
            }
        }
    }
    
    // Step 7: Clean up and set outputs
    spatial_hash_free(grid);
    free(num_patches_per_res);
    
    *out_patches = patch_buffers;
    *out_patch_counts = valid_counts;
    *out_coordinates = norm_coords;
    *out_target_indices = target_inds;
    
    return 0;
}
