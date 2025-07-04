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
) {
    // Allocate the distance map
    int* distance_map = (int*)malloc(H * W * sizeof(int));
    if (!distance_map) {
        printf("Error allocating memory for distance map\n");
        return -1;
    }
    
    const int INF = H + W;  // A value larger than any possible distance
    
    // Allocate queue for BFS
    int* queue = (int*)malloc(H * W * 2 * sizeof(int));
    
    if (!queue) {
        printf("Error allocating memory for queue\n");
        if (queue) free(queue);
        free(distance_map);
        return -1;
    }
    
    int front = 0, back = 0;

    static const int NUM_OFFSETS = 4;
    static const int offsets[4][2] = {
                  {-1,  0},
        { 0, -1},           { 0,  1},
                  { 1,  0}, 
    };
    
    // Initialize distance map and find boundary pixels
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            int idx = r * W + c;
            distance_map[idx] = INF;
            
            uint8_t val = mask_data[idx];
            int is_boundary = 0;
            
            for (int i = 0; i < NUM_OFFSETS; i++) {
                int nr = r + offsets[i][0];
                int nc = c + offsets[i][1];
                if (nr >= 0 && nr < H && nc >= 0 && nc < W) {
                    if (mask_data[nr * W + nc] != val) {
                        is_boundary = 1;
                        break;
                    }
                }
            }
            
            if (is_boundary) {
                distance_map[idx] = 0;
                queue[back * 2 + 0] = r;
                queue[back * 2 + 1] = c;
                back++;
            }
        }
    }
    
    // BFS to propagate distances
    while (front < back) {
        int r = queue[front * 2 + 0];
        int c = queue[front * 2 + 1];
        front++;
        
        int dist = distance_map[r * W + c];
        
        for (int i = 0; i < NUM_OFFSETS; i++) {
            int nr = r + offsets[i][0];
            int nc = c + offsets[i][1];
            
            if (nr >= 0 && nr < H && nc >= 0 && nc < W) {
                int new_idx = nr * W + nc;
                
                if (distance_map[new_idx] > dist + 1) {
                    distance_map[new_idx] = dist + 1;
                    queue[back * 2 + 0] = nr;
                    queue[back * 2 + 1] = nc;
                    back++;
                }
            }
        }
    }
    
    free(queue);
    
    *out_distance_map = distance_map;
    return 0;
}

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
) {
    // First compute the distance map
    int* distance_map = NULL;
    if (compute_mask_boundary_distance_map(mask_data, H, W, &distance_map) < 0) {
        return -1;
    }
    
    // Define our distance ranges - we'll use 7 ranges
    const int NUM_RANGES = 7;
    const int range_ends[7] = {2, 4, 8, 16, 32, 64, H*W};  // Inclusive upper bounds
    
    // Allocate output arrays
    float** inside_coord_arrays = (float**)malloc(NUM_RANGES * sizeof(float*));
    float** outside_coord_arrays = (float**)malloc(NUM_RANGES * sizeof(float*));
    int* inside_counts = (int*)malloc(NUM_RANGES * sizeof(int));
    int* outside_counts = (int*)malloc(NUM_RANGES * sizeof(int));
    
    if (!inside_coord_arrays || !outside_coord_arrays || !inside_counts || !outside_counts) {
        if (inside_coord_arrays) free(inside_coord_arrays);
        if (outside_coord_arrays) free(outside_coord_arrays);
        if (inside_counts) free(inside_counts);
        if (outside_counts) free(outside_counts);
        free(distance_map);
        return -1;
    }
    
    // Initialize counts and arrays
    for (int i = 0; i < NUM_RANGES; i++) {
        inside_counts[i] = 0;
        outside_counts[i] = 0;
        inside_coord_arrays[i] = NULL;
        outside_coord_arrays[i] = NULL;
    }
    
    // First pass: Count pixels in each range, separated by inside/outside
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            const int idx = r * W + c;
            int dist = distance_map[idx];
            uint8_t is_inside = mask_data[idx];
            
            for (int i = 0; i < NUM_RANGES; i++) {
                if (dist <= range_ends[i]) {
                    if (is_inside) {
                        inside_counts[i]++;
                    } else {
                        outside_counts[i]++;
                    }
                }
            }
        }
    }
    
    // Allocate memory for each coordinate array
    for (int i = 0; i < NUM_RANGES; i++) {
        if (inside_counts[i] > 0) {
            inside_coord_arrays[i] = (float*)malloc(2 * inside_counts[i] * sizeof(float));
            if (!inside_coord_arrays[i]) {
                // Clean up on allocation failure
                for (int j = 0; j < NUM_RANGES; j++) {
                    if (inside_coord_arrays[j]) free(inside_coord_arrays[j]);
                    if (outside_coord_arrays[j]) free(outside_coord_arrays[j]);
                }
                free(inside_coord_arrays);
                free(outside_coord_arrays);
                free(inside_counts);
                free(outside_counts);
                free(distance_map);
                return -1;
            }
        }
        
        if (outside_counts[i] > 0) {
            outside_coord_arrays[i] = (float*)malloc(2 * outside_counts[i] * sizeof(float));
            if (!outside_coord_arrays[i]) {
                // Clean up on allocation failure
                for (int j = 0; j < NUM_RANGES; j++) {
                    if (inside_coord_arrays[j]) free(inside_coord_arrays[j]);
                    if (outside_coord_arrays[j]) free(outside_coord_arrays[j]);
                }
                free(inside_coord_arrays);
                free(outside_coord_arrays);
                free(inside_counts);
                free(outside_counts);
                free(distance_map);
                return -1;
            }
        }
        
        // Reset counts for second pass
        inside_counts[i] = 0;
        outside_counts[i] = 0;
    }
    
    // Second pass: Store pixel coordinates in appropriate arrays
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            const int idx = r * W + c;
            int dist = distance_map[idx];
            uint8_t is_inside = mask_data[idx];
            
            for (int i = 0; i < NUM_RANGES; i++) {
                if (dist <= range_ends[i]) {
                    if (is_inside) {
                        int count_idx = inside_counts[i]++;
                        inside_coord_arrays[i][2 * count_idx] = (float)c;       // x-coordinate
                        inside_coord_arrays[i][2 * count_idx + 1] = (float)r;   // y-coordinate
                    } else {
                        int count_idx = outside_counts[i]++;
                        outside_coord_arrays[i][2 * count_idx] = (float)c;       // x-coordinate
                        outside_coord_arrays[i][2 * count_idx + 1] = (float)r;   // y-coordinate
                    }
                }
            }
        }
    }
    
    // Free the distance map, we no longer need it
    free(distance_map);
    
    // Set output parameters
    *out_inside_coord_arrays = inside_coord_arrays;
    *out_outside_coord_arrays = outside_coord_arrays;
    *out_inside_counts = inside_counts;
    *out_outside_counts = outside_counts;
    
    return 0;
}

/**
 * Sample pixels from specific coordinates with perturbation.
 * 
 * @param mask_data: Binary mask (uint8_t array)
 * @param H: Height of the mask
 * @param W: Width of the mask
 * @param coords_array: Array of coordinates to sample from
 * @param coords_count: Number of coordinates in the array
 * @param samples_needed: Number of samples to collect
 * @param is_inside: Whether these are inside (1) or outside (0) mask samples
 * @param pixel_values: Output array to store sampled pixel values
 * @param coordinates: Output array to store normalized coordinates
 * 
 * @return: Number of samples collected
 */
int sample_from_coords(
    uint8_t* mask_data,
    int H, int W,
    float* coords_array,
    int coords_count,
    int samples_needed,
    int is_inside,
    uint8_t* pixel_values,
    float* coordinates
) {
    // If no coordinates available, return 0
    if (coords_count <= 0 || samples_needed <= 0) {
        return 0;
    }
    
    int samples_collected = 0;
    
    while (samples_collected < samples_needed) {
        
        // Randomly select a coordinate
        int random_idx = randi() % coords_count;
        float base_x = coords_array[2 * random_idx];
        float base_y = coords_array[2 * random_idx + 1];
        
        // Apply random perturbation in range (-0.5, 0.5)
        float perturb_x = (float)rand() - 0.5f;
        float perturb_y = (float)rand() - 0.5f;
        
        float sample_x = base_x + perturb_x;
        float sample_y = base_y + perturb_y;
        
        // Store the pixel value 
        pixel_values[samples_collected] = is_inside;
        
        // Store coordinates
        coordinates[2 * samples_collected]     = sample_x;
        coordinates[2 * samples_collected + 1] = sample_y;
        
        // Update counter
        samples_collected++;
    }
    
    return samples_collected;
}

/**
 * Sample pixels from a mask using coordinates grouped by boundary distance.
 * For each group, samples pixels inside and outside the mask.
 *
 * @param mask_data: Binary mask (uint8_t array)
 * @param H: Height of the mask
 * @param W: Width of the mask
 * @param inside_coord_arrays: Array of 7 float* arrays of inside mask coordinates
 * @param outside_coord_arrays: Array of 7 float* arrays of outside mask coordinates
 * @param inside_counts: Array of 7 integers giving the count of inside coordinates in each group
 * @param outside_counts: Array of 7 integers giving the count of outside coordinates in each group
 * @param samples_per_group: Array of 7 integers specifying total samples to take from each group
 * @param out_pixel_values: Output array of sampled pixel values (uint8_t)
 * @param out_coordinates: Output array of normalized coordinates (float)
 * @param out_count: Output total number of sampled pixels
 *
 * @return: 0 on success, -1 on failure
 */
int sample_mask_by_boundary_distance(
    uint8_t* mask_data,
    int H, int W,
    float** inside_coord_arrays,
    float** outside_coord_arrays,
    int* inside_counts,
    int* outside_counts,
    int* samples_per_group,
    /* outputs: */
    uint8_t** out_pixel_values,
    float** out_coordinates,
    int* out_count
) {
    const int NUM_GROUPS = 7;
    
    // Calculate total samples needed
    int total_samples = 0;
    for (int i = 0; i < NUM_GROUPS; i++) {
        total_samples += samples_per_group[i];
    }
    
    // Allocate memory for outputs
    uint8_t* pixel_values = (uint8_t*)malloc(total_samples * sizeof(uint8_t));
    float* coordinates = (float*)malloc(2 * total_samples * sizeof(float));
    
    if (!pixel_values || !coordinates) {
        if (pixel_values) free(pixel_values);
        if (coordinates) free(coordinates);
        return -1;
    }
    
    int sample_count = 0; // Track total samples collected
    
    // For each boundary distance group
    for (int group = 0; group < NUM_GROUPS; group++) {
        int group_samples_needed = samples_per_group[group];
        
        // Skip if no samples needed from this group
        if (group_samples_needed <= 0) {
            continue;
        }
        
        // We want 50% inside and 50% outside
        int inside_samples_needed = group_samples_needed / 2;
        int outside_samples_needed = group_samples_needed - inside_samples_needed;
        
        // Process inside samples
        if (inside_samples_needed > 0) {
            int group_offset = 0;
            int inside_samples_collected = 0;
            
            while (group_offset < NUM_GROUPS && inside_samples_collected < inside_samples_needed) {
                if (group - group_offset >= 0) {
                    int group_to_sample = group - group_offset;
                    int samples_from_group = sample_from_coords(
                        mask_data, H, W,
                        inside_coord_arrays[group_to_sample], inside_counts[group_to_sample],
                        inside_samples_needed - inside_samples_collected,
                        1, // is_inside = true
                        pixel_values + (sample_count + inside_samples_collected), 
                        coordinates + 2 * (sample_count + inside_samples_collected) 
                    );
                
                    inside_samples_collected += samples_from_group;
                }
                if (group + group_offset < NUM_GROUPS) {
                    int group_to_sample = group + group_offset;
                    int samples_from_group = sample_from_coords(
                        mask_data, H, W,
                        inside_coord_arrays[group_to_sample], inside_counts[group_to_sample],
                        inside_samples_needed - inside_samples_collected,
                        1, // is_inside = true
                        pixel_values + (sample_count + inside_samples_collected), 
                        coordinates + 2 * (sample_count + inside_samples_collected)
                    );
                
                    inside_samples_collected += samples_from_group;
                }
                group_offset++; // Try the next larger or smaller group if needed
            }

            // If we still need more inside samples abort
            if (inside_samples_collected < inside_samples_needed) {
                free(pixel_values);
                free(coordinates);
                return -1;
            }
            
            sample_count += inside_samples_collected;
        }
        
        // Process outside samples
        if (outside_samples_needed > 0) {
            int group_offset = 0;
            int outside_samples_collected = 0;
            
            while (group_offset < NUM_GROUPS && outside_samples_collected < outside_samples_needed) {
                if (group - group_offset >= 0) {
                    int group_to_sample = group - group_offset;
                    int samples_from_group = sample_from_coords(
                        mask_data, H, W,
                        outside_coord_arrays[group_to_sample], outside_counts[group_to_sample],
                        outside_samples_needed - outside_samples_collected,
                        0, // is_inside = false
                        pixel_values + (sample_count + outside_samples_collected), 
                        coordinates + 2 * (sample_count + outside_samples_collected) 
                    );
                    
                    outside_samples_collected += samples_from_group;
                }
                if (group + group_offset < NUM_GROUPS) {
                    int group_to_sample = group + group_offset;
                    int samples_from_group = sample_from_coords(
                        mask_data, H, W,
                        outside_coord_arrays[group_to_sample], outside_counts[group_to_sample],
                        outside_samples_needed - outside_samples_collected,
                        0, // is_inside = false
                        pixel_values + (sample_count + outside_samples_collected), 
                        coordinates + 2 * (sample_count + outside_samples_collected) 
                    );
                    
                    outside_samples_collected += samples_from_group;
                }
                group_offset++; // Try the next larger / smaller group if needed
            }
            
            // If we still need more outside samples abort
            if (outside_samples_collected < outside_samples_needed) {
                free(pixel_values);
                free(coordinates);
                return -1;
            }
            
            sample_count += outside_samples_collected;
        }
    }
    
    // Set outputs
    *out_pixel_values = pixel_values;
    *out_coordinates = coordinates;
    *out_count = sample_count;
    
    return 0;
}

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
) {
    // Input validation
    if (!mask_data || H <= 0 || W <= 0 || !samples_per_group || !out_pixel_values || !out_coordinates || !out_count) {
        printf("Error: Invalid input parameters\n");
        return -1;
    }
    
    // Initialize output parameters
    *out_pixel_values = NULL;
    *out_coordinates = NULL;
    *out_count = 0;
    
    float sigma_iso = max(sigma_x, sigma_y);
    
    // Use 3-sigma rule to define bounding box
    int x_min = max(0, (int)floorf(mu_x - 5 * sigma_iso));
    int y_min = max(0, (int)floorf(mu_y - 5 * sigma_iso));
    int x_max = min(W - 1, (int)ceilf(mu_x + 5 * sigma_iso));
    int y_max = min(H - 1, (int)ceilf(mu_y + 5 * sigma_iso));
    
    // 2. Compute cropped mask dimensions
    int crop_W = x_max - x_min + 1;
    int crop_H = y_max - y_min + 1;
    
    // Sanity check for cropped dimensions
    if (crop_W <= 0 || crop_H <= 0) {
        printf("Error: Invalid crop dimensions: W=%d, H=%d\n", crop_W, crop_H);
        return -1;
    }
    
    // Special case: If the crop is almost the same size as original, skip cropping
    if (crop_W > W * 0.9 && crop_H > H * 0.9) {
        // Just call the original function directly
        float** inside_coord_arrays = NULL;
        float** outside_coord_arrays = NULL;
        int* inside_counts = NULL;
        int* outside_counts = NULL;
        
        if (group_coordinates_by_boundary_distance(
                mask_data, H, W,
                &inside_coord_arrays, &outside_coord_arrays,
                &inside_counts, &outside_counts) < 0) 
        {
            return -1;
        }
        
        int result = sample_mask_by_boundary_distance(
            mask_data, H, W,
            inside_coord_arrays, outside_coord_arrays,
            inside_counts, outside_counts,
            samples_per_group,
            out_pixel_values, out_coordinates, out_count);
        
        // Clean up
        for (int i = 0; i < 7; i++) {
            if (inside_coord_arrays && inside_coord_arrays[i]) free(inside_coord_arrays[i]);
            if (outside_coord_arrays && outside_coord_arrays[i]) free(outside_coord_arrays[i]);
        }
        free(inside_coord_arrays);
        free(outside_coord_arrays);
        free(inside_counts);
        free(outside_counts);

        // normalize coordinates
        float* coordinates = *out_coordinates;
        int count = *out_count;
        for (int i = 0; i < count; i++) {
            float x = coordinates[2 * i];
            float y = coordinates[2 * i + 1];
            
            // Convert to normalized coordinates in original space
            coordinates[2 * i] = x / 128.0f - ((float)W) / 256.0f;
            coordinates[2 * i + 1] = y / 128.0f - ((float)H) / 256.0f;
        }
        
        return result;
    }
    
    // 3. Allocate and create the cropped mask
    uint8_t* cropped_mask = (uint8_t*)malloc(crop_W * crop_H * sizeof(uint8_t));
    if (!cropped_mask) {
        printf("Error allocating memory for cropped mask\n");
        return -1;
    }
    
    // 4. Copy the mask data within the bounding box - optimized copy by rows
    for (int y = 0; y < crop_H; y++) {
        memcpy(
            cropped_mask + y * crop_W,
            mask_data + ((y + y_min) * W + x_min),
            crop_W * sizeof(uint8_t)
        );
    }
    
    // 5. Process the cropped mask using existing functions
    float** inside_coord_arrays = NULL;
    float** outside_coord_arrays = NULL;
    int* inside_counts = NULL;
    int* outside_counts = NULL;
    
    // Group coordinates by boundary distance in the cropped mask
    if (group_coordinates_by_boundary_distance(
            cropped_mask, crop_H, crop_W,
            &inside_coord_arrays, &outside_coord_arrays,
            &inside_counts, &outside_counts) < 0) 
    {
        printf("Error in group_coordinates_by_boundary_distance\n");
        free(cropped_mask);
        return -1;
    }
    
    // Sample pixels from the cropped mask
    uint8_t* pixel_values = NULL;
    float* coordinates = NULL;
    int count = 0;
    
    if (sample_mask_by_boundary_distance(
            cropped_mask, crop_H, crop_W,
            inside_coord_arrays, outside_coord_arrays,
            inside_counts, outside_counts,
            samples_per_group,
            &pixel_values, &coordinates, &count) < 0) 
    {
        // Clean up
        for (int i = 0; i < 7; i++) {
            if (inside_coord_arrays && inside_coord_arrays[i]) free(inside_coord_arrays[i]);
            if (outside_coord_arrays && outside_coord_arrays[i]) free(outside_coord_arrays[i]);
        }
        free(inside_coord_arrays);
        free(outside_coord_arrays);
        free(inside_counts);
        free(outside_counts);
        free(cropped_mask);
        return -1;
    }
    
    // 6. Adjust coordinates back to original image coordinates
    for (int i = 0; i < count; i++) {
        float x = coordinates[2 * i];
        float y = coordinates[2 * i + 1];
        
        // Adjust to original image space
        x += x_min;
        y += y_min;
        
        // Convert to normalized coordinates in original space
        coordinates[2 * i] = x / 128.0f - ((float)W) / 256.0f;
        coordinates[2 * i + 1] = y / 128.0f - ((float)H) / 256.0f;
    }
    
    // 7. Clean up temporary data
    for (int i = 0; i < 7; i++) {
        if (inside_coord_arrays && inside_coord_arrays[i]) free(inside_coord_arrays[i]);
        if (outside_coord_arrays && outside_coord_arrays[i]) free(outside_coord_arrays[i]);
    }
    free(inside_coord_arrays);
    free(outside_coord_arrays);
    free(inside_counts);
    free(outside_counts);
    free(cropped_mask);
    
    // 8. Set outputs
    *out_pixel_values = pixel_values;
    *out_coordinates = coordinates;
    *out_count = count;
    
    return 0;
}

/**
 * Sample pixels from a grayscale image using continuous 2D Gaussian distribution
 * with bilinear interpolation for exact pixel values.
 */
int extract_grayscale_pixels_with_indices_native(
    uint8_t* image_data, int H, int W,
    float mu_x, float mu_y, float sigma_x, float sigma_y,
    float rot_a, float rot_b,
    int target_num_pixels,
    /* outputs: */
    uint8_t **out_pixel_values,
    int *out_count,
    float **out_coordinates
) {
    // Normalize rotation
    float scale = sqrt(rot_a * rot_a + rot_b * rot_b);
    if (scale > 1e-16f) {
        rot_a /= scale;
        rot_b /= scale;
    }
    
    // If target is 0, return without doing any work
    if (target_num_pixels <= 0) {
        *out_pixel_values = NULL;
        *out_count = 0;
        *out_coordinates = NULL;
        return 0;
    }

    // Allocate output arrays
    uint8_t* pixel_values = (uint8_t*)malloc(target_num_pixels * sizeof(uint8_t));
    float* coordinates = (float*)malloc(2 * target_num_pixels * sizeof(float));
    if (!pixel_values || !coordinates) {
        if (pixel_values) free(pixel_values);
        if (coordinates) free(coordinates);
        return -1;
    }

    // Track attempts to prevent infinite loops
    int total_attempts = 0;
    int max_attempts = target_num_pixels * 10; // Allow up to 10x attempts
    
    // Sample exact number of pixels with bilinear interpolation
    int valid_count = 0;
    while (valid_count < target_num_pixels && total_attempts < max_attempts) {
        total_attempts++;
        
        // Sample from 2D Gaussian (continuous coordinates)
        
        // Sample standard Gaussian using Ziggurat
        float z_x = randn();
        float z_y = randn();

        // Apply scaling
        z_x *= sigma_x;
        z_y *= sigma_y;

        // Apply rotation
        float x_rot = rot_a * z_x - rot_b * z_y;
        float y_rot = rot_b * z_x + rot_a * z_y;

        // Translate by mean to get final coordinates
        float x = x_rot + mu_x;
        float y = y_rot + mu_y;
        
        // Check if within image bounds for interpolation
        // Need a margin of 1 pixel for interpolation
        if (x >= 0 && x < W-1 && y >= 0 && y < H-1) {
            // Get the four surrounding integer coordinates
            int x0 = (int)floorf(x);
            int y0 = (int)floorf(y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            // Get the four surrounding pixel values
            uint8_t f00 = image_data[y0 * W + x0];
            uint8_t f01 = image_data[y0 * W + x1];
            uint8_t f10 = image_data[y1 * W + x0];
            uint8_t f11 = image_data[y1 * W + x1];
            
            // Calculate the interpolation weights
            float dx = x - x0;
            float dy = y - y0;
            
            // Perform bilinear interpolation
            float value = f00 * (1 - dx) * (1 - dy) +
                           f01 * dx * (1 - dy) +
                           f10 * (1 - dx) * dy +
                           f11 * dx * dy;
            
            // Store the pixel value and coordinates
            pixel_values[valid_count] = (uint8_t)roundf(value * 255.0f);
            
            // Compute normalized coordinates
            coordinates[2 * valid_count]     = x / 128.0f - ((float)W) / 256.0f;
            coordinates[2 * valid_count + 1] = y / 128.0f - ((float)H) / 256.0f;
            
            valid_count++;
        }
        
        // If we're getting a bad success rate, adjust the distribution
        if (total_attempts >= target_num_pixels * 2 && valid_count < target_num_pixels / 2) {
            // Adjust by clamping mu_x, mu_y to be within image bounds
            mu_x = max(0.0f, min((float)W - 1, mu_x));
            mu_y = max(0.0f, min((float)H - 1, mu_y));
            
            // Reduce sigma if it's too large compared to image dimensions
            sigma_x = min(sigma_x, (float)W / 3.0f);
            sigma_y = min(sigma_y, (float)H / 3.0f);
        }
    }
    
    // If we couldn't sample enough pixels
    if (valid_count < target_num_pixels) {
        // Resize the output arrays to actual number of valid pixels
        uint8_t* resized_values = (uint8_t*)realloc(pixel_values, valid_count * sizeof(uint8_t));
        float* resized_coords = (float*)realloc(coordinates, 2 * valid_count * sizeof(float));
        
        if (resized_values) pixel_values = resized_values;
        if (resized_coords) coordinates = resized_coords;
    }
    
    // Set outputs
    *out_pixel_values = pixel_values;
    *out_count = valid_count;
    *out_coordinates = coordinates;
    
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
