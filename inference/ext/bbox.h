// bounding_box.h
#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Epsilon for floating-point comparisons
#define BB_EPSILON 1e-6f

// Memory pool structure
typedef struct {
    char* buffer;           // Pre-allocated memory buffer
    size_t size;            // Total size of the buffer
    size_t used;            // Amount of memory currently used
} MemoryPool;


// Bounding box structure
typedef struct BoundingBox {
    float x_min;   // Left edge
    float y_min;   // Top edge
    float x_max;   // Right edge
    float y_max;   // Bottom edge
    struct BoundingBox* next;  // Next box in the list
} BoundingBox;


// Spatial hash table cell
typedef struct SpatialHashCell {
    int cell_x;                     // Cell X coordinate
    int cell_y;                     // Cell Y coordinate
    BoundingBox* boxes;             // Pointer to box data
    struct SpatialHashCell* next;  // Next cell in hash bucket
} SpatialHashCell;

// Spatial hash grid
typedef struct {
    SpatialHashCell** hash_table;  // Hash table
    int hash_size;                  // Size of hash table
    float cell_size;                // Size of each cell
    MemoryPool* memory;             // Memory pool
} SpatialHashGrid;

// Initialize a spatial hash grid
SpatialHashGrid* spatial_hash_init(float *patch_sizes, int num_patch_sizes, int *num_patches_per_resolution, int max_overlaps);

// Add a box to the spatial hash grid
void spatial_hash_add_box(SpatialHashGrid* grid, float x_center, float y_center, float patch_size);

// Check if a new box would overlap with any existing boxes
int spatial_hash_has_overlap(SpatialHashGrid* grid, float x_center, float y_center, float patch_size);

// Calculate the overlap between a new box and existing boxes
int spatial_hash_has_significant_overlap(SpatialHashGrid* grid, float x_center, float y_center, float patch_size, float threshold);

// Free the spatial hash grid and its memory
void spatial_hash_free(SpatialHashGrid* grid);

#endif /* BOUNDING_BOX_H */
