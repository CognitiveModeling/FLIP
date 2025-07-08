// bounding_box.c
#include "bbox.h"
#include <string.h>  // For memset
#include <math.h>    // For ceil, floor

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

// Initialize a memory pool
static inline MemoryPool* memory_pool_init(size_t size) {
    MemoryPool* pool = (MemoryPool*)malloc(sizeof(MemoryPool));
    if (!pool) return NULL;
    
    pool->buffer = (char*)malloc(size);
    if (!pool->buffer) {
        free(pool);
        return NULL;
    }
    
    pool->size = size;
    pool->used = 0;
    
    return pool;
}

// Allocate memory from the memory pool
static inline void* memory_pool_alloc(MemoryPool* pool, size_t size) {
    // Align to 8-byte boundary for better performance
    size_t aligned_size = (size + 7) & ~7;
    
    if (pool->used + aligned_size > pool->size) {
        return NULL;  // Not enough space
    }
    
    void* ptr = pool->buffer + pool->used;
    pool->used += aligned_size;
    
    return ptr;
}

// Free memory pool
static inline void memory_pool_free(MemoryPool* pool) {
    if (pool) {
        if (pool->buffer) {
            free(pool->buffer);
        }
        free(pool);
    }
}

// Hash function for cell coordinates
static inline unsigned int hash_cell_coords(int x, int y, int hash_size) {
    unsigned int hash = (unsigned int)((x * 73856093) ^ (y * 19349663));
    return hash % hash_size;
}

// Find or create a cell at the given coordinates
static inline SpatialHashCell* get_or_create_cell(SpatialHashGrid* grid, int x, int y) {
    unsigned int hash = hash_cell_coords(x, y, grid->hash_size);
    
    // Look for existing cell
    SpatialHashCell* cell = grid->hash_table[hash];
    while (cell) {
        if (cell->cell_x == x && cell->cell_y == y) {
            return cell;
        }
        cell = cell->next;
    }
    
    // Cell doesn't exist, create a new one
    SpatialHashCell* new_cell = (SpatialHashCell*)memory_pool_alloc(grid->memory, sizeof(SpatialHashCell));
    if (!new_cell) return NULL;
    
    new_cell->cell_x = x;
    new_cell->cell_y = y;
    new_cell->boxes = NULL;
    
    // Insert at the beginning of the hash bucket
    new_cell->next = grid->hash_table[hash];
    grid->hash_table[hash] = new_cell;
    
    return new_cell;
}

// Find a cell if it exists
static inline SpatialHashCell* find_cell(SpatialHashGrid* grid, int x, int y) {
    unsigned int hash = hash_cell_coords(x, y, grid->hash_size);
    
    SpatialHashCell* cell = grid->hash_table[hash];
    while (cell) {
        if (cell->cell_x == x && cell->cell_y == y) {
            return cell;
        }
        cell = cell->next;
    }
    
    return NULL;  // Cell not found
}

// Function to find the index of the most requested patch size
static inline int find_most_requested_patch_index(int *num_patches_per_resolution, int num_patch_sizes) {
    int max_index = 0;
    int max_count = num_patches_per_resolution[0];
    
    for (int i = 1; i < num_patch_sizes; i++) {
        if (num_patches_per_resolution[i] > max_count) {
            max_count = num_patches_per_resolution[i];
            max_index = i;
        }
    }
    
    return max_index;
}


// Initialize a spatial hash grid with automatic memory allocation
SpatialHashGrid* spatial_hash_init(float *patch_sizes, int num_patch_sizes, int *num_patches_per_resolution, int max_overlaps) {

    int most_requested_index = find_most_requested_patch_index(num_patches_per_resolution, num_patch_sizes);
    float most_requested_patch_size = patch_sizes[most_requested_index];
    float cell_size = most_requested_patch_size / 2.0f;  // Half the most requested patch size
    
    // Estimate total number of patches
    int total_patches = 0;
    int total_cells = 0;
    for (int i = 0; i < num_patch_sizes; i++) {
        total_patches += num_patches_per_resolution[i];
    
        // Calculate how many cells a patch can span (worst case)
        int cells_per_dimension = (int)ceil(patch_sizes[i] / cell_size) + 1;
        int max_cells_per_patch = cells_per_dimension * cells_per_dimension;

        total_cells += max_cells_per_patch * num_patches_per_resolution[i];
    }
    
    // Determine hash table size (next power of 2 greater than or equal to estimated entries)
    int num_cells = max((int) ceil(total_cells / 0.75f), 16);  // Load factor 0.75
    int hash_size = 1 << (size_t)ceil(log2(num_cells));
    
    // Estimate memory needed for data structures
    size_t grid_memory = sizeof(SpatialHashGrid);
    size_t hash_table_memory = hash_size * sizeof(SpatialHashCell*);
    size_t cell_memory = sizeof(SpatialHashCell) * total_cells;
    size_t box_memory = (total_patches + total_cells * max_overlaps * 2) * sizeof(BoundingBox);
    
    // Total memory with safety overhead (20%)
    size_t total_memory = (grid_memory + hash_table_memory + cell_memory + box_memory) * 1.2;
    
    // Create memory pool
    MemoryPool* pool = memory_pool_init(total_memory);
    if (!pool) return NULL;
    
    // Allocate grid
    SpatialHashGrid* grid = (SpatialHashGrid*)memory_pool_alloc(pool, sizeof(SpatialHashGrid));
    if (!grid) {
        memory_pool_free(pool);
        return NULL;
    }
    
    // Allocate hash table
    grid->hash_table = (SpatialHashCell**)memory_pool_alloc(pool, hash_size * sizeof(SpatialHashCell*));
    if (!grid->hash_table) {
        memory_pool_free(pool);
        return NULL;
    }
    
    // Initialize hash table
    memset(grid->hash_table, 0, hash_size * sizeof(SpatialHashCell*));
    
    grid->hash_size = hash_size;
    grid->cell_size = cell_size;
    grid->memory = pool;
    
    return grid;
}

// Free the spatial hash grid and its memory
void spatial_hash_free(SpatialHashGrid* grid) {
    if (grid) {
        memory_pool_free(grid->memory);
        // The grid itself was allocated from the pool, so it's freed automatically
    }
}

// Add a box to the spatial hash grid
void spatial_hash_add_box(SpatialHashGrid* grid, float x_center, float y_center, float patch_size) {
    // Calculate half patch size
    float half_size = patch_size / 2.0f;
    
    // Create new box
    BoundingBox *box = (BoundingBox*)memory_pool_alloc(grid->memory, sizeof(BoundingBox));
    box->x_min = x_center - half_size;
    box->y_min = y_center - half_size;
    box->x_max = x_center + half_size;
    box->y_max = y_center + half_size;
    
    // Calculate the cell ranges
    int min_cell_x = (int)floorf(box->x_min / grid->cell_size);
    int min_cell_y = (int)floorf(box->y_min / grid->cell_size);
    int max_cell_x = (int)floorf(box->x_max / grid->cell_size);
    int max_cell_y = (int)floorf(box->y_max / grid->cell_size);
    
    // Add the box to all overlapping cells
    for (int cy = min_cell_y; cy <= max_cell_y; cy++) {
        for (int cx = min_cell_x; cx <= max_cell_x; cx++) {
            SpatialHashCell* cell = get_or_create_cell(grid, cx, cy);
            if (cell) {
                box->next = cell->boxes;
                cell->boxes = box;
            }
        }
    }
}

// Check if a new box would overlap with any existing boxes
int spatial_hash_has_overlap(SpatialHashGrid* grid, float x_center, float y_center, float patch_size) {
    float half_size = patch_size / 2.0f;
    
    // Create temporary box for the new patch
    float x_min = x_center - half_size;
    float y_min = y_center - half_size;
    float x_max = x_center + half_size;
    float y_max = y_center + half_size;
    
    // Calculate the cell ranges for this box
    int min_cell_x = (int)floorf(x_min / grid->cell_size);
    int min_cell_y = (int)floorf(y_min / grid->cell_size);
    int max_cell_x = (int)floorf(x_max / grid->cell_size);
    int max_cell_y = (int)floorf(y_max / grid->cell_size);
    
    // Check all potentially overlapping boxes
    for (int cy = min_cell_y; cy <= max_cell_y; cy++) {
        for (int cx = min_cell_x; cx <= max_cell_x; cx++) {
            SpatialHashCell* cell = find_cell(grid, cx, cy);
            if (!cell) continue;  // Cell doesn't exist, skip
            
            // Check each box in this cell
            BoundingBox* box = cell->boxes;
            while (box) {
                
                // Check for overlap
                if (!(x_max < box->x_min + BB_EPSILON || 
                      x_min > box->x_max - BB_EPSILON || 
                      y_max < box->y_min + BB_EPSILON || 
                      y_min > box->y_max - BB_EPSILON)) {
                    return 1; // Overlap detected
                }

                box = box->next;
            }
        }
    }
    
    return 0; // No overlaps found
}

// Calculate overlap percentage
int spatial_hash_has_significant_overlap(SpatialHashGrid* grid, float x_center, float y_center, float patch_size, float threshold) {
    float half_size = patch_size / 2.0f;
    float patch_area = patch_size * patch_size;
    float max_allowed_overlap = patch_area * threshold;
    float total_overlap_area = 0.0f;
    
    // Create temporary box for the new patch
    float x_min = x_center - half_size;
    float y_min = y_center - half_size;
    float x_max = x_center + half_size;
    float y_max = y_center + half_size;
    
    // Calculate the cell ranges for this box
    int min_cell_x = (int)floorf(x_min / grid->cell_size);
    int min_cell_y = (int)floorf(y_min / grid->cell_size);
    int max_cell_x = (int)floorf(x_max / grid->cell_size);
    int max_cell_y = (int)floorf(y_max / grid->cell_size);
    
    // Check all potentially overlapping boxes
    for (int cy = min_cell_y; cy <= max_cell_y; cy++) {
        for (int cx = min_cell_x; cx <= max_cell_x; cx++) {
            SpatialHashCell* cell = find_cell(grid, cx, cy);
            if (!cell) continue;  // Cell doesn't exist, skip
            
            // Check each box in this cell
            BoundingBox* box = cell->boxes;
            while (box) {
            
                // Check for overlap
                if (!(x_max < box->x_min + BB_EPSILON || 
                      x_min > box->x_max - BB_EPSILON || 
                      y_max < box->y_min + BB_EPSILON || 
                      y_min > box->y_max - BB_EPSILON)) {
                    
                    // Calculate the overlap rectangle
                    float overlap_x_min = max(x_min, box->x_min);
                    float overlap_y_min = max(y_min, box->y_min);
                    float overlap_x_max = min(x_max, box->x_max);
                    float overlap_y_max = min(y_max, box->y_max);
                    
                    // Calculate the overlap area
                    float overlap_width = overlap_x_max - overlap_x_min;
                    float overlap_height = overlap_y_max - overlap_y_min;
                    float overlap_area = overlap_width * overlap_height;
                    
                    // Add to the total overlap area
                    total_overlap_area += overlap_area;
                    
                    // Early termination if overlap exceeds threshold
                    if (total_overlap_area > max_allowed_overlap) {
                        return 1;
                    }
                }

                box = box->next;
            }
        }
    }
    
    // If we got here, overlap is below threshold
    return 0;
}
