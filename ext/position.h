// position.h
#ifndef POSITION_H
#define POSITION_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "gaussian.h"

// Function to compute position rotation from rho representation
void compute_position_rot_from_rho(float *position_rho, float *output_position);

// Position augmentation functions
void apply_position_augmentation(float *gt_position, float *output_position);

// Helper functions for position augmentation
void rescale_position_for_position_augmentation(float *position, float angular_distance);

// Function to sample 2D Gaussian points
void sample_2d_gaussian_simple(float mu_x, float mu_y, float sigma_x, float sigma_y, float rot_a, float rot_b, float *out_x, float *out_y);

// Function to rotate a vector
void rotate_vector(float *v, float theta, float *out_rotated);

// Function to sample number of tokens with a beta distribution
int sample_num_tokens(int desired_min, int desired_max, float desired_mean);


#endif /* POSITION_H */
