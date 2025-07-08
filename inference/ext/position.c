// position.c
#include "position.h"
#include <string.h>

// Compute position rotation from rho representation
void compute_position_rot_from_rho(float *position_rho, float *output_position) {
    float mu_x = position_rho[0];
    float mu_y = position_rho[1];
    float sigma_x2 = position_rho[2];
    float sigma_y2 = position_rho[3];
    float sigma_xy = position_rho[4];
    
    // Eigen decomposition
    float term = sqrtf(fmaxf((sigma_x2 - sigma_y2) * (sigma_x2 - sigma_y2) + 4.0f * sigma_xy * sigma_xy, 1e-16f));
    float sigma_x = sqrtf(fmaxf((sigma_x2 + sigma_y2) / 2.0f - term / 2.0f, 1e-16f));
    float sigma_y = sqrtf(fmaxf((sigma_x2 + sigma_y2) / 2.0f + term / 2.0f, 1e-16f));
    
    // Compute eigenvectors
    float rot_b = 1.0f;
    float rot_a = 2.0f * sigma_xy * rot_b / fmaxf(sigma_x2 - sigma_y2 + term, 1e-16f);

    // Normalize rotation components
    float scale = sqrtf(fmaxf(rot_a * rot_a + rot_b * rot_b, 1e-16f));
    rot_a /= scale;
    rot_b /= scale;
    
    // Set output
    output_position[0] = mu_x;
    output_position[1] = mu_y;
    output_position[2] = sigma_x;
    output_position[3] = sigma_y;
    output_position[4] = rot_a;
    output_position[5] = rot_b;
}

