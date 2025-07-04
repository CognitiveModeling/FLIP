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

// Position integral function (area covered by the Gaussian)
// Sample 2D Gaussian
void sample_2d_gaussian_simple(float mu_x, float mu_y, float sigma_x, float sigma_y, float rot_a, float rot_b, float *out_x, float *out_y) {
    // Sample standard Gaussian values using Ziggurat algorithm
    float z_x = (float)r4_nor_value();
    float z_y = (float)r4_nor_value();

    // Scale the standard normal samples by the provided standard deviations
    z_x *= sigma_x;
    z_y *= sigma_y;

    // Apply rotation
    float x_rot = rot_a * z_x - rot_b * z_y;
    float y_rot = rot_b * z_x + rot_a * z_y;

    // Translate by the mean offsets
    *out_x = x_rot + mu_x;
    *out_y = y_rot + mu_y;
}

// Rotate vector
void rotate_vector(float *v, float theta, float *out_rotated) {
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    // Apply rotation
    out_rotated[0] = cos_theta * v[0] - sin_theta * v[1];
    out_rotated[1] = sin_theta * v[0] + cos_theta * v[1];
}

// Apply conservative tracking
void apply_position_augmentation(float *gt_position, float *output_position) {
    // Copy ground truth position
    memcpy(output_position, gt_position, 6 * sizeof(float));
    
    // Create a temporary position for sampling with smaller variance
    float sample_position[6];
    memcpy(sample_position, gt_position, 6 * sizeof(float));
    sample_position[2] *= 0.1f; // scale sigma_x
    sample_position[3] *= 0.1f; // scale sigma_y
    
    // Sample new position with smaller variance
    sample_2d_gaussian_simple(
        sample_position[0], sample_position[1], 
        sample_position[2], sample_position[3], 
        sample_position[4], sample_position[5], 
        output_position, output_position + 1
    );
    
    // Small rotation based on aspect ratio
    float angle_factor = fminf(2.0f/36.0f * fminf(output_position[2], output_position[3]) / 
                          fmaxf(fmaxf(output_position[2], output_position[3]), 1e-8f), 2.0f/36.0f);
    float rotated[2];
    float v[2] = {output_position[4], output_position[5]};
    rotate_vector(v, (r4_uni_value() * 2.0f - 1.0f) * 2.0f * (float)M_PI * angle_factor, rotated);
    output_position[4] = rotated[0];
    output_position[5] = rotated[1];
    
    // Rescale position based on uncertainty
    rescale_position_for_position_augmentation(output_position, angle_factor);
}

// Rescale position for conservative tracking
void rescale_position_for_position_augmentation(float *position, float angular_distance) {
    // Add small noise to the std
    float noise_sigma_x = (float)r4_nor_value() * position[2] * 0.1f;
    float noise_sigma_y = (float)r4_nor_value() * position[3] * 0.1f;
    
    position[2] += fmaxf(fminf(noise_sigma_x, position[2] * 0.3f), -position[2] * 0.3f);
    position[3] += fmaxf(fminf(noise_sigma_y, position[3] * 0.3f), -position[3] * 0.3f);
    
    // Blend between original shape and isotropic shape based on rotation amount
    float alpha = 1.0f - expf(-fabsf(angular_distance));
    float sigma_iso = sqrtf(position[2]*position[2] + position[3]*position[3]);
    position[2] = position[2] * (1.0f - alpha) + sigma_iso * alpha;
    position[3] = position[3] * (1.0f - alpha) + sigma_iso * alpha;
}

// assumes alpha > 1
static inline float sample_gamma(float alpha) {
    // Marsaglia & Tsang method for alpha >= 1.
    float d = alpha - 1.0f / 3.0f;
    float c = 1.0f / sqrtf(9.0f * d);
    float x, v, u;
    while (1) {
        do {
            x = (float)r4_nor_value();
            v = 1.0f + c * x;
        } while (v <= 0.0f);
        v = v * v * v;  // v = (1 + c*x)^3
        u = (float)r4_uni_value();
        if (u < 1.0f - 0.0331f * (x * x * x * x))
            return d * v;
        if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v)))
            return d * v;
    }
}

// Improved sample_num_tokens using proper Beta distribution sampling
int sample_num_tokens(int desired_min, int desired_max, float desired_mean) {
    // Handle edge cases
    if (desired_min == desired_max) {
        return (int)desired_mean;
    }
    
    if (desired_max - desired_min <= 1) {
        return (int)roundf(desired_mean);
    }

    // Calculate the parameters of the Beta distribution
    float alpha = 2.0f;
    float dist_mean = (desired_mean - desired_min) / (desired_max - desired_min);
    float beta = (1.0f/dist_mean - 1.0f) * alpha;

    // Clamp alpha and beta to be positive
    alpha = fmaxf(alpha, 0.001f);
    beta = fmaxf(beta, 0.001f);

    // Sample from the Beta distribution using relationship with Gamma
    float gamma1 = sample_gamma(alpha);
    float gamma2 = sample_gamma(beta);
    float sample = gamma1 / (gamma1 + gamma2);
    
    // Clip to [0,1] to handle floating point errors
    sample = fmaxf(0.0f, fminf(1.0f, sample));
    
    // Scale to the desired range
    int result = desired_min + (int)roundf(sample * (desired_max - desired_min));
    
    // Ensure the result is within bounds
    return fmaxf(fminf(result, desired_max), desired_min);
}

