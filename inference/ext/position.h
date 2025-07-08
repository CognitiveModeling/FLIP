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

#endif /* POSITION_H */
