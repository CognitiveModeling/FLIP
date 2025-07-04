// position.c
#include "position.h"
#include <string.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


// Python wrapper for compute_position_rot_from_rho
static PyObject* py_compute_position_rot_from_rho(PyObject* self, PyObject* args) {
    PyObject* position_rho_obj;
    
    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "O", &position_rho_obj)) {
        return NULL;
    }
    
    // Convert to numpy array
    PyArrayObject* position_rho_array = (PyArrayObject*)PyArray_FROM_OTF(position_rho_obj, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
    if (position_rho_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "Expected a numpy array for position_rho");
        return NULL;
    }
    
    // Check dimensions
    if (PyArray_NDIM(position_rho_array) != 2 || PyArray_DIM(position_rho_array, 1) != 5) {
        PyErr_SetString(PyExc_ValueError, "Expected position_rho to be a 2D array with 5 columns");
        Py_DECREF(position_rho_array);
        return NULL;
    }
    
    int num_positions = (int)PyArray_DIM(position_rho_array, 0);
    
    // Create output array
    npy_intp dims[2] = {num_positions, 6};
    PyArrayObject* output_array = (PyArrayObject*)PyArray_EMPTY(2, dims, NPY_FLOAT, 0);
    if (output_array == NULL) {
        Py_DECREF(position_rho_array);
        return NULL;
    }
    
    // Process each position
    for (int i = 0; i < num_positions; i++) {
        float* position_rho = (float*)PyArray_GETPTR2(position_rho_array, i, 0);
        float* output_position = (float*)PyArray_GETPTR2(output_array, i, 0);
        
        compute_position_rot_from_rho(position_rho, output_position);
    }
    
    Py_DECREF(position_rho_array);
    return PyArray_Return(output_array);
}

// Python wrapper for apply_position_augmentation
static PyObject* py_apply_position_augmentation(PyObject* self, PyObject* args) {
    PyObject* gt_position_obj;
    
    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "O", &gt_position_obj)) {
        return NULL;
    }
    
    // Convert to numpy array
    PyArrayObject* gt_position_array = (PyArrayObject*)PyArray_FROM_OTF(gt_position_obj, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
    if (gt_position_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "Expected a numpy array for gt_position");
        return NULL;
    }
    
    // Check dimensions
    if (PyArray_NDIM(gt_position_array) != 1 || PyArray_DIM(gt_position_array, 0) != 6) {
        PyErr_SetString(PyExc_ValueError, "Expected gt_position to be a 1D array with 6 elements");
        Py_DECREF(gt_position_array);
        return NULL;
    }
    
    // Create output array
    npy_intp dims[1] = {6};
    PyArrayObject* output_array = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_FLOAT, 0);
    if (output_array == NULL) {
        Py_DECREF(gt_position_array);
        return NULL;
    }
    
    // Get pointers to the data
    float* gt_position = (float*)PyArray_DATA(gt_position_array);
    float* output_position = (float*)PyArray_DATA(output_array);
    
    // Apply conservative tracking
    apply_position_augmentation(gt_position, output_position);
    
    Py_DECREF(gt_position_array);
    return PyArray_Return(output_array);
}

// Python wrapper for sample_num_tokens
static PyObject* py_sample_num_tokens(PyObject* self, PyObject* args) {
    int desired_min, desired_max;
    float desired_mean;
    
    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "iif", &desired_min, &desired_max, &desired_mean)) {
        return NULL;
    }
    
    // Sample the number of tokens
    int num_tokens = sample_num_tokens(desired_min, desired_max, desired_mean);
    
    // Return the result
    return PyLong_FromLong(num_tokens);
}

// Python wrapper for sample_continuous_patches
static PyObject* py_sample_continuous_patches(PyObject* self, PyObject* args) {
    PyArrayObject *image_obj;
    PyObject *position_obj;
    int num_tokens;
    PyObject *patch_sizes_list;
    float max_overlap_threshold = -1.0;  // Default value indicating "use random"
    float coverage = -1.0;               // Default value indicating "use random"
    
    // Parse arguments
    if (!PyArg_ParseTuple(args, "OOiO|ff", 
                         &image_obj, 
                         &position_obj,
                         &num_tokens,
                         &patch_sizes_list,
                         &max_overlap_threshold,
                         &coverage)) {
        return NULL;
    }
    
    // Check that image is a 3D uint8 array
    if (PyArray_NDIM(image_obj) != 3 || 
        PyArray_TYPE(image_obj) != NPY_UINT8 || 
        PyArray_DIM(image_obj, 2) != 3) {
        PyErr_SetString(PyExc_TypeError, "Image must be a 3-channel uint8 numpy array.");
        return NULL;
    }
    
    // Check that position is a 1D array with 6 elements
    PyArrayObject* position_array = (PyArrayObject*)PyArray_FROM_OTF(position_obj, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
    if (position_array == NULL || 
        PyArray_NDIM(position_array) != 1 || 
        PyArray_DIM(position_array, 0) != 6) {
        PyErr_SetString(PyExc_TypeError, "Position must be a 1D array with 6 elements.");
        Py_XDECREF(position_array);
        return NULL;
    }
    
    // Check that patch_sizes is a list
    if (!PyList_Check(patch_sizes_list)) {
        PyErr_SetString(PyExc_TypeError, "patch_sizes must be a list.");
        Py_DECREF(position_array);
        return NULL;
    }
    
    int num_patch_sizes = (int)PyList_Size(patch_sizes_list);
    float* patch_sizes = (float*)malloc(num_patch_sizes * sizeof(float));
    if (!patch_sizes) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for patch_sizes.");
        Py_DECREF(position_array);
        return NULL;
    }
    
    for (int i = 0; i < num_patch_sizes; i++) {
        PyObject* item = PyList_GetItem(patch_sizes_list, i);
        if (!PyFloat_Check(item) && !PyLong_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "patch_sizes must contain only numbers.");
            free(patch_sizes);
            Py_DECREF(position_array);
            return NULL;
        }
        patch_sizes[i] = PyFloat_AsDouble(item);
    }
    
    // Get image dimensions and data
    int H = (int)PyArray_DIM(image_obj, 0);
    int W = (int)PyArray_DIM(image_obj, 1);
    int C = (int)PyArray_DIM(image_obj, 2);
    uint8_t* image_data = (uint8_t*)PyArray_DATA(image_obj);
    
    // Get position data
    float* position = (float*)PyArray_DATA(position_array);
    float mu_x = position[0];
    float mu_y = position[1];
    float sigma_x = position[2];
    float sigma_y = position[3];
    float rot_a = position[4];
    float rot_b = position[5];
    
    // Apply sqrtf(2)x sigma for sampling (as in original sample_patches)
    sigma_x *= sqrtf(2);
    sigma_y *= sqrtf(2);
    
    // Convert from normalized coordinates (-1 to 1) to image coordinates
    mu_x = (mu_x + 1) * W / 2;
    mu_y = (mu_y + 1) * H / 2;
    sigma_x = sigma_x * W / 2;
    sigma_y = sigma_y * H / 2;
    
    // Flip rotation for sampling (as in the original code)
    rot_a = -rot_a;
    
    // Use provided parameters or generate random ones
    if (max_overlap_threshold < 0) {
        //max_overlap_threshold = powf(r4_uni_value(), 3) * 4.0;
        max_overlap_threshold = r4_uni_value() * 4.0;
    }
    if (coverage < 0) {
        //coverage = 0.1 + powf(r4_uni_value(), 3) * 1.9;
        coverage = 0.1 + r4_uni_value() * 1.9;
    }
    
    // Call the gaussian C extension for continuous patch sampling
    uint8_t **patch_buffers = NULL;
    int *patch_counts = NULL;
    float **coordinates = NULL;
    int **target_indices = NULL;
    
    if (extract_continuous_patches_with_indices_native(
            image_data, H, W, C,
            mu_x, mu_y, sigma_x, sigma_y, rot_a, rot_b,
            num_tokens, patch_sizes, num_patch_sizes,
            max_overlap_threshold, coverage,
            &patch_buffers, &patch_counts,
            &coordinates, &target_indices) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to extract patches.");
        free(patch_sizes);
        Py_DECREF(position_array);
        return NULL;
    }
    
    // Create Python lists for the outputs
    PyObject *patches_list = PyList_New(num_patch_sizes);
    PyObject *coords_list = PyList_New(num_patch_sizes);
    PyObject *indices_list = PyList_New(num_patch_sizes);
    PyObject *lengths_list = PyList_New(num_patch_sizes);
    
    if (!patches_list || !coords_list || !indices_list || !lengths_list) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create output lists.");
        free(patch_sizes);
        Py_DECREF(position_array);
        return NULL;
    }
    
    // Fill the lists with numpy arrays
    for (int i = 0; i < num_patch_sizes; i++) {
        int p = (int)patch_sizes[i];
        int count = patch_counts[i];
        
        // Create numpy array for patches
        npy_intp patch_dims[4] = {count, p, p, 3};
        PyObject *patch_array = PyArray_SimpleNewFromData(4, patch_dims, NPY_UINT8, patch_buffers[i]);
        if (!patch_array) {
            PyErr_SetString(PyExc_MemoryError, "Failed to create numpy array for patches.");
            free(patch_sizes);
            Py_DECREF(position_array);
            Py_DECREF(patches_list);
            Py_DECREF(coords_list);
            Py_DECREF(indices_list);
            Py_DECREF(lengths_list);
            return NULL;
        }
        PyArray_ENABLEFLAGS((PyArrayObject*)patch_array, NPY_ARRAY_OWNDATA);
        PyList_SetItem(patches_list, i, patch_array);
        
        // Create numpy array for coordinates
        npy_intp coord_dims[2] = {count, 2};
        PyObject *coord_array = PyArray_SimpleNewFromData(2, coord_dims, NPY_FLOAT, coordinates[i]);
        if (!coord_array) {
            PyErr_SetString(PyExc_MemoryError, "Failed to create numpy array for coordinates.");
            free(patch_sizes);
            Py_DECREF(position_array);
            Py_DECREF(patches_list);
            Py_DECREF(coords_list);
            Py_DECREF(indices_list);
            Py_DECREF(lengths_list);
            return NULL;
        }
        PyArray_ENABLEFLAGS((PyArrayObject*)coord_array, NPY_ARRAY_OWNDATA);
        PyList_SetItem(coords_list, i, coord_array);
        
        // Create numpy array for target indices
        npy_intp index_dims[1] = {count};
        PyObject *index_array = PyArray_SimpleNewFromData(1, index_dims, NPY_INT32, target_indices[i]);
        if (!index_array) {
            PyErr_SetString(PyExc_MemoryError, "Failed to create numpy array for indices.");
            free(patch_sizes);
            Py_DECREF(position_array);
            Py_DECREF(patches_list);
            Py_DECREF(coords_list);
            Py_DECREF(indices_list);
            Py_DECREF(lengths_list);
            return NULL;
        }
        PyArray_ENABLEFLAGS((PyArrayObject*)index_array, NPY_ARRAY_OWNDATA);
        PyList_SetItem(indices_list, i, index_array);
        
        // Create tensor for sequence length
        PyObject *length_tensor = PyLong_FromLong(count);
        PyList_SetItem(lengths_list, i, length_tensor);
    }
    
    // Clean up
    free(patch_sizes);
    free(patch_buffers);
    free(coordinates);
    free(target_indices);
    free(patch_counts);
    Py_DECREF(position_array);
    
    // Return the tuple (patches, coordinates, indices, seq_lengths)
    PyObject *result = Py_BuildValue("NNNN", patches_list, coords_list, indices_list, lengths_list);
    return result;
}

// Method definitions
static PyMethodDef PositionMethods[] = {
    {"compute_position_rot_from_rho", py_compute_position_rot_from_rho, METH_VARARGS, "Compute position rotation from rho representation."},
    {"apply_position_augmentation", py_apply_position_augmentation, METH_VARARGS, "Apply conservative tracking to a position."},
    {"sample_num_tokens", py_sample_num_tokens, METH_VARARGS, "Sample the number of tokens from a beta distribution."},
    {"sample_continuous_patches", py_sample_continuous_patches, METH_VARARGS, "Sample continuous patches from an image."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef positionmodule = {
    PyModuleDef_HEAD_INIT,
    "flip_position",      // Module name
    NULL,           // Documentation
    -1,             // Size of per-interpreter state or -1
    PositionMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_flip_position(void) {
    // Initialize numpy array API
    import_array();
    
    initialize_ziggurat(-1); // Initialize Ziggurat generator
    return PyModule_Create(&positionmodule);
}
