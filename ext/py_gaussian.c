#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "gaussian.h"

// Python wrapper for the continuous patch sampling function
static PyObject* py_get_continuous_patches_with_indices(PyObject* self, PyObject* args) {
    PyArrayObject *image_obj;
    float mu_x, mu_y, sigma_x, sigma_y, rot_a, rot_b;
    int target_num_patches;
    PyObject *patch_sizes_list;
    float max_overlap_threshold;  
    float coverage;
    
    // Parse arguments: image, mu_x, mu_y, sigma_x, sigma_y, rot_a, rot_b, target_num_patches, patch_sizes (list), overlap_threshold, coverage.
    if (!PyArg_ParseTuple(args, "O!ffffffiOff", 
            &PyArray_Type, &image_obj,
            &mu_x, &mu_y, &sigma_x, &sigma_y, &rot_a, &rot_b,
            &target_num_patches, &patch_sizes_list, 
            &max_overlap_threshold, &coverage))
    {
        return NULL;
    }
    
    // Check that the image is a 3-dimensional uint8 array
    if (PyArray_NDIM(image_obj) != 3 || PyArray_TYPE(image_obj) != NPY_UINT8) 
    {
        PyErr_SetString(PyExc_TypeError, "Image must be a 3-dimensional uint8 numpy array.");
        return NULL;
    }
    
    int H = (int) PyArray_DIM(image_obj, 0);
    int W = (int) PyArray_DIM(image_obj, 1);
    int C = (int) PyArray_DIM(image_obj, 2);
    uint8_t *image_data = (uint8_t*) PyArray_DATA(image_obj);
    
    // Convert patch_sizes_list to a C array
    if (!PyList_Check(patch_sizes_list)) {
        PyErr_SetString(PyExc_TypeError, "patch_sizes must be a list.");
        return NULL;
    }
    
    int num_patch_sizes = (int)PyList_Size(patch_sizes_list);
    float* patch_sizes = (float*) malloc(num_patch_sizes * sizeof(float));
    if (!patch_sizes) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for patch_sizes.");
        return NULL;
    }
    
    for (int i = 0; i < num_patch_sizes; i++) {
        PyObject* item = PyList_GetItem(patch_sizes_list, i); // Borrowed reference
        if (!PyFloat_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "patch_sizes must contain only floats.");
            return NULL;
        }
        patch_sizes[i] = (float)PyFloat_AsDouble(item);
    }
    
    // Call the native function
    uint8_t **patch_buffers = NULL;
    int *patch_counts = NULL;
    float **coordinates = NULL;
    int **target_indices = NULL;
    
    if (extract_continuous_patches_with_indices_native(
            image_data, H, W, C,
            mu_x, mu_y, sigma_x, sigma_y,
            rot_a, rot_b,
            target_num_patches,
            patch_sizes, num_patch_sizes, 
            max_overlap_threshold, coverage,
            &patch_buffers, &patch_counts,
            &coordinates, &target_indices) < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to extract continuous patches with indices.");
        return NULL;
    }
    free(patch_sizes);
    
    /* Create Python lists for the outputs */
    PyObject *patches_list_py = PyList_New(num_patch_sizes);
    PyObject *coords_list_py = PyList_New(num_patch_sizes);
    PyObject *tgt_inds_list_py = PyList_New(num_patch_sizes);
    if (!patches_list_py || !coords_list_py || !tgt_inds_list_py) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create output lists.");
        return NULL;
    }
    
    for (int i = 0; i < num_patch_sizes; i++) {
        // For each patch resolution, retrieve the patch size from the Python patch_sizes_list
        PyObject* item = PyList_GetItem(patch_sizes_list, i);
        float p_d = (float)PyFloat_AsDouble(item);
        int patch_size_int = (int) p_d;
        
        int count = patch_counts[i];
        /* Wrap patch buffer into a NumPy array of shape (count, patch_size, patch_size, 3) */
        npy_intp patch_dims[4] = { count, patch_size_int, patch_size_int, 3 };
        PyObject *patch_arr = PyArray_SimpleNewFromData(4, patch_dims, NPY_UINT8, (void*) patch_buffers[i]);
        if (!patch_arr) {
            PyErr_SetString(PyExc_MemoryError, "Failed to create numpy array for patches.");
            return NULL;
        }
        PyArray_ENABLEFLAGS((PyArrayObject*)patch_arr, NPY_ARRAY_OWNDATA);
        PyList_SET_ITEM(patches_list_py, i, patch_arr);
        
        /* Wrap coordinates: shape (count,2) float64 */
        npy_intp coord_dims[2] = { count, 2 };
        PyObject *coord_arr = PyArray_SimpleNewFromData(2, coord_dims, NPY_FLOAT, (void*) coordinates[i]);
        if (!coord_arr) {
            PyErr_SetString(PyExc_MemoryError, "Failed to create numpy array for coordinates.");
            return NULL;
        }
        PyArray_ENABLEFLAGS((PyArrayObject*)coord_arr, NPY_ARRAY_OWNDATA);
        PyList_SET_ITEM(coords_list_py, i, coord_arr);
        
        /* Wrap target indices: shape (count,) int32 */
        npy_intp tgt_dims[1] = { count };
        PyObject *tgt_arr = PyArray_SimpleNewFromData(1, tgt_dims, NPY_INT32, (void*) target_indices[i]);
        if (!tgt_arr) {
            PyErr_SetString(PyExc_MemoryError, "Failed to create numpy array for target indices.");
            return NULL;
        }
        PyArray_ENABLEFLAGS((PyArrayObject*)tgt_arr, NPY_ARRAY_OWNDATA);
        PyList_SET_ITEM(tgt_inds_list_py, i, tgt_arr);
    }
    
    /* Wrap patch_counts into a Python list */
    PyObject *patch_counts_list_py = PyList_New(num_patch_sizes);
    for (int i = 0; i < num_patch_sizes; i++) {
        PyObject *num_obj = PyLong_FromLong(patch_counts[i]);
        PyList_SET_ITEM(patch_counts_list_py, i, num_obj);
    }
    free(patch_counts);
    free(patch_buffers);
    free(coordinates);
    free(target_indices);
    
    /* Return a tuple: (patches_list, coordinates_list, target_indices_list, patch_counts_list) */
    return Py_BuildValue("NNNN", patches_list_py, coords_list_py, tgt_inds_list_py, patch_counts_list_py);
}

static PyObject* py_sample_mask_with_cropping(PyObject* self, PyObject* args) {
    PyArrayObject *mask_obj;
    float mu_x, mu_y, sigma_x, sigma_y, rot_a, rot_b;
    PyObject *samples_per_group_list;
    
    // Parse arguments: mask, mu_x, mu_y, sigma_x, sigma_y, rot_a, rot_b, samples_per_group (list)
    if (!PyArg_ParseTuple(args, "O!ffffffO", 
            &PyArray_Type, &mask_obj,
            &mu_x, &mu_y, &sigma_x, &sigma_y, &rot_a, &rot_b,
            &samples_per_group_list))
    {
        return NULL;
    }
    
    // Check that the mask is a 2D uint8 array
    if (PyArray_NDIM(mask_obj) != 2 ||
        PyArray_TYPE(mask_obj) != NPY_UINT8) 
    {
        PyErr_SetString(PyExc_TypeError, "Mask must be a 2D uint8 numpy array.");
        return NULL;
    }
    
    int H = (int) PyArray_DIM(mask_obj, 0);
    int W = (int) PyArray_DIM(mask_obj, 1);
    uint8_t *mask_data = (uint8_t*) PyArray_DATA(mask_obj);
    
    // Convert samples_per_group_list to a C array
    if (!PyList_Check(samples_per_group_list)) {
        PyErr_SetString(PyExc_TypeError, "samples_per_group must be a list of integers.");
        return NULL;
    }
    
    int num_groups = (int)PyList_Size(samples_per_group_list);
    if (num_groups != 7) {
        PyErr_SetString(PyExc_ValueError, "samples_per_group must contain exactly 7 values.");
        return NULL;
    }
    
    int* samples_per_group = (int*)malloc(num_groups * sizeof(int));
    if (!samples_per_group) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for samples_per_group.");
        return NULL;
    }
    
    for (int i = 0; i < num_groups; i++) {
        PyObject* item = PyList_GetItem(samples_per_group_list, i);
        if (!PyLong_Check(item)) {
            free(samples_per_group);
            PyErr_SetString(PyExc_TypeError, "samples_per_group must contain only integers.");
            return NULL;
        }
        samples_per_group[i] = (int)PyLong_AsLong(item);
        if (samples_per_group[i] < 0) {
            free(samples_per_group);
            PyErr_SetString(PyExc_ValueError, "samples_per_group values must be non-negative.");
            return NULL;
        }
    }
    
    // Normalize rotation vector if needed
    float rot_norm = sqrtf(rot_a * rot_a + rot_b * rot_b);
    if (rot_norm > 1e-5f) {
        rot_a /= rot_norm;
        rot_b /= rot_norm;
    } else {
        // Default to no rotation if vector is too small
        rot_a = 1.0f;
        rot_b = 0.0f;
    }
    
    // Call the native function
    uint8_t* pixel_values = NULL;
    float* coordinates = NULL;
    int count = 0;
    
    if (sample_mask_with_cropping(
            mask_data, H, W,
            mu_x, mu_y, sigma_x, sigma_y, rot_a, rot_b,
            samples_per_group,
            &pixel_values, &coordinates, &count) < 0)
    {
        free(samples_per_group);
        Py_RETURN_NONE;
    }
    
    free(samples_per_group);
    
    // Handle the case where no pixels were sampled
    if (count == 0) {
        // Create empty arrays with correct shapes
        npy_intp empty_value_dims[1] = {0};
        PyObject *empty_value_arr = PyArray_EMPTY(1, empty_value_dims, NPY_UINT8, 0);
        
        npy_intp empty_coord_dims[2] = {0, 2};
        PyObject *empty_coord_arr = PyArray_EMPTY(2, empty_coord_dims, NPY_FLOAT, 0);
        
        // Build and return the result
        PyObject *result = Py_BuildValue("OOI", empty_value_arr, empty_coord_arr, 0);
        Py_DECREF(empty_value_arr);
        Py_DECREF(empty_coord_arr);
        
        // Free any allocated memory (should be NULL in this case, but being cautious)
        if (pixel_values) free(pixel_values);
        if (coordinates) free(coordinates);
        
        return result;
    }
    
    // Create NumPy arrays for outputs
    npy_intp value_dims[1] = {count};
    PyObject *value_arr = PyArray_SimpleNewFromData(1, value_dims, NPY_UINT8, pixel_values);
    if (!value_arr) {
        free(pixel_values);
        free(coordinates);
        PyErr_SetString(PyExc_MemoryError, "Failed to create NumPy array for pixel values.");
        return NULL;
    }
    PyArray_ENABLEFLAGS((PyArrayObject*)value_arr, NPY_ARRAY_OWNDATA);
    
    npy_intp coord_dims[2] = {count, 2};
    PyObject *coord_arr = PyArray_SimpleNewFromData(2, coord_dims, NPY_FLOAT, coordinates);
    if (!coord_arr) {
        Py_DECREF(value_arr);
        // pixel_values will be freed with value_arr
        free(coordinates);
        PyErr_SetString(PyExc_MemoryError, "Failed to create NumPy array for coordinates.");
        return NULL;
    }
    PyArray_ENABLEFLAGS((PyArrayObject*)coord_arr, NPY_ARRAY_OWNDATA);
    
    // Return (pixel_values, coordinates, count) as a tuple
    PyObject *result = Py_BuildValue("NNI", value_arr, coord_arr, count);
    return result;
}

// Python wrapper for seed initialization
static PyObject* py_initialize_ziggurat(PyObject* self, PyObject* args) {
    int seed = -1;

    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "|i", &seed)) {
        return NULL;
    }

    initialize_ziggurat(seed);

    Py_RETURN_NONE;
}

// Method definitions
static PyMethodDef GaussianMethods[] = {
    {"get_continuous_patches_with_indices", py_get_continuous_patches_with_indices, METH_VARARGS, "Extract continuous patches from an image."},
    {"seed", py_initialize_ziggurat, METH_VARARGS, "Initialize the Ziggurat random number generator."},
    {"sample_mask_with_cropping", py_sample_mask_with_cropping, METH_VARARGS, "Sample pixels from a mask using efficient cropping based on rotated Gaussian parameters."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef gaussianmodule = {
    PyModuleDef_HEAD_INIT,
    "flip_gaussian", // Module name
    NULL,       // Documentation string
    -1,         // Size of per-interpreter state of the module
    GaussianMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_flip_gaussian(void) {
    initialize_ziggurat(-1); // Initialize Ziggurat generator
    import_array();  // Ensure NumPy API is initialized; do this once in your module.
    return PyModule_Create(&gaussianmodule);
}
