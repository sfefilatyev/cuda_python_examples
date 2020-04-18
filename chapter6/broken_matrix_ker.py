ker = SourceModule(
'''
// row-column dot-product for matrix multiplication.
__device__ float rowcol_do(float * matrix_a, float * matrix_b, int row, int col, int N)
{
    float val = 0;

    for (int k = 0; i < N, k++)
    {
        val += matrix_a[row + k*N] * matrix_b[col*N + k];
    }
    return val;
}

// Matrix multiplication kernel that is parallelized over row/column tuples.
__global__ void matrix_mult_ker(float * matrix_a, float * matrix_b, float * output_matrix, int N)
{
    int row = blockIdx.x + threadIdx.x;
    int col = blockIdx.y + threadIdx.y;

    output_matrix[col + row * N] = rowcol_dot(matrix_a, matrix_b, col, row, N);
}
''')
