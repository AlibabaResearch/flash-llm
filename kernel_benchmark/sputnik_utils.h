/***************************************************************************
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
// Note: this file is extended from Sputnik repo

#ifndef THIRD_PARTY_SPUTNIK_MATRIX_UTILS_H_
#define THIRD_PARTY_SPUTNIK_MATRIX_UTILS_H_

/**
 * @file @brief Utilities for creating sparse and dense matrices for
 * tests and benchmarks.
 */
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "sputnik/cuda_utils.h"
#include "sputnik/test_utils.h"
#include "sputnik/type_utils.h"

namespace sputnik_utils {

/**
 * @brief Type conversion utilities.
 */
template<typename In, typename Out>
cudaError_t Convert(const In* in, Out* out, int n);

/**
 * @brief Create a row swizzle that maps thread blocks to rows in order.
 */
void IdentityRowSwizzle(int rows, const int* row_offsets, int* row_indices);

/**
 * @brief Create a row swizzle that maps thread blocks to rows in order of
 * decreasing size.
 */
void SortedRowSwizzle(int rows, const int* row_offsets, int* row_indices);

/**
 * @brief Enumeration of different row swizzles we can create.
 */
enum Swizzle {
    // Do not reorder the rows at all.
    IDENTITY = 0,
    // Sort the rows by size s.t. we process larger rows first.
    SORTED = 1
};

/**
 * @brief Enumeration of different sparse matrices that we can create.
 */
enum ElementDistribution {
    // Randomly sample which weights to zero with uniform probability.
    // Rows can have different numbers of non-zero weights, but they
    // have the same expected number of non-zero weights.
    RANDOM_UNIFORM = 0,
    // Divide the weights evenly across the rows and then randomly sample
    // which weights should be sets to zero in each row. All rows have
    // exactly the same number of non-zero weights.
    PERFECT_UNIFORM = 1
};

// Prototype for CudaSparseMatrix s.t. we can reference in SparseMatrix API.
template<typename Value>
class CudaSparseMatrix;

/**
 * @brief Simple sparse-matrix class to for managing pointers and memory
 * allocation/deallocation.
 */
class SparseMatrix {
public:
    SparseMatrix(int rows, int columns, float* originals, Swizzle row_swizzle = IDENTITY, int pad_rows_to = 4);
    /**
     * @brief Construct a sparse matrix from a CUDA sparse matrix.
     */
    explicit SparseMatrix(const CudaSparseMatrix<float>& sparse_matrix);

    /**
     * @brief Cleanup the underlying storage.
     */
    ~SparseMatrix()
    {
        delete[] values_;
        delete[] row_offsets_;
        delete[] column_indices_;
        delete[] row_indices_;
    }

    SparseMatrix(const SparseMatrix&) = delete;
    SparseMatrix& operator=(const SparseMatrix&) = delete;
    SparseMatrix(SparseMatrix&&)                 = delete;
    SparseMatrix& operator=(SparseMatrix&&) = delete;

    const float* Values() const
    {
        return values_;
    }
    float* Values()
    {
        return values_;
    }

    const int* RowOffsets() const
    {
        return row_offsets_;
    }
    int* RowOffsets()
    {
        return row_offsets_;
    }

    const int* ColumnIndices() const
    {
        return column_indices_;
    }
    int* ColumnIndices()
    {
        return column_indices_;
    }

    const int* RowIndices() const
    {
        return row_indices_;
    }
    int* RowIndices()
    {
        return row_indices_;
    }

    int Rows() const
    {
        return rows_;
    }

    int Columns() const
    {
        return columns_;
    }

    int Nonzeros() const
    {
        return nonzeros_;
    }

    int PadRowsTo() const
    {
        return pad_rows_to_;
    }

    int NumElementsWithPadding() const
    {
        return num_elements_with_padding_;
    }

    ElementDistribution WeightDistribution() const
    {
        return weight_distribution_;
    }

    Swizzle RowSwizzle() const
    {
        return row_swizzle_;
    }

protected:
    SparseMatrix():
        values_(nullptr),
        row_offsets_(nullptr),
        column_indices_(nullptr),
        row_indices_(nullptr),
        rows_(0),
        columns_(0),
        nonzeros_(0),
        pad_rows_to_(0),
        num_elements_with_padding_(0),
        weight_distribution_(RANDOM_UNIFORM),
        row_swizzle_(IDENTITY)
    {
    }

    // Matrix value and index storage.
    float* values_;
    int*   row_offsets_;
    int*   column_indices_;

    // Swizzled row indices for load balancing.
    int* row_indices_;

    // Matrix meta-data.
    int                 rows_, columns_, nonzeros_;
    int                 pad_rows_to_, num_elements_with_padding_;
    ElementDistribution weight_distribution_;
    Swizzle             row_swizzle_;

    void InitFromCudaSparseMatrix(const CudaSparseMatrix<float>& sparse_matrix);
};

/**
 * @brief Simple gpu sparse-matrix class to for managing pointers and
 * memory allocation/deallocation.
 */
template<typename Value>
class CudaSparseMatrix {
public:
    /**
     * @brief Create a sparse matrix with the specified properties.
     *
     * @param row The number of rows in the matrix.
     * @param columns The number of columns in the matrix.
     * @param nonzeros The number of nonzero values in the matrix.
     * @param weight_distribution The distribution of non-zero weights
     * across the rows of the matrix.
     * @param row_swizzle The type of row swizzle to apply. Defaults to
     * IDENTITY.
     * @param pad_to_rows Each row in the sparse matrix will be padded to a
     * multiple of this value. Defaults to 4, which enables the user of
     * 4-element vector loads and stores. For best performance, pad to
     * `kBlockItemsK`.
     */

    /**
     * @brief Construct a CUDA sparse matrix from a host sparse matrix.
     */
    explicit CudaSparseMatrix(const SparseMatrix& sparse_matrix);

    /**
     * @brief Cleanup the underlying storage.
     */
    ~CudaSparseMatrix()
    {
        CUDA_CALL(cudaFree(values_));
        CUDA_CALL(cudaFree(row_offsets_));
        CUDA_CALL(cudaFree(column_indices_));
        CUDA_CALL(cudaFree(row_indices_));
    }

    CudaSparseMatrix(const CudaSparseMatrix&) = delete;
    CudaSparseMatrix& operator=(const CudaSparseMatrix&) = delete;
    CudaSparseMatrix(CudaSparseMatrix&&)                 = delete;
    CudaSparseMatrix& operator=(CudaSparseMatrix&&) = delete;

    // Datatype for indices in this matrix.
    typedef typename sputnik::Value2Index<Value>::Index Index;

    const Value* Values() const
    {
        return values_;
    }
    Value* Values()
    {
        return values_;
    }

    const int* RowOffsets() const
    {
        return row_offsets_;
    }
    int* RowOffsets()
    {
        return row_offsets_;
    }

    const Index* ColumnIndices() const
    {
        return column_indices_;
    }
    Index* ColumnIndices()
    {
        return column_indices_;
    }

    const int* RowIndices() const
    {
        return row_indices_;
    }
    int* RowIndices()
    {
        return row_indices_;
    }

    int Rows() const
    {
        return rows_;
    }

    int Columns() const
    {
        return columns_;
    }

    int Nonzeros() const
    {
        return nonzeros_;
    }

    int PadRowsTo() const
    {
        return pad_rows_to_;
    }

    int NumElementsWithPadding() const
    {
        return num_elements_with_padding_;
    }

    ElementDistribution WeightDistribution() const
    {
        return weight_distribution_;
    }

    Swizzle RowSwizzle() const
    {
        return row_swizzle_;
    }

protected:
    CudaSparseMatrix():
        values_(nullptr),
        row_offsets_(nullptr),
        column_indices_(nullptr),
        row_indices_(nullptr),
        rows_(0),
        columns_(0),
        nonzeros_(0),
        pad_rows_to_(0),
        num_elements_with_padding_(0),
        weight_distribution_(RANDOM_UNIFORM),
        row_swizzle_(IDENTITY)
    {
    }

    // Matrix value and index storage.
    Value* values_;
    int*   row_offsets_;
    Index* column_indices_;

    // Swizzled row indices for load balancing.
    int* row_indices_;

    // Matrix meta-data.
    int                 rows_, columns_, nonzeros_;
    int                 pad_rows_to_, num_elements_with_padding_;
    ElementDistribution weight_distribution_;
    Swizzle             row_swizzle_;

    void InitFromSparseMatrix(const SparseMatrix& sparse_matrix);
};

/**
 * @brief Helper to load sparse matrix values into a std::vector.
 */
inline std::vector<float> ToVector(const SparseMatrix& sparse_matrix)
{
    int                num = sparse_matrix.NumElementsWithPadding();
    std::vector<float> out(sparse_matrix.Values(), sparse_matrix.Values() + num);
    return out;
}

namespace {

/**
 * @brief Helper to convert float data to half precision data.
 */
__global__ void ConvertKernel(const float* in_f, half2* out, int n)
{
    const float2* in = reinterpret_cast<const float2*>(in_f);
    n /= 2;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;
    out[idx] = __float22half2_rn(in[idx]);
}

__global__ void ConvertKernel(const int* in_i, short2* out, int n)
{
    const int2* in = reinterpret_cast<const int2*>(in_i);
    n /= 2;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;
    int2   a = in[idx];
    short2 b;
    b.x      = static_cast<short>(a.x);
    b.y      = static_cast<short>(a.y);
    out[idx] = b;
}

__global__ void ConvertKernel(const half2* in, float* out_f, int n)
{
    float2* out = reinterpret_cast<float2*>(out_f);
    n /= 2;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;
    out[idx] = __half22float2(in[idx]);
}

__global__ void ConvertKernel(const short2* in, int* out_i, int n)
{
    int2* out = reinterpret_cast<int2*>(out_i);
    n /= 2;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;
    short2 a = in[idx];
    int2   b;
    b.x      = static_cast<int>(a.x);
    b.y      = static_cast<int>(a.y);
    out[idx] = b;
}

void PadSparseMatrix(const std::vector<int>&   row_offsets,
                     const std::vector<float>& values,
                     const std::vector<int>&   column_indices,
                     int                       row_padding,
                     std::vector<int>*         row_offsets_out,
                     std::vector<float>*       values_out,
                     std::vector<int>*         column_indices_out)
{
    CHECK_GE(row_padding, 0) << "Row padding factor must be greater than zero.";
    if (row_padding < 2) {
        // For row padding to the nearest 1 element, copy the input to the
        // output and return early. We also execute this code path for
        // `row_padding` == 0, which indicates no padding is to be added.
        row_offsets_out->assign(row_offsets.begin(), row_offsets.end());
        values_out->assign(values.begin(), values.end());
        column_indices_out->assign(column_indices.begin(), column_indices.end());
        return;
    }
    row_offsets_out->push_back(0);

    int offset = 0;
    for (int i = 0; i < row_offsets.size() - 1; ++i) {
        // Copy the existing values and column indices for this row to
        // the output.
        int row_length = row_offsets[i + 1] - row_offsets[i];
        values_out->resize(values_out->size() + row_length);
        column_indices_out->resize(column_indices_out->size() + row_length);
        std::copy(values.begin() + row_offsets[i], values.begin() + row_offsets[i + 1], values_out->begin() + offset);
        std::copy(column_indices.begin() + row_offsets[i],
                  column_indices.begin() + row_offsets[i + 1],
                  column_indices_out->begin() + offset);
        offset += row_length;

        // Calculate the number of zeros that need to be inserted in
        // this row to reach the desired padding factor.
        int residue = offset % row_padding;
        int to_add  = (row_padding - residue) % row_padding;
        for (; to_add > 0; --to_add) {
            values_out->push_back(0.0);

            // NOTE: When we pad with zeros the column index that we assign
            // the phantom zero needs to be a valid column index s.t. we
            // don't index out-of-range into the dense rhs matrix when
            // computing spmm. Here we set all padding column-offsets to
            // the same column as the final non-padding weight in the row.
            column_indices_out->push_back(column_indices_out->back());
            ++offset;
        }
        row_offsets_out->push_back(offset);
    }
}

}  // namespace

template<typename In, typename Out>
cudaError_t Convert(const In* in, Out* out, int n)
{
    if (n == 0)
        return cudaSuccess;
    CHECK_EQ(n % 2, 0) << "Number of elements must be multiple of 2.";

    int threads_per_block = 64;
    int blocks_per_grid   = (n + threads_per_block - 1) / threads_per_block;
    ConvertKernel<<<blocks_per_grid, threads_per_block, 0, 0>>>(in, out, n);
    return cudaGetLastError();
}

template<>
cudaError_t Convert(const float* in, float* out, int n)
{
    return cudaMemcpy(out, in, n * sizeof(float), cudaMemcpyDeviceToDevice);
}

template<>
cudaError_t Convert(const int* in, int* out, int n)
{
    return cudaMemcpy(out, in, n * sizeof(int), cudaMemcpyDeviceToDevice);
}

void IdentityRowSwizzle(int rows, const int* /* unused */, int* row_indices)
{
    std::iota(row_indices, row_indices + rows, 0);
}

void SortedRowSwizzle(int rows, const int* row_offsets, int* row_indices)
{
    // Create our unsorted row indices.
    std::vector<int> swizzle_staging(rows);
    std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

    // Argsort the row indices based on their length.
    std::sort(swizzle_staging.begin(), swizzle_staging.end(), [&row_offsets](int idx_a, int idx_b) {
        int length_a = row_offsets[idx_a + 1] - row_offsets[idx_a];
        int length_b = row_offsets[idx_b + 1] - row_offsets[idx_b];
        return length_a > length_b;
    });

    // Copy the ordered row indices to the output.
    std::memcpy(row_indices, swizzle_staging.data(), sizeof(int) * rows);
}

SparseMatrix::SparseMatrix(int rows, int columns, float* originals, Swizzle row_swizzle, int pad_rows_to):
    rows_(rows), columns_(columns), pad_rows_to_(pad_rows_to), row_swizzle_(row_swizzle)
{

    CHECK_LE(pad_rows_to_, columns) << "Rows cannot be padded to more values than there are columns.";

    // calculate the matrix's non zeros
    float zeroThreshold = 1e-7;
    nonzeros_           = 0;
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < columns; ++j) {
            int64_t idx = i * columns + j;
            if (abs(originals[idx]) <= zeroThreshold) {
                nonzeros_++;
            }
        }
    }
    // transfer to CSR format
    std::vector<float> values(nonzeros_);
    std::vector<int>   row_offsets(rows_ + 1);
    std::vector<int>   column_indices(nonzeros_);
    // Create the compressed sparse row indices and offsets.
    int64_t offset = 0;
    row_offsets[0] = 0;
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < columns; ++j) {
            int64_t idx = i * columns + j;
            if (abs(originals[idx]) > zeroThreshold) {
                values[offset]         = originals[idx];
                column_indices[offset] = j;
                ++offset;
            }
        }
        row_offsets[i + 1] = offset;
    }

    // Pad the rows to the desired length.
    std::vector<int>   row_offsets_staging, column_indices_staging;
    std::vector<float> values_staging;
    PadSparseMatrix(row_offsets,
                    values,
                    column_indices,
                    pad_rows_to,
                    &row_offsets_staging,
                    &values_staging,
                    &column_indices_staging);

    // Figure out exactly how much storage we need for the padded matrices,
    // allocate the storage, and copy the matrices into our storage.
    num_elements_with_padding_ = row_offsets_staging[rows_];

    values_         = new float[num_elements_with_padding_];
    column_indices_ = new int[num_elements_with_padding_];
    row_offsets_    = new int[rows_ + 1];

    // Copy the data into our allocated buffers.
    std::memcpy(values_, values_staging.data(), num_elements_with_padding_ * sizeof(float));
    std::memcpy(column_indices_, column_indices_staging.data(), num_elements_with_padding_ * sizeof(int));
    std::memcpy(row_offsets_, row_offsets_staging.data(), (rows_ + 1) * sizeof(int));

    // Allocate storage for our swizzled row indices and set the values.
    row_indices_ = new int[rows_];
    if (row_swizzle_ == IDENTITY) {
        IdentityRowSwizzle(rows_, row_offsets_, row_indices_);
    }
    else {
        SortedRowSwizzle(rows_, row_offsets_, row_indices_);
    }
}

SparseMatrix::SparseMatrix(const CudaSparseMatrix<float>& sparse_matrix)
{
    InitFromCudaSparseMatrix(sparse_matrix);
}

void SparseMatrix::InitFromCudaSparseMatrix(const CudaSparseMatrix<float>& sparse_matrix)
{
    // Copy the sparse matrix meta-data.
    rows_                      = sparse_matrix.Rows();
    columns_                   = sparse_matrix.Columns();
    nonzeros_                  = sparse_matrix.Nonzeros();
    pad_rows_to_               = sparse_matrix.PadRowsTo();
    num_elements_with_padding_ = sparse_matrix.NumElementsWithPadding();
    weight_distribution_       = sparse_matrix.WeightDistribution();
    row_swizzle_               = sparse_matrix.RowSwizzle();

    // Allocate memory on the CPU for our matrix.
    values_         = new float[num_elements_with_padding_];
    column_indices_ = new int[num_elements_with_padding_];
    row_offsets_    = new int[rows_ + 1];
    row_indices_    = new int[rows_];

    // Copy the results to the CPU.
    CUDA_CALL(cudaMemcpy(
        values_, sparse_matrix.Values(), sizeof(float) * num_elements_with_padding_, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(column_indices_,
                         sparse_matrix.ColumnIndices(),
                         sizeof(int) * num_elements_with_padding_,
                         cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(row_offsets_, sparse_matrix.RowOffsets(), sizeof(int) * (rows_ + 1), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(row_indices_, sparse_matrix.RowIndices(), sizeof(int) * rows_, cudaMemcpyDeviceToHost));
}

template<typename Value>
CudaSparseMatrix<Value>::CudaSparseMatrix(const SparseMatrix& sparse_matrix)
{
    // The number of nonzeros in each row must be divisible by the number of
    // elements per scalar for the specified data type.
    for (int i = 0; i < sparse_matrix.Rows(); ++i) {
        int nnz = sparse_matrix.RowOffsets()[i + 1] - sparse_matrix.RowOffsets()[i];
        CHECK_EQ(nnz % sputnik::TypeUtils<Value>::kElementsPerScalar, 0)
            << "The number of elements in each row must be divisible by "
            << "the number of elements per scalar value for the specified "
            << "data type.";
    }
    InitFromSparseMatrix(sparse_matrix);
}

template<typename Value>
void CudaSparseMatrix<Value>::InitFromSparseMatrix(const SparseMatrix& sparse_matrix)
{
    // Copy the sparse matrix meta-data.
    rows_                      = sparse_matrix.Rows();
    columns_                   = sparse_matrix.Columns();
    nonzeros_                  = sparse_matrix.Nonzeros();
    pad_rows_to_               = sparse_matrix.PadRowsTo();
    num_elements_with_padding_ = sparse_matrix.NumElementsWithPadding();
    weight_distribution_       = sparse_matrix.WeightDistribution();
    row_swizzle_               = sparse_matrix.RowSwizzle();

    // Allocate memory on the GPU for our matrix.
    float* values_float       = nullptr;
    int*   column_indices_int = nullptr;
    CUDA_CALL(cudaMalloc(&values_float, sizeof(float) * num_elements_with_padding_));
    CUDA_CALL(cudaMalloc(&column_indices_int, sizeof(int) * num_elements_with_padding_));
    CUDA_CALL(cudaMalloc(&row_offsets_, sizeof(int) * (rows_ + 1)));
    CUDA_CALL(cudaMalloc(&row_indices_, sizeof(int) * rows_));

    // Copy the results to the GPU.
    CUDA_CALL(cudaMemcpy(
        values_float, sparse_matrix.Values(), sizeof(float) * num_elements_with_padding_, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(column_indices_int,
                         sparse_matrix.ColumnIndices(),
                         sizeof(int) * num_elements_with_padding_,
                         cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(row_offsets_, sparse_matrix.RowOffsets(), sizeof(int) * (rows_ + 1), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(row_indices_, sparse_matrix.RowIndices(), sizeof(int) * rows_, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaStreamSynchronize(nullptr));

    // Allocate memory for the values and indices in the target datatype.
    int elements = num_elements_with_padding_ / sputnik::TypeUtils<Value>::kElementsPerScalar;
    CUDA_CALL(cudaMalloc(&values_, sizeof(Value) * elements));
    CUDA_CALL(cudaMalloc(&column_indices_, sizeof(Index) * elements));

    // Convert to the target datatype.
    CUDA_CALL(Convert(values_float, values_, num_elements_with_padding_));
    CUDA_CALL(Convert(column_indices_int, column_indices_, num_elements_with_padding_));
    CUDA_CALL(cudaStreamSynchronize(nullptr));

    // Free the temporary memory.
    CUDA_CALL(cudaFree(values_float));
    CUDA_CALL(cudaFree(column_indices_int));
}

// Explicit instantiations for template functions and classes.
template class CudaSparseMatrix<float>;
template class CudaSparseMatrix<half2>;

}  // namespace sputnik_utils

#endif  // THIRD_PARTY_SPUTNIK_MATRIX_UTILS_H_
