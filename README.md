# Lina — C++ Linear Algebra for 3D Graphics

[![Tests](https://github.com/illia-bilinskyi/Lina/workflows/CMake%20on%20multiple%20platforms/badge.svg)](https://github.com/illia-bilinskyi/Lina/actions)
[![C++14/17/20](https://img.shields.io/badge/C%2B%2B-14%2F17%2F20-blue.svg)](https://isocpp.org/std/the-standard)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A header-only, `constexpr`-friendly linear algebra library for 3D geometry and graphics. Templated matrices and vectors with compile-time size checking, CUDA-ready.

## Motivation

Got tired of GLM's conventions and the variable-inspection nightmare during debugging — opaque types, layouts that fight the debugger, and surprising defaults. Lina is an attempt at a better library: straightforward templated types, `constexpr`-powered operations, and matrices that read clearly in the debugger (with custom LLDB formatters included).

## Quick Start

```cpp
#include "lina/lina.h"
using namespace lina;

constexpr vec3f v1{1, 2, 3};
constexpr vec3f v2{4, 5, 6};
constexpr auto d = dot(v1, v2);     // 32, at compile time
constexpr auto c = cross(v1, v2);   // {-3, 6, -3}

constexpr mat4f model = translation({0, 0, -5}) * rotation_y(pi<float> / 4);
constexpr mat4f view  = look_at({0, 0, 0}, {0, 0, -1}, {0, 1, 0});
constexpr mat4f proj  = perspective(pi<float> / 4, 16.0f / 9.0f, 0.1f, 100.0f);
constexpr mat4f mvp   = proj * view * model;   // entirely compile-time
```

## Types

```cpp
mat2<T>, mat3<T>, mat4<T>      // generic square matrices
mat2f, mat3f, mat4f            // float
mat2d, mat3d, mat4d            // double
vec3<T>, vec3f, vec3d          // 3D vectors
```

## Build

```bash
cmake --preset default
cmake --build build/default
cd build/default && ctest
```

Row-major by default; define `LINA_MAT_COLUMN_MAJOR` before including for column-major storage.

## Debug Visualization

Custom LLDB formatters render matrices row-by-row in the debugger:

```bash
(lldb) command script import path/to/lina/lldb_matrix_formatters.py
```

## Benchmarks

Lina vs GLM vs Eigen — Release mode, lower is better:

| Operation                | Lina         | GLM          | Eigen       | Lina vs GLM   | Lina vs Eigen |
|--------------------------|--------------|--------------|-------------|---------------|---------------|
| **Vector Operations**    |              |              |             |               |               |
| Construction (vec3)      | 1.14 ns      | 0.73 ns      | 0.29 ns     | 36% slower    | 75% slower    |
| Addition                 | 1.08 ns      | 1.01 ns      | 0.82 ns     | ~Equal        | 24% slower    |
| Dot Product              | 0.86 ns      | 0.60 ns      | 0.69 ns     | 30% slower    | 20% slower    |
| Cross Product            | 1.44 ns      | 2.32 ns      | 1.69 ns     | 38% faster    | 15% faster    |
| Normalization            | 12.70 ns     | 3.69 ns      | 3.90 ns     | 71% slower    | 69% slower    |
| **Matrix Operations**    |              |              |             |               |               |
| Construction (4x4)       | 1.29 ns      | 1.53 ns      | 3.07 ns     | 16% faster    | 58% faster    |
| **Multiplication (4x4)** | **15.11 ns** | **14.71 ns** | **4.71 ns** | ~Equal        | 69% slower    |
| Transpose (4x4)          | 2.58 ns      | 2.48 ns      | 2.95 ns     | ~Equal        | 13% faster    |
| Determinant (4x4)        | 4.53 ns      | 4.83 ns      | 4.23 ns     | ~Equal        | ~Equal        |
| **Transformations**      |              |              |             |               |               |
| Translation Matrix       | 8.70 ns      | 2.02 ns      | 6.46 ns     | 77% slower    | 26% slower    |
| Rotation Matrix (X)      | 41.80 ns     | 39.80 ns     | 16.20 ns    | ~Equal        | 61% slower    |
| Perspective Projection   | 41.44 ns     | 9.44 ns      | 9.51 ns     | 77% slower    | 77% slower    |

*MSVC 14.50 (VS 2026), Release, 1M iterations. Results vary by platform. See [`benchmarks/BENCHMARK_README.md`](benchmarks/BENCHMARK_README.md) for details.*
