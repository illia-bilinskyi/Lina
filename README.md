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

| Operation                | Lina         | GLM          | Eigen       | Lina vs GLM       | Lina vs Eigen     |
|--------------------------|--------------|--------------|-------------|-------------------|-------------------|
| **Vector Operations**    |              |              |             |                   |                   |
| Construction (vec3)      | 0.28 ns      | 0.38 ns      | 0.27 ns     | 36% faster        | ~Equal            |
| Addition                 | 0.24 ns      | 0.22 ns      | 4.85 ns     | ~Equal            | 20x faster        |
| Dot Product              | 0.23 ns      | 0.22 ns      | 0.22 ns     | ~Equal            | ~Equal            |
| Cross Product            | 0.97 ns      | 0.26 ns      | 0.23 ns     | 73% slower        | 76% slower        |
| Normalization            | 10.57 ns     | 1.45 ns      | 2.66 ns     | 86% slower        | 75% slower        |
| **Matrix Operations**    |              |              |             |                   |                   |
| Construction (4x4)       | 2.40 ns      | 2.29 ns      | 6.47 ns     | ~Equal            | 63% faster        |
| **Multiplication (4x4)** | **11.81 ns** | **13.71 ns** | **0.22 ns** | 14% faster        | 98% slower        |
| Transpose (4x4)          | 2.21 ns      | 2.25 ns      | 0.23 ns     | ~Equal            | 90% slower        |
| Determinant (4x4)        | 3.94 ns      | 3.73 ns      | 4.95 ns     | ~Equal            | 20% faster        |
| **Transformations**      |              |              |             |                   |                   |
| Translation Matrix       | 6.24 ns      | 1.73 ns      | 7.91 ns     | 72% slower        | 21% faster        |
| Rotation Matrix          | 29.01 ns     | 24.97 ns     | 12.39 ns    | 14% slower        | 57% slower        |
| Perspective Projection   | 31.57 ns     | 8.04 ns      | 6.56 ns     | 75% slower        | 79% slower        |

*Run with MSVC, 1M iterations. Results vary by platform. See [`benchmarks/BENCHMARK_README.md`](benchmarks/BENCHMARK_README.md) for details.*
