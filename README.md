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
| Construction (vec3)      | 1.77 ns      | 0.69 ns      | 0.69 ns     | 61% slower    | 61% slower    |
| Addition                 | 1.18 ns      | 1.18 ns      | 1.78 ns     | ~Equal        | 34% faster    |
| Dot Product              | 2.19 ns      | 1.28 ns      | 1.18 ns     | 42% slower    | 46% slower    |
| Cross Product            | 2.84 ns      | 2.18 ns      | 2.21 ns     | 23% slower    | 22% slower    |
| Normalization            | 28.62 ns     | 3.61 ns      | 6.96 ns     | 87% slower    | 76% slower    |
| **Matrix Operations**    |              |              |             |               |               |
| Construction (4x4)       | 2.98 ns      | 2.97 ns      | 6.48 ns     | ~Equal        | 54% faster    |
| **Multiplication (4x4)** | **32.64 ns** | **32.68 ns** | **9.58 ns** | ~Equal        | 71% slower    |
| Transpose (4x4)          | 8.01 ns      | 7.45 ns      | 7.25 ns     | ~Equal        | ~Equal        |
| Determinant (4x4)        | 11.77 ns     | 10.99 ns     | 10.74 ns    | ~Equal        | ~Equal        |
| **Transformations**      |              |              |             |               |               |
| Translation Matrix       | 20.71 ns     | 4.81 ns      | 16.84 ns    | 77% slower    | 19% slower    |
| Rotation Matrix (X)      | 101.48 ns    | 61.20 ns     | 32.82 ns    | 40% slower    | 68% slower    |
| Perspective Projection   | 84.45 ns     | 25.22 ns     | 21.29 ns    | 70% slower    | 75% slower    |

*MSVC 14.50 (VS 2026), Release, 1M iterations. Absolute ns vary by hardware/thermal state — the ratios are what matter. See [`benchmarks/BENCHMARK_README.md`](benchmarks/BENCHMARK_README.md) for details.*

> **Bench methodology.** Eigen comparisons force materialization (lambda return type is `Matrix4f`/`Vector3f`) to defeat lazy expression-template evaluation. Matmul/transpose/determinant pull inputs from a 64-element pool indexed by the iteration counter so the optimizer cannot hoist a loop-invariant computation out of the timed region. Both fixes were necessary — without them, Eigen's matmul appeared ~50× faster than it really is.
