# Lina - Modern C++ Linear Algebra Library

[![Tests](https://github.com/illia-bilinskyi/Lina/workflows/CMake%20on%20multiple%20platforms/badge.svg)](https://github.com/illia-bilinskyi/Lina/actions)
[![C++11/14/17](https://img.shields.io/badge/C%2B%2B-11%2F14%2F17-blue.svg)](https://isocpp.org/std/the-standard)
[![Header-Only](https://img.shields.io/badge/Header--Only-Yes-brightgreen.svg)]()
[![CUDA Ready](https://img.shields.io/badge/CUDA-Ready-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Lina is a **lightweight, efficient, and GPU-ready** C++ linear algebra library designed specifically for 3D geometry and graphics applications. Built with modern C++ principles, it provides compile-time optimizations, comprehensive CUDA support, and mathematical precision in a single header file.

## 🌟 Key Features

### **Header-Only Design**
- **Single file deployment** - Just include `lina/lina.h`
- **Zero dependencies** - No external libraries required
- **Easy integration** - Drop into any project instantly

### **Modern C++ Excellence**
- **Multi-standard support** - Compatible with C++11, C++14, and C++17
- **Extensive constexpr support** - Most operations computed at compile-time
- **Template metaprogramming** - Type-safe generic programming
- **SFINAE and type traits** - Robust template constraints
- **Exception safety** - Clear error handling with meaningful messages

### **GPU Acceleration Ready**
- **Comprehensive CUDA support** - `CUDA_MODIFIER` annotations throughout
- **Host/device compatibility** - Seamless execution on CPU and GPU
- **Device-safe design** - Proper separation of host-only features
- **Memory efficient** - Stack-based allocation, no dynamic memory

### **Mathematical Precision**
- **Advanced constexpr math** - Custom Taylor series trigonometry, Newton-Raphson sqrt
- **Numerical stability** - Epsilon-based comparisons, range normalization
- **Compile-time evaluation** - Complex calculations at compile-time
- **IEEE 754 compliance** - Proper floating-point handling

## 🚀 Performance Features

- **Zero overhead** - Template-based design with no virtual functions
- **Compile-time optimization** - Most operations resolved at compile-time
- **Cache-friendly** - Contiguous memory layout for optimal performance
- **SIMD ready** - Structure supports vectorization optimizations
- **Stack allocation** - No heap usage for predictable performance

## 🛠 Technical Specifications

**Tested Configurations:**
- **C++ Standards**: Full support with all features on **C++11/14/17**
- **Platforms**: Linux (Ubuntu), Windows (MSVC), macOS (Clang)
- **Compilers**: GCC, Clang, MSVC
- **Build Types**: Debug and Release configurations

### **Supported Types**
```cpp
// Matrix types
mat2<T>, mat3<T>, mat4<T>  // Generic square matrices
mat2f, mat3f, mat4f        // Float specializations
mat2d, mat3d, mat4d        // Double specializations

// Vector types  
vec3<T>                    // Generic 3D vectors
vec3f, vec3d              // Float/double specializations
```

### **Core Operations**
- **Matrix operations**: Identity, transpose, determinant, inverse (2x2, 3x3, 4x4)
- **Vector operations**: Dot product, cross product, normalization, distance
- **Transformations**: Translation, rotation (axis-angle, Euler, matrix), scaling
- **Camera matrices**: Look-at, perspective, orthographic projection matrices

### **Constexpr Mathematical Functions**
- **Trigonometry**: `sin()`, `cos()`, `tan()` with Taylor series (16-term precision)
- **Square root**: `sqrt()` using Newton-Raphson method (machine precision)
- **Absolute value**: `abs()` for all arithmetic types
- **Comparisons**: Epsilon-based floating-point comparisons

## 📋 Quick Start

### **Basic Usage**
```cpp
#include "lina/lina.h"
using namespace lina;

// Compile-time vector operations
constexpr vec3f v1{1.0f, 2.0f, 3.0f};
constexpr vec3f v2{4.0f, 5.0f, 6.0f};
constexpr auto dot_product = dot(v1, v2);        // 32.0f
constexpr auto cross_product = cross(v1, v2);   // {-3, 6, -3}

// Matrix transformations
constexpr auto translation_mat = translation(vec3f{1, 2, 3});
constexpr auto rotation_mat = rotation_x(pi<float> / 4);
constexpr auto transform = translation_mat * rotation_mat;

// Advanced compile-time calculations
constexpr auto angle = pi<double> / 6.0;  // 30 degrees
constexpr auto sine_val = sin(angle);     // 0.5, computed at compile-time
```

### **CUDA Usage**
```cpp
// Device kernel using Lina operations
__global__ void transform_vertices(vec3f* vertices, mat4f transform, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        vertices[idx] = transform * vertices[idx];  // Works on device!
    }
}

// All constexpr operations work in device code
__device__ constexpr vec3f compute_normal(vec3f a, vec3f b, vec3f c) {
    return normalize(cross(b - a, c - a));
}
```

### **Graphics Pipeline Example**
```cpp
// Complete 3D transformation pipeline
constexpr mat4f model = translation({0, 0, -5}) * 
                       rotation_y(pi<float> / 4) * 
                       scale({2, 2, 2});

constexpr mat4f view = look_at({0, 0, 0}, {0, 0, -1}, {0, 1, 0});
constexpr mat4f proj = perspective(pi<float> / 4, 16.0f/9.0f, 0.1f, 100.0f);
constexpr mat4f mvp = proj * view * model;  // All computed at compile-time!
```

## 🏗 Build System Integration

### **CMake Integration**
```cmake
# Add Lina to your project
add_subdirectory(lina)
target_link_libraries(your_target lina)

# Or simply include the header
target_include_directories(your_target PRIVATE path/to/lina)
```

## 🧪 Testing

### **Comprehensive Test Coverage**
- **Functional Tests**: Complete Google Test suite with C++17
- **Build Verification**: Compilation testing across C++11/14/17
- **Cross-Platform**: Automated testing on Linux, Windows, and macOS
- **Multi-Compiler**: GCC, Clang, and MSVC validation

```bash
# Build and run tests
cmake --preset default
cmake --build build/default
cd build/default && ctest

# Build verification test (tests compilation across C++ standards)
cmake --build build/default --target test_build_lina
```

## 🔧 Configuration Options

### **Storage Layout**
```cpp
// Row-major (default)
#include "lina/lina.h"

// Column-major 
#define LINA_MAT_COLUMN_MAJOR
#include "lina/lina.h"
```

### **CUDA Compatibility**
```cpp
// Automatic CUDA detection
#ifdef __CUDA_ARCH__
    // Device code - optimized for GPU
#else  
    // Host code - full standard library access
#endif
```

### **Precision Control**
```cpp
// Default epsilon for floating-point comparisons
template<typename T> 
inline constexpr T COMPARE_EPSILON_DEFAULT = static_cast<T>(1e-6);

// Usage
bool equal = almost_equal(a, b, 1e-10);  // Custom precision
```

## 🎯 Use Cases

- **Graphics engines** - Transform matrices, camera systems
- **Game development** - 3D math, physics calculations
- **GPU computing** - CUDA kernels, parallel algorithms

## 📚 API Reference

### **Matrix Class (`mat<T, R, C>`)**
```cpp
// Construction
mat<float, 3, 3> m1;                    // Zero matrix
mat<float, 3, 3> m2{1, 2, 3, 4, 5, 6, 7, 8, 9};  // Direct initialization
auto m3 = identity<float, 4>();         // Identity matrix

// Access
m(i, j)              // Functional access
m[index]             // Linear access  
m.get<r, c>()        // Compile-time access
m.row(i), m.col(j)   // Row/column extraction

// Operations
auto result = m1 + m2;     // Addition
auto product = m1 * m2;    // Multiplication
auto transposed = transpose(m);
auto determinant = det(m);
auto inverted = inverse(m);
```

### **Vector Class (`vec3<T>`)**
```cpp
// Construction
vec3f v1{1, 2, 3};
vec3f v2{x, y, z};

// Operations
auto length = norm(v);
auto unit = normalize(v);
auto dot_prod = dot(v1, v2);
auto cross_prod = cross(v1, v2);
auto dist = distance(v1, v2);

// Access
v.x, v.y, v.z        // Named access
v[i]                 // Indexed access
v.get<i>()           // Compile-time access
```