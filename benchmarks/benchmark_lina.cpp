#include "../lina/lina.h"
#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Benchmark infrastructure
class BenchmarkTimer {
public:
    using clock = std::chrono::high_resolution_clock;
    using duration = std::chrono::nanoseconds;
    
    static constexpr size_t DEFAULT_ITERATIONS = 1000000;
    static constexpr size_t WARMUP_ITERATIONS = 10000;
    
    template<typename Func>
    static double benchmark(const std::string& name, Func&& func, size_t iterations = DEFAULT_ITERATIONS) {
        // Warmup
        for (size_t i = 0; i < WARMUP_ITERATIONS; ++i) {
            func();
        }
        
        // Actual benchmark
        auto start = clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            func();
        }
        auto end = clock::now();
        
        auto total_ns = std::chrono::duration_cast<duration>(end - start).count();
        double avg_ns = static_cast<double>(total_ns) / iterations;
        
        std::cout << std::left << std::setw(35) << name 
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) 
                  << avg_ns << " ns" << std::endl;
        
        return avg_ns;
    }
    
    template<typename Func>
    static double benchmark_with_result(const std::string& name, Func&& func, size_t iterations = DEFAULT_ITERATIONS) {
        auto sink = func(); // Result sink to prevent optimization
        
        // Warmup
        for (size_t i = 0; i < WARMUP_ITERATIONS; ++i) {
            auto temp = func();
            // Use the result to prevent complete optimization
            if (std::is_arithmetic_v<decltype(temp)>) {
                volatile auto vol_sink = temp;
                (void)vol_sink;
            } else {
                // For complex types, just ensure they're not optimized away
                volatile auto* ptr = &temp;
                (void)ptr;
            }
        }
        
        // Actual benchmark
        auto start = clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            auto temp = func();
            // Use the result to prevent complete optimization
            if (std::is_arithmetic_v<decltype(temp)>) {
                volatile auto vol_sink = temp;
                (void)vol_sink;
            } else {
                // For complex types, just ensure they're not optimized away
                volatile auto* ptr = &temp;
                (void)ptr;
            }
        }
        auto end = clock::now();
        
        // Keep sink alive
        volatile auto* sink_ptr = &sink;
        (void)sink_ptr;
        
        auto total_ns = std::chrono::duration_cast<duration>(end - start).count();
        double avg_ns = static_cast<double>(total_ns) / iterations;
        
        std::cout << std::left << std::setw(35) << name 
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) 
                  << avg_ns << " ns" << std::endl;
        
        return avg_ns;
    }
};

// Random data generators
template<typename T>
class RandomGenerator {
    mutable std::mt19937 gen{std::random_device{}()};
    mutable std::uniform_real_distribution<T> dist{-10.0, 10.0};
    
public:
    T random() const { return dist(gen); }
    
    lina::vec3<T> random_vec3() const {
        return {random(), random(), random()};
    }
    
    lina::mat<T, 2, 2> random_mat2() const {
        return {random(), random(), 
                random(), random()};
    }
    
    lina::mat<T, 3, 3> random_mat3() const {
        return {random(), random(), random(),
                random(), random(), random(),
                random(), random(), random()};
    }
    
    lina::mat<T, 4, 4> random_mat4() const {
        return {random(), random(), random(), random(),
                random(), random(), random(), random(),
                random(), random(), random(), random(),
                random(), random(), random(), random()};
    }
    
    // Generate non-singular matrix for inverse testing
    lina::mat<T, 3, 3> random_invertible_mat3() const {
        lina::mat<T, 3, 3> m;
        do {
            m = random_mat3();
        } while (lina::almost_zero(lina::det(m)));
        return m;
    }
    
    lina::mat<T, 4, 4> random_invertible_mat4() const {
        lina::mat<T, 4, 4> m;
        do {
            m = random_mat4();
        } while (lina::almost_zero(lina::det(m)));
        return m;
    }
};

// Vector operation benchmarks
template<typename T>
void benchmark_vector_operations() {
    RandomGenerator<T> rng;
    const auto v1 = rng.random_vec3();
    const auto v2 = rng.random_vec3();
    
    std::cout << "\n=== Vector Operations (" << (sizeof(T) == 4 ? "float" : "double") << ") ===" << std::endl;
    
    // Vector norm (length)
    BenchmarkTimer::benchmark_with_result("Vector norm", [&]() {
        return lina::norm(v1);
    });
    
    // Vector norm squared (more efficient)
    BenchmarkTimer::benchmark_with_result("Vector norm squared", [&]() {
        return lina::norm2(v1);
    });
    
    // Vector normalization
    BenchmarkTimer::benchmark_with_result("Vector normalize", [&]() {
        return lina::normalize(v1);
    });
    
    // Dot product
    BenchmarkTimer::benchmark_with_result("Vector dot product", [&]() {
        return lina::dot(v1, v2);
    });
    
    // Cross product
    BenchmarkTimer::benchmark_with_result("Vector cross product", [&]() {
        return lina::cross(v1, v2);
    });
    
    // Vector distance
    BenchmarkTimer::benchmark_with_result("Vector distance", [&]() {
        return lina::distance(v1, v2);
    });
    
    // Vector-scalar operations
    BenchmarkTimer::benchmark_with_result("Vector * scalar", [&]() {
        return v1 * static_cast<T>(2.5);
    });
    
    BenchmarkTimer::benchmark_with_result("Vector + Vector", [&]() {
        return v1 + v2;
    });
}

// Matrix operation benchmarks
template<typename T>
void benchmark_matrix_operations() {
    RandomGenerator<T> rng;
    
    std::cout << "\n=== Matrix Operations (" << (sizeof(T) == 4 ? "float" : "double") << ") ===" << std::endl;
    
    // 2x2 Matrix operations
    {
        const auto m1 = rng.random_mat2();
        const auto m2 = rng.random_mat2();
        
        BenchmarkTimer::benchmark_with_result("Matrix 2x2 multiply", [&]() {
            return m1 * m2;
        });
        
        BenchmarkTimer::benchmark_with_result("Matrix 2x2 transpose", [&]() {
            return lina::transpose(m1);
        });
        
        BenchmarkTimer::benchmark_with_result("Matrix 2x2 determinant", [&]() {
            return lina::det(m1);
        });
        
        const auto inv_m = rng.random_invertible_mat3(); // Use 3x3 for invertible
        BenchmarkTimer::benchmark_with_result("Matrix 2x2 inverse (constexpr)", [&]() {
            return lina::c_inverse(m1);
        });
    }
    
    // 3x3 Matrix operations
    {
        const auto m1 = rng.random_mat3();
        const auto m2 = rng.random_mat3();
        const auto v = rng.random_vec3();
        
        BenchmarkTimer::benchmark_with_result("Matrix 3x3 multiply", [&]() {
            return m1 * m2;
        });
        
        BenchmarkTimer::benchmark_with_result("Matrix 3x3 * Vector", [&]() {
            return m1 * v;
        });
        
        BenchmarkTimer::benchmark_with_result("Matrix 3x3 transpose", [&]() {
            return lina::transpose(m1);
        });
        
        BenchmarkTimer::benchmark_with_result("Matrix 3x3 determinant", [&]() {
            return lina::det(m1);
        });
        
        const auto inv_m = rng.random_invertible_mat3();
        BenchmarkTimer::benchmark_with_result("Matrix 3x3 inverse (constexpr)", [&]() {
            return lina::c_inverse(inv_m);
        });
        
        BenchmarkTimer::benchmark_with_result("Matrix 3x3 inverse (runtime)", [&]() {
            return lina::inverse(inv_m);
        });
    }
    
    // 4x4 Matrix operations
    {
        const auto m1 = rng.random_mat4();
        const auto m2 = rng.random_mat4();
        const auto v = rng.random_vec3();
        
        BenchmarkTimer::benchmark_with_result("Matrix 4x4 multiply", [&]() {
            return m1 * m2;
        });
        
        BenchmarkTimer::benchmark_with_result("Matrix 4x4 * Vector (homogeneous)", [&]() {
            return m1 * v;
        });
        
        BenchmarkTimer::benchmark_with_result("Matrix 4x4 transpose", [&]() {
            return lina::transpose(m1);
        });
        
        BenchmarkTimer::benchmark_with_result("Matrix 4x4 determinant", [&]() {
            return lina::det(m1);
        });
        
        const auto inv_m = rng.random_invertible_mat4();
        BenchmarkTimer::benchmark_with_result("Matrix 4x4 inverse (constexpr)", [&]() {
            return lina::c_inverse(inv_m);
        });
        
        BenchmarkTimer::benchmark_with_result("Matrix 4x4 inverse (runtime)", [&]() {
            return lina::inverse(inv_m);
        });
    }
}

// Constexpr math function benchmarks
template<typename T>
void benchmark_constexpr_math() {
    std::cout << "\n=== Constexpr Math Functions (" << (sizeof(T) == 4 ? "float" : "double") << ") ===" << std::endl;
    
    const T angle = lina::pi<T> / 6; // 30 degrees
    const T value = static_cast<T>(2.5);
    
    BenchmarkTimer::benchmark_with_result("sin(π/6)", [&]() {
        return lina::sin(angle);
    });
    
    BenchmarkTimer::benchmark_with_result("cos(π/6)", [&]() {
        return lina::cos(angle);
    });
    
    BenchmarkTimer::benchmark_with_result("tan(π/6)", [&]() {
        return lina::tan(angle);
    });
    
    BenchmarkTimer::benchmark_with_result("sqrt(2.5)", [&]() {
        return lina::sqrt(value);
    });
    
    BenchmarkTimer::benchmark_with_result("abs(-2.5)", [&]() {
        return lina::abs(-value);
    });
}

// 3D transformation benchmarks
template<typename T>
void benchmark_transformations() {
    RandomGenerator<T> rng;
    const auto vec = rng.random_vec3();
    const T angle = lina::pi<T> / 4;
    
    std::cout << "\n=== 3D Transformations (" << (sizeof(T) == 4 ? "float" : "double") << ") ===" << std::endl;
    
    BenchmarkTimer::benchmark_with_result("Identity matrix 4x4", [&]() {
        return lina::identity<T, 4>();
    });
    
    BenchmarkTimer::benchmark_with_result("Translation matrix", [&]() {
        return lina::translation(vec);
    });
    
    BenchmarkTimer::benchmark_with_result("Rotation X matrix", [&]() {
        return lina::rotation_x(angle);
    });
    
    BenchmarkTimer::benchmark_with_result("Rotation Y matrix", [&]() {
        return lina::rotation_y(angle);
    });
    
    BenchmarkTimer::benchmark_with_result("Rotation Z matrix", [&]() {
        return lina::rotation_z(angle);
    });
    
    BenchmarkTimer::benchmark_with_result("Rotation arbitrary axis", [&]() {
        return lina::rotation(lina::normalize(vec), angle);
    });
    
    BenchmarkTimer::benchmark_with_result("Scale matrix", [&]() {
        return lina::scale(vec);
    });
    
    const auto eye = rng.random_vec3();
    const auto center = rng.random_vec3();
    const auto up = lina::vec3<T>{0, 1, 0};
    
    BenchmarkTimer::benchmark_with_result("Look-at matrix", [&]() {
        return lina::look_at(eye, center, up);
    });
    
    BenchmarkTimer::benchmark_with_result("Perspective projection", [&]() {
        return lina::perspective(lina::pi<T>/4, static_cast<T>(16.0/9.0), 
                                static_cast<T>(0.1), static_cast<T>(100.0));
    });
    
    BenchmarkTimer::benchmark_with_result("Orthographic projection", [&]() {
        return lina::ortho(static_cast<T>(-10), static_cast<T>(10), 
                          static_cast<T>(-10), static_cast<T>(10),
                          static_cast<T>(0.1), static_cast<T>(100));
    });
}

// Comparison benchmarks
template<typename T>
void benchmark_comparisons() {
    RandomGenerator<T> rng;
    const auto v1 = rng.random_vec3();
    const auto v2 = v1 + lina::vec3<T>{static_cast<T>(1e-7), 0, 0}; // Nearly equal
    const auto m1 = rng.random_mat3();
    const auto m2 = m1;
    
    std::cout << "\n=== Comparison Operations (" << (sizeof(T) == 4 ? "float" : "double") << ") ===" << std::endl;
    
    BenchmarkTimer::benchmark_with_result("almost_equal (scalars)", [&]() {
        return lina::almost_equal(v1.x, v2.x);
    });
    
    BenchmarkTimer::benchmark_with_result("almost_equal (vectors)", [&]() {
        return lina::almost_equal(v1, v2);
    });
    
    BenchmarkTimer::benchmark_with_result("almost_equal (matrices)", [&]() {
        return lina::almost_equal(m1, m2);
    });
    
    BenchmarkTimer::benchmark_with_result("almost_zero (scalar)", [&]() {
        return lina::almost_zero(static_cast<T>(1e-8));
    });
}

// Compile-time evaluation showcase
void benchmark_compile_time() {
    std::cout << "\n=== Compile-time Evaluation Showcase ===" << std::endl;
    
    // These are computed at compile-time but we benchmark the "runtime" access
    constexpr auto ct_identity = lina::identity<float, 4>();
    constexpr auto ct_translation = lina::translation(lina::vec3f{1, 2, 3});
    constexpr auto ct_rotation = lina::rotation_x(lina::pi<float> / 4);
    constexpr auto ct_mvp = ct_translation * ct_rotation;
    constexpr auto ct_sin = lina::sin(lina::pi<float> / 6);
    constexpr auto ct_sqrt = lina::sqrt(2.0f);
    constexpr auto ct_normalized = lina::normalize(lina::vec3f{3, 4, 0});
    
    // Benchmark accessing compile-time computed values (should be near-zero cost)
    BenchmarkTimer::benchmark_with_result("Access compile-time matrix", [&]() {
        return ct_mvp(0, 0);
    });
    
    BenchmarkTimer::benchmark_with_result("Access compile-time sin(π/6)", [&]() {
        return ct_sin;
    });
    
    BenchmarkTimer::benchmark_with_result("Access compile-time sqrt(2)", [&]() {
        return ct_sqrt;
    });
    
    BenchmarkTimer::benchmark_with_result("Access compile-time normalized", [&]() {
        return ct_normalized.x;
    });
    
    std::cout << "\nCompile-time computed values:" << std::endl;
    std::cout << "  sin(π/6) = " << ct_sin << " (expected: 0.5)" << std::endl;
    std::cout << "  sqrt(2) = " << ct_sqrt << " (expected: ~1.414)" << std::endl;
    std::cout << "  normalize({3,4,0}) = {" << ct_normalized.x << ", " 
              << ct_normalized.y << ", " << ct_normalized.z << "}" << std::endl;
}

// Memory layout benchmark
template<typename T>
void benchmark_memory_access() {
    std::cout << "\n=== Memory Access Patterns (" << (sizeof(T) == 4 ? "float" : "double") << ") ===" << std::endl;
    
    // Create arrays of matrices and vectors
    constexpr size_t ARRAY_SIZE = 10000;
    std::vector<lina::mat<T, 4, 4>> matrices(ARRAY_SIZE);
    std::vector<lina::vec3<T>> vectors(ARRAY_SIZE);
    
    RandomGenerator<T> rng;
    for (auto& m : matrices) m = rng.random_mat4();
    for (auto& v : vectors) v = rng.random_vec3();
    
    // Sequential access
    BenchmarkTimer::benchmark_with_result("Sequential matrix access", [&]() {
        T sum = 0;
        for (const auto& m : matrices) {
            sum += m(0, 0);
        }
        return sum;
    }, 1000); // Fewer iterations for array operations
    
    BenchmarkTimer::benchmark_with_result("Sequential vector access", [&]() {
        T sum = 0;
        for (const auto& v : vectors) {
            sum += v.x;
        }
        return sum;
    }, 1000);
    
    // Batch vector operations
    BenchmarkTimer::benchmark_with_result("Batch vector normalize", [&]() {
        T sum = 0;
        for (const auto& v : vectors) {
            sum += lina::norm(v);
        }
        return sum;
    }, 100); // Even fewer iterations for complex operations
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "         LINA BENCHMARK SUITE          " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Operation Name                     Avg Time" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // Float benchmarks
    benchmark_vector_operations<float>();
    benchmark_matrix_operations<float>();
    benchmark_constexpr_math<float>();
    benchmark_transformations<float>();
    benchmark_comparisons<float>();
    benchmark_memory_access<float>();
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    
    // Double benchmarks
    benchmark_vector_operations<double>();
    benchmark_matrix_operations<double>();
    benchmark_constexpr_math<double>();
    benchmark_transformations<double>();
    benchmark_comparisons<double>();
    benchmark_memory_access<double>();
    
    // Compile-time showcase
    benchmark_compile_time();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "         BENCHMARK COMPLETE            " << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}