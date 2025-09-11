#include "lina/lina.h"

#include <Eigen/Geometry>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Benchmark infrastructure
class ComparisonTimer
{
  public:
    using clock    = std::chrono::high_resolution_clock;
    using duration = std::chrono::nanoseconds;

    static constexpr size_t DEFAULT_ITERATIONS = 1000000;
    static constexpr size_t WARMUP_ITERATIONS  = 10000;

    template <typename Func>
    static double benchmark(Func&& func, size_t iterations = DEFAULT_ITERATIONS)
    {
        // Warmup
        for (size_t i = 0; i < WARMUP_ITERATIONS; ++i)
        {
            func();
        }

        // Actual benchmark
        auto start = clock::now();
        for (size_t i = 0; i < iterations; ++i)
        {
            func();
        }
        auto end = clock::now();

        auto total_ns = std::chrono::duration_cast<duration>(end - start).count();
        return static_cast<double>(total_ns) / iterations;
    }

    template <typename Func>
    static double benchmark_with_result(Func&& func, size_t iterations = DEFAULT_ITERATIONS)
    {
        volatile auto sink = func(); // Result sink to prevent optimization

        // Warmup
        for (size_t i = 0; i < WARMUP_ITERATIONS; ++i)
        {
            volatile auto temp = func();
            (void)temp;
        }

        // Actual benchmark
        auto start = clock::now();
        for (size_t i = 0; i < iterations; ++i)
        {
            volatile auto result = func();
            (void)result;
        }
        auto end = clock::now();

        auto total_ns = std::chrono::duration_cast<duration>(end - start).count();
        return static_cast<double>(total_ns) / iterations;
    }
};

// Test data generators
class TestData
{
  public:
    static std::vector<float> random_floats(size_t count, float min = -10.0f, float max = 10.0f)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min, max);

        std::vector<float> data;
        data.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            data.push_back(dis(gen));
        }
        return data;
    }
};

void print_header(const std::string& test_name)
{
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << " " << test_name << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::left << std::setw(20) << "Library" << std::right << std::setw(12) << "Time (ns)" << std::setw(15)
              << "Relative" << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
}

void print_result(const std::string& name, double time_ns, double baseline_ns)
{
    double relative = time_ns / baseline_ns;
    double speedup  = baseline_ns / time_ns;

    std::cout << std::left << std::setw(20) << name << std::right << std::setw(12) << std::fixed << std::setprecision(2)
              << time_ns << std::setw(15) << std::setprecision(2) << relative << "x" << std::setw(15)
              << std::setprecision(2) << speedup << "x" << std::endl;
}

void benchmark_vector_operations()
{
    // Test data
    auto data = TestData::random_floats(12); // 4 vectors × 3 components

    // Vector construction
    print_header("Vector Construction (vec3)");

    double lina_construct = ComparisonTimer::benchmark_with_result(
        [&]() -> lina::vec3f { return lina::vec3f{ data[0], data[1], data[2] }; });

    double glm_construct =
        ComparisonTimer::benchmark_with_result([&]() -> glm::vec3 { return glm::vec3{ data[0], data[1], data[2] }; });

    double eigen_construct = ComparisonTimer::benchmark_with_result(
        [&]() -> Eigen::Vector3f { return Eigen::Vector3f{ data[0], data[1], data[2] }; });

    print_result("Lina", lina_construct, lina_construct);
    print_result("GLM", glm_construct, lina_construct);
    print_result("Eigen", eigen_construct, lina_construct);

    // Vector addition
    print_header("Vector Addition");

    lina::vec3f lv1{ data[0], data[1], data[2] };
    lina::vec3f lv2{ data[3], data[4], data[5] };
    glm::vec3 gv1{ data[0], data[1], data[2] };
    glm::vec3 gv2{ data[3], data[4], data[5] };
    Eigen::Vector3f ev1{ data[0], data[1], data[2] };
    Eigen::Vector3f ev2{ data[3], data[4], data[5] };

    double lina_add = ComparisonTimer::benchmark_with_result([&]() { return lv1 + lv2; });

    double glm_add = ComparisonTimer::benchmark_with_result([&]() { return gv1 + gv2; });

    double eigen_add = ComparisonTimer::benchmark_with_result([&]() { return ev1 + ev2; });

    print_result("Lina", lina_add, lina_add);
    print_result("GLM", glm_add, lina_add);
    print_result("Eigen", eigen_add, lina_add);

    // Dot product
    print_header("Vector Dot Product");

    double lina_dot = ComparisonTimer::benchmark_with_result([&]() { return lina::dot(lv1, lv2); });

    double glm_dot = ComparisonTimer::benchmark_with_result([&]() { return glm::dot(gv1, gv2); });

    double eigen_dot = ComparisonTimer::benchmark_with_result([&]() { return ev1.dot(ev2); });

    print_result("Lina", lina_dot, lina_dot);
    print_result("GLM", glm_dot, lina_dot);
    print_result("Eigen", eigen_dot, lina_dot);

    // Cross product
    print_header("Vector Cross Product");

    double lina_cross = ComparisonTimer::benchmark_with_result([&]() { return lina::cross(lv1, lv2); });

    double glm_cross = ComparisonTimer::benchmark_with_result([&]() { return glm::cross(gv1, gv2); });

    double eigen_cross = ComparisonTimer::benchmark_with_result([&]() { return ev1.cross(ev2); });

    print_result("Lina", lina_cross, lina_cross);
    print_result("GLM", glm_cross, lina_cross);
    print_result("Eigen", eigen_cross, lina_cross);

    // Vector normalization
    print_header("Vector Normalization");

    double lina_norm = ComparisonTimer::benchmark_with_result([&]() { return lina::normalize(lv1); });

    double glm_norm = ComparisonTimer::benchmark_with_result([&]() { return glm::normalize(gv1); });

    double eigen_norm = ComparisonTimer::benchmark_with_result([&]() { return ev1.normalized(); });

    print_result("Lina", lina_norm, lina_norm);
    print_result("GLM", glm_norm, lina_norm);
    print_result("Eigen", eigen_norm, lina_norm);
}

void benchmark_matrix_operations()
{
    // Test data for 4x4 matrices
    auto data = TestData::random_floats(32); // 2 matrices × 16 elements

    // Matrix construction
    print_header("Matrix Construction (4x4)");

    double lina_construct = ComparisonTimer::benchmark_with_result([&]() -> lina::mat4f {
        return lina::mat4f{ data[0], data[1], data[2],  data[3],  data[4],  data[5],  data[6],  data[7],
                            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15] };
    });

    double glm_construct = ComparisonTimer::benchmark_with_result([&]() -> glm::mat4 {
        return glm::mat4{ data[0], data[1], data[2],  data[3],  data[4],  data[5],  data[6],  data[7],
                          data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15] };
    });

    double eigen_construct = ComparisonTimer::benchmark_with_result([&]() -> Eigen::Matrix4f {
        Eigen::Matrix4f m;
        m << data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10],
            data[11], data[12], data[13], data[14], data[15];
        return m;
    });

    print_result("Lina", lina_construct, lina_construct);
    print_result("GLM", glm_construct, lina_construct);
    print_result("Eigen", eigen_construct, lina_construct);

    // Matrix multiplication
    print_header("Matrix Multiplication (4x4)");

    lina::mat4f lm1{ data[0], data[1], data[2],  data[3],  data[4],  data[5],  data[6],  data[7],
                     data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15] };
    lina::mat4f lm2{ data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
                     data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31] };

    glm::mat4 gm1{ data[0], data[1], data[2],  data[3],  data[4],  data[5],  data[6],  data[7],
                   data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15] };
    glm::mat4 gm2{ data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
                   data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31] };

    Eigen::Matrix4f em1, em2;
    em1 << data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11],
        data[12], data[13], data[14], data[15];
    em2 << data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23], data[24], data[25], data[26],
        data[27], data[28], data[29], data[30], data[31];

    double lina_mult = ComparisonTimer::benchmark_with_result([&]() { return lm1 * lm2; });

    double glm_mult = ComparisonTimer::benchmark_with_result([&]() { return gm1 * gm2; });

    double eigen_mult = ComparisonTimer::benchmark_with_result([&]() { return em1 * em2; });

    print_result("Lina", lina_mult, lina_mult);
    print_result("GLM", glm_mult, lina_mult);
    print_result("Eigen", eigen_mult, lina_mult);

    // Matrix transpose
    print_header("Matrix Transpose (4x4)");

    double lina_transpose = ComparisonTimer::benchmark_with_result([&]() { return lina::transpose(lm1); });

    double glm_transpose = ComparisonTimer::benchmark_with_result([&]() { return glm::transpose(gm1); });

    double eigen_transpose = ComparisonTimer::benchmark_with_result([&]() { return em1.transpose(); });

    print_result("Lina", lina_transpose, lina_transpose);
    print_result("GLM", glm_transpose, lina_transpose);
    print_result("Eigen", eigen_transpose, lina_transpose);

    // Matrix determinant
    print_header("Matrix Determinant (4x4)");

    double lina_det = ComparisonTimer::benchmark_with_result([&]() { return lina::det(lm1); });

    double glm_det = ComparisonTimer::benchmark_with_result([&]() { return glm::determinant(gm1); });

    double eigen_det = ComparisonTimer::benchmark_with_result([&]() { return em1.determinant(); });

    print_result("Lina", lina_det, lina_det);
    print_result("GLM", glm_det, lina_det);
    print_result("Eigen", eigen_det, lina_det);
}

void benchmark_transformation_operations()
{
    auto data = TestData::random_floats(12);

    // Translation matrix
    print_header("Translation Matrix Creation");

    double lina_translation = ComparisonTimer::benchmark_with_result(
        [&]() { return lina::translation(lina::vec3f{ data[0], data[1], data[2] }); });

    double glm_translation = ComparisonTimer::benchmark_with_result(
        [&]() { return glm::translate(glm::mat4(1.0f), glm::vec3{ data[0], data[1], data[2] }); });

    double eigen_translation = ComparisonTimer::benchmark_with_result([&]() {
        Eigen::Affine3f t = Eigen::Affine3f::Identity();
        t.translation() << data[0], data[1], data[2];
        return t.matrix();
    });

    print_result("Lina", lina_translation, lina_translation);
    print_result("GLM", glm_translation, lina_translation);
    print_result("Eigen", eigen_translation, lina_translation);

    // Rotation matrix (around X axis)
    print_header("Rotation Matrix Creation (X-axis)");

    float angle = data[0];

    double lina_rotation = ComparisonTimer::benchmark_with_result([&]() { return lina::rotation_x(angle); });

    double glm_rotation = ComparisonTimer::benchmark_with_result(
        [&]() { return glm::rotate(glm::mat4(1.0f), angle, glm::vec3(1.0f, 0.0f, 0.0f)); });

    double eigen_rotation = ComparisonTimer::benchmark_with_result([&]() {
        Eigen::Matrix4f mat   = Eigen::Matrix4f::Identity();
        mat.block<3, 3>(0, 0) = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitX()).toRotationMatrix();
        return mat;
    });

    print_result("Lina", lina_rotation, lina_rotation);
    print_result("GLM", glm_rotation, lina_rotation);
    print_result("Eigen", eigen_rotation, lina_rotation);

    // Perspective projection
    print_header("Perspective Projection Matrix");

    float fovy       = data[3];
    float aspect     = data[4];
    float near_plane = 0.1f;
    float far_plane  = 100.0f;

    double lina_perspective = ComparisonTimer::benchmark_with_result(
        [&]() { return lina::perspective(fovy, aspect, near_plane, far_plane); });

    double glm_perspective =
        ComparisonTimer::benchmark_with_result([&]() { return glm::perspective(fovy, aspect, near_plane, far_plane); });

    double eigen_perspective = ComparisonTimer::benchmark_with_result([&]() {
        // Manual perspective matrix construction for Eigen
        Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
        float f              = 1.0f / std::tan(fovy * 0.5f);
        proj(0, 0)           = f / aspect;
        proj(1, 1)           = f;
        proj(2, 2)           = (far_plane + near_plane) / (near_plane - far_plane);
        proj(2, 3)           = (2.0f * far_plane * near_plane) / (near_plane - far_plane);
        proj(3, 2)           = -1.0f;
        return proj;
    });

    print_result("Lina", lina_perspective, lina_perspective);
    print_result("GLM", glm_perspective, lina_perspective);
    print_result("Eigen", eigen_perspective, lina_perspective);
}

void run_comparison_benchmarks()
{
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << " LINA vs GLM vs EIGEN - PERFORMANCE COMPARISON" << std::endl;
    std::cout << " Iterations: " << ComparisonTimer::DEFAULT_ITERATIONS << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    benchmark_vector_operations();
    benchmark_matrix_operations();
    benchmark_transformation_operations();

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << " BENCHMARK COMPLETE" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "\nNotes:" << std::endl;
    std::cout << "• Lower time values indicate better performance" << std::endl;
    std::cout << "• Relative shows performance compared to Lina (baseline)" << std::endl;
    std::cout << "• Speedup shows how many times faster compared to baseline" << std::endl;
    std::cout << "• Results may vary based on compiler optimizations and CPU" << std::endl;
}