#include <gtest/gtest.h>
#include <cmath>
#include "lina/lina.h"
constexpr double EPSILON = 1e-6;

// ========================= Constexpr Sin Tests =========================

TEST(ConstexprMathTest, SinBasicValues)
{
    // Test basic known values
    EXPECT_NEAR(lina::sin(0.0), 0.0, EPSILON);
    EXPECT_NEAR(lina::sin(lina::pi<double> / 6), 0.5, EPSILON);          // sin(30°) = 0.5
    EXPECT_NEAR(lina::sin(lina::pi<double> / 4), std::sqrt(2)/2, EPSILON);  // sin(45°) = √2/2
    EXPECT_NEAR(lina::sin(lina::pi<double> / 3), std::sqrt(3)/2, EPSILON);  // sin(60°) = √3/2
    EXPECT_NEAR(lina::sin(lina::pi<double> / 2), 1.0, EPSILON);          // sin(90°) = 1
    EXPECT_NEAR(lina::sin(lina::pi<double>), 0.0, EPSILON);              // sin(180°) = 0
    EXPECT_NEAR(lina::sin(3 * lina::pi<double> / 2), -1.0, EPSILON);     // sin(270°) = -1
}

TEST(ConstexprMathTest, SinNegativeValues)
{
    // Test negative values (sin is odd function)
    EXPECT_NEAR(lina::sin(-lina::pi<double> / 6), -0.5, EPSILON);
    EXPECT_NEAR(lina::sin(-lina::pi<double> / 4), -std::sqrt(2)/2, EPSILON);
    EXPECT_NEAR(lina::sin(-lina::pi<double> / 2), -1.0, EPSILON);
}

TEST(ConstexprMathTest, SinLargeValues)
{
    // Test that normalization works for large values
    EXPECT_NEAR(lina::sin(5 * lina::pi<double>), std::sin(5 * lina::pi<double>), EPSILON);
    EXPECT_NEAR(lina::sin(-7 * lina::pi<double>), std::sin(-7 * lina::pi<double>), EPSILON);
}

TEST(ConstexprMathTest, SinCompileTimeEvaluation)
{
    // Test that sin can be evaluated at compile-time
    constexpr double sin_zero = lina::sin(0.0);
    constexpr double sin_pi_2 = lina::sin(lina::pi<double> / 2);
    constexpr double sin_pi = lina::sin(lina::pi<double>);
    
    EXPECT_NEAR(sin_zero, 0.0, EPSILON);
    EXPECT_NEAR(sin_pi_2, 1.0, EPSILON);
    EXPECT_NEAR(sin_pi, 0.0, EPSILON);
    
    // Test with float
    constexpr float sin_pi_6_f = lina::sin(lina::pi<float> / 6);
    EXPECT_NEAR(sin_pi_6_f, 0.5f, 1e-5f);
}

// ========================= Constexpr Cos Tests =========================

TEST(ConstexprMathTest, CosBasicValues)
{
    // Test basic known values
    EXPECT_NEAR(lina::cos(0.0), 1.0, EPSILON);
    EXPECT_NEAR(lina::cos(lina::pi<double> / 6), std::sqrt(3)/2, EPSILON);  // cos(30°) = √3/2
    EXPECT_NEAR(lina::cos(lina::pi<double> / 4), std::sqrt(2)/2, EPSILON);  // cos(45°) = √2/2
    EXPECT_NEAR(lina::cos(lina::pi<double> / 3), 0.5, EPSILON);          // cos(60°) = 0.5
    EXPECT_NEAR(lina::cos(lina::pi<double> / 2), 0.0, EPSILON);          // cos(90°) = 0
    EXPECT_NEAR(lina::cos(lina::pi<double>), -1.0, EPSILON);             // cos(180°) = -1
    EXPECT_NEAR(lina::cos(3 * lina::pi<double> / 2), 0.0, EPSILON);      // cos(270°) = 0
}

TEST(ConstexprMathTest, CosNegativeValues)
{
    // Test negative values (cos is even function)
    EXPECT_NEAR(lina::cos(-lina::pi<double> / 6), std::sqrt(3)/2, EPSILON);
    EXPECT_NEAR(lina::cos(-lina::pi<double> / 4), std::sqrt(2)/2, EPSILON);
    EXPECT_NEAR(lina::cos(-lina::pi<double> / 2), 0.0, EPSILON);
}

TEST(ConstexprMathTest, CosCompileTimeEvaluation)
{
    // Test that cos can be evaluated at compile-time
    constexpr double cos_zero = lina::cos(0.0);
    constexpr double cos_pi_2 = lina::cos(lina::pi<double> / 2);
    constexpr double cos_pi = lina::cos(lina::pi<double>);
    
    EXPECT_NEAR(cos_zero, 1.0, EPSILON);
    EXPECT_NEAR(cos_pi_2, 0.0, EPSILON);
    EXPECT_NEAR(cos_pi, -1.0, EPSILON);
    
    // Test with float
    constexpr float cos_pi_3_f = lina::cos(lina::pi<float> / 3);
    EXPECT_NEAR(cos_pi_3_f, 0.5f, 1e-5f);
}

// ========================= Constexpr Tan Tests =========================

TEST(ConstexprMathTest, TanBasicValues)
{
    // Test basic known values
    EXPECT_NEAR(lina::tan(0.0), 0.0, EPSILON);
    EXPECT_NEAR(lina::tan(lina::pi<double> / 6), 1.0/std::sqrt(3), EPSILON);  // tan(30°) = 1/√3
    EXPECT_NEAR(lina::tan(lina::pi<double> / 4), 1.0, EPSILON);               // tan(45°) = 1
    EXPECT_NEAR(lina::tan(lina::pi<double> / 3), std::sqrt(3), EPSILON);      // tan(60°) = √3
    EXPECT_NEAR(lina::tan(lina::pi<double>), 0.0, EPSILON);                   // tan(180°) = 0
}

TEST(ConstexprMathTest, TanNegativeValues)
{
    // Test negative values (tan is odd function)
    EXPECT_NEAR(lina::tan(-lina::pi<double> / 6), -1.0/std::sqrt(3), EPSILON);
    EXPECT_NEAR(lina::tan(-lina::pi<double> / 4), -1.0, EPSILON);
}

TEST(ConstexprMathTest, TanCompileTimeEvaluation)
{
    // Test that tan can be evaluated at compile-time
    constexpr double tan_zero = lina::tan(0.0);
    constexpr double tan_pi_4 = lina::tan(lina::pi<double> / 4);
    constexpr double tan_pi = lina::tan(lina::pi<double>);
    
    EXPECT_NEAR(tan_zero, 0.0, EPSILON);
    EXPECT_NEAR(tan_pi_4, 1.0, EPSILON);
    EXPECT_NEAR(tan_pi, 0.0, EPSILON);
    
    // Test with float
    constexpr float tan_pi_6_f = lina::tan(lina::pi<float> / 6);
    EXPECT_NEAR(tan_pi_6_f, 1.0f/std::sqrt(3.0f), 1e-4f);
}

// ========================= Constexpr Sqrt Tests =========================

TEST(ConstexprMathTest, SqrtBasicValues)
{
    // Test basic known values
    EXPECT_NEAR(lina::sqrt(0.0), 0.0, EPSILON);
    EXPECT_NEAR(lina::sqrt(1.0), 1.0, EPSILON);
    EXPECT_NEAR(lina::sqrt(4.0), 2.0, EPSILON);
    EXPECT_NEAR(lina::sqrt(9.0), 3.0, EPSILON);
    EXPECT_NEAR(lina::sqrt(16.0), 4.0, EPSILON);
    EXPECT_NEAR(lina::sqrt(25.0), 5.0, EPSILON);
    EXPECT_NEAR(lina::sqrt(100.0), 10.0, EPSILON);
}

TEST(ConstexprMathTest, SqrtFractionalValues)
{
    // Test fractional values
    EXPECT_NEAR(lina::sqrt(0.25), 0.5, EPSILON);
    EXPECT_NEAR(lina::sqrt(0.5), std::sqrt(0.5), EPSILON);
    EXPECT_NEAR(lina::sqrt(2.0), std::sqrt(2.0), EPSILON);
    EXPECT_NEAR(lina::sqrt(3.0), std::sqrt(3.0), EPSILON);
    EXPECT_NEAR(lina::sqrt(0.01), 0.1, EPSILON);
}

TEST(ConstexprMathTest, SqrtNegativeValues)
{
    // Test negative values (should return 0)
    EXPECT_EQ(lina::sqrt(-1.0), 0.0);
    EXPECT_EQ(lina::sqrt(-4.0), 0.0);
    EXPECT_EQ(lina::sqrt(-100.0), 0.0);
}

TEST(ConstexprMathTest, SqrtCompileTimeEvaluation)
{
    // Test that sqrt can be evaluated at compile-time
    constexpr double sqrt_0 = lina::sqrt(0.0);
    constexpr double sqrt_1 = lina::sqrt(1.0);
    constexpr double sqrt_4 = lina::sqrt(4.0);
    constexpr double sqrt_9 = lina::sqrt(9.0);
    constexpr double sqrt_2 = lina::sqrt(2.0);
    
    EXPECT_EQ(sqrt_0, 0.0);
    EXPECT_EQ(sqrt_1, 1.0);
    EXPECT_EQ(sqrt_4, 2.0);
    EXPECT_EQ(sqrt_9, 3.0);
    EXPECT_NEAR(sqrt_2, std::sqrt(2.0), EPSILON);
    
    // Test with float
    constexpr float sqrt_25_f = lina::sqrt(25.0f);
    EXPECT_EQ(sqrt_25_f, 5.0f);
}

TEST(ConstexprMathTest, SqrtAccuracyComparison)
{
    // Compare with std::sqrt for various values
    std::vector<double> test_values = {
        0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 
        5.0, 7.0, 10.0, 15.0, 25.0, 50.0, 100.0, 1000.0
    };
    
    for (double x : test_values) {
        EXPECT_NEAR(lina::sqrt(x), std::sqrt(x), EPSILON) 
            << "Failed for sqrt(" << x << ")";
    }
}

TEST(ConstexprMathTest, SqrtLargeValues)
{
    // Test larger values
    EXPECT_NEAR(lina::sqrt(10000.0), 100.0, EPSILON);
    EXPECT_NEAR(lina::sqrt(1000000.0), 1000.0, EPSILON);
}

TEST(ConstexprMathTest, SqrtSmallValues)
{
    // Test very small values
    EXPECT_NEAR(lina::sqrt(1e-6), 1e-3, 1e-9);
    EXPECT_NEAR(lina::sqrt(1e-10), 1e-5, 1e-11);
}

// ========================= Accuracy Comparison Tests =========================

TEST(ConstexprMathTest, CompareWithStdMath)
{
    // Compare our constexpr implementations with std library functions
    std::vector<double> test_values = {
        0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 
        lina::pi<double> / 6, lina::pi<double> / 4, lina::pi<double> / 3,
        lina::pi<double> / 2, lina::pi<double>, 3 * lina::pi<double> / 2,
        -0.5, -1.0, -lina::pi<double> / 4
    };
    
    for (double x : test_values)
    {
        EXPECT_NEAR(lina::sin(x), std::sin(x), EPSILON) << "Failed for sin(" << x << ")";
        EXPECT_NEAR(lina::cos(x), std::cos(x), EPSILON) << "Failed for cos(" << x << ")";
        
        // Skip tan comparison for values close to π/2 + nπ where tan is undefined
        if (std::abs(std::remainder(x - lina::pi<double>/2, lina::pi<double>)) > 0.1)
        {
            EXPECT_NEAR(lina::tan(x), std::tan(x), EPSILON) << "Failed for tan(" << x << ")";
        }
    }
}

// ========================= Trigonometric Identity Tests =========================

TEST(ConstexprMathTest, TrigonometricIdentities)
{
    std::vector<double> test_values = {0.1, 0.5, 1.0, 1.5, 2.0, 3.0};
    
    for (double x : test_values)
    {
        // sin²(x) + cos²(x) = 1
        double sin_x = lina::sin(x);
        double cos_x = lina::cos(x);
        EXPECT_NEAR(sin_x * sin_x + cos_x * cos_x, 1.0, EPSILON) 
            << "Pythagorean identity failed for x=" << x;
        
        // tan(x) = sin(x) / cos(x) (when cos(x) ≠ 0)
        if (std::abs(cos_x) > EPSILON)
        {
            double tan_x = lina::tan(x);
            EXPECT_NEAR(tan_x, sin_x / cos_x, EPSILON) 
                << "tan = sin/cos identity failed for x=" << x;
        }
    }
}

// ========================= Vector Operations with Constexpr Math Tests =========================

TEST(ConstexprMathTest, ConstexprVectorOperations)
{
    // Test that vector operations using sqrt work at compile-time
    constexpr lina::vec3<double> v1{3.0, 4.0, 0.0};
    constexpr lina::vec3<double> v2{1.0, 1.0, 1.0};
    
    constexpr double norm_v1 = lina::norm(v1);
    constexpr double length_v1 = lina::length(v1);
    constexpr double distance_v1_v2 = lina::distance(v1, v2);
    
    EXPECT_NEAR(norm_v1, 5.0, EPSILON);  // sqrt(3²+4²) = 5
    EXPECT_NEAR(length_v1, 5.0, EPSILON);
    EXPECT_EQ(norm_v1, length_v1);
    
    // Distance between (3,4,0) and (1,1,1) = sqrt((3-1)²+(4-1)²+(0-1)²) = sqrt(4+9+1) = sqrt(14)
    EXPECT_NEAR(distance_v1_v2, lina::sqrt(14.0), EPSILON);
}

TEST(ConstexprMathTest, ConstexprNormalize)
{
    // Test normalization at compile-time
    constexpr lina::vec3<double> v{3.0, 4.0, 0.0};
    constexpr auto normalized = lina::normalize(v);
    
    // Normalized vector should have length 1
    constexpr double norm_normalized = lina::norm(normalized);
    EXPECT_NEAR(norm_normalized, 1.0, EPSILON);
    
    // Components should be original divided by magnitude
    EXPECT_NEAR(normalized.x, 3.0/5.0, EPSILON);
    EXPECT_NEAR(normalized.y, 4.0/5.0, EPSILON);
    EXPECT_NEAR(normalized.z, 0.0, EPSILON);
}

// ========================= Constexpr Rotation Matrix Tests =========================

TEST(ConstexprMathTest, ConstexprRotationMatrices)
{
    // Test that rotation matrices can be created at compile-time
    constexpr auto rot_x_90 = lina::rotation_x<double>(lina::pi<double> / 2);
    constexpr auto rot_y_90 = lina::rotation_y<double>(lina::pi<double> / 2);
    constexpr auto rot_z_90 = lina::rotation_z<double>(lina::pi<double> / 2);
    
    // Test X rotation by 90 degrees
    EXPECT_NEAR(rot_x_90(0, 0), 1.0, EPSILON);
    EXPECT_NEAR(rot_x_90(1, 1), 0.0, EPSILON);
    EXPECT_NEAR(rot_x_90(1, 2), -1.0, EPSILON);
    EXPECT_NEAR(rot_x_90(2, 1), 1.0, EPSILON);
    EXPECT_NEAR(rot_x_90(2, 2), 0.0, EPSILON);
    
    // Test Y rotation by 90 degrees
    EXPECT_NEAR(rot_y_90(0, 0), 0.0, EPSILON);
    EXPECT_NEAR(rot_y_90(0, 2), 1.0, EPSILON);
    EXPECT_NEAR(rot_y_90(1, 1), 1.0, EPSILON);
    EXPECT_NEAR(rot_y_90(2, 0), -1.0, EPSILON);
    EXPECT_NEAR(rot_y_90(2, 2), 0.0, EPSILON);
    
    // Test Z rotation by 90 degrees
    EXPECT_NEAR(rot_z_90(0, 0), 0.0, EPSILON);
    EXPECT_NEAR(rot_z_90(0, 1), -1.0, EPSILON);
    EXPECT_NEAR(rot_z_90(1, 0), 1.0, EPSILON);
    EXPECT_NEAR(rot_z_90(1, 1), 0.0, EPSILON);
    EXPECT_NEAR(rot_z_90(2, 2), 1.0, EPSILON);
}

TEST(ConstexprMathTest, ConstexprEulerRotation)
{
    // Test Euler rotation at compile-time
    constexpr auto euler_rot = lina::rotation<double>(
        lina::pi<double> / 4,  // α (X rotation)
        lina::pi<double> / 6,  // β (Y rotation) 
        lina::pi<double> / 3   // γ (Z rotation)
    );
    
    // The matrix should be valid (determinant ≈ 1)
    double det = lina::det(lina::mat3<double>{
        euler_rot(0,0), euler_rot(0,1), euler_rot(0,2),
        euler_rot(1,0), euler_rot(1,1), euler_rot(1,2),
        euler_rot(2,0), euler_rot(2,1), euler_rot(2,2)
    });
    EXPECT_NEAR(det, 1.0, EPSILON);
}

TEST(ConstexprMathTest, ConstexprPerspectiveMatrix)
{
    // Test that perspective matrix can be created at compile-time
    constexpr auto persp = lina::perspective<double>(
        lina::radians(90.0),  // 90 degree FOV
        16.0 / 9.0,           // 16:9 aspect ratio
        0.1,                  // near plane
        100.0                 // far plane
    );
    
    // Verify key properties of perspective matrix
    EXPECT_GT(persp(0, 0), 0.0);  // X scaling should be positive
    EXPECT_GT(persp(1, 1), 0.0);  // Y scaling should be positive
    EXPECT_LT(persp(2, 2), 0.0);  // Z should be negative (NDC mapping)
    EXPECT_EQ(persp(3, 2), -1.0); // Perspective divide trigger
}

// ========================= Performance Tests =========================

TEST(ConstexprMathTest, CompileTimeVsRuntime)
{
    // This test ensures that constexpr functions can indeed be evaluated at compile-time
    // The actual performance measurement would require benchmarking tools
    
    // These should all be computed at compile-time
    constexpr double ct_sin = lina::sin(1.0);
    constexpr double ct_cos = lina::cos(1.0);
    constexpr double ct_tan = lina::tan(1.0);
    constexpr double ct_sqrt = lina::sqrt(16.0);
    constexpr auto ct_matrix = lina::rotation_x<double>(lina::pi<double> / 4);
    
    // Runtime versions for comparison
    double rt_sin = lina::sin(1.0);
    double rt_cos = lina::cos(1.0);
    double rt_tan = lina::tan(1.0);
    double rt_sqrt = lina::sqrt(16.0);
    auto rt_matrix = lina::rotation_x<double>(lina::pi<double> / 4);
    
    // Results should be identical
    EXPECT_EQ(ct_sin, rt_sin);
    EXPECT_EQ(ct_cos, rt_cos);
    EXPECT_EQ(ct_tan, rt_tan);
    EXPECT_EQ(ct_sqrt, rt_sqrt);
    
    for (std::size_t i = 0; i < 4; ++i)
    {
        for (std::size_t j = 0; j < 4; ++j)
        {
            EXPECT_EQ(ct_matrix(i, j), rt_matrix(i, j));
        }
    }
}