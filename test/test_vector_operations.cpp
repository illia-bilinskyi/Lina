#include "lina/lina.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace lina;

class VectorOperationsTest : public ::testing::Test
{
  protected:
    static constexpr vec3f v1{ 1.0f, 2.0f, 3.0f };
    static constexpr vec3f v2{ 4.0f, 5.0f, 6.0f };
    static constexpr vec3f unit_x{ 1.0f, 0.0f, 0.0f };
    static constexpr vec3f unit_y{ 0.0f, 1.0f, 0.0f };
    static constexpr vec3f unit_z{ 0.0f, 0.0f, 1.0f };
    static constexpr vec3f zero_vec{ 0.0f, 0.0f, 0.0f };
};

TEST_F(VectorOperationsTest, DotProduct_Basic)
{
    constexpr float result = dot(v1, v2);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_FLOAT_EQ(result, 32.0f);
}

TEST_F(VectorOperationsTest, DotProduct_OrthogonalVectors)
{
    constexpr float result = dot(unit_x, unit_y);
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST_F(VectorOperationsTest, DotProduct_SameVector)
{
    constexpr float result = dot(v1, v1);
    // 1² + 2² + 3² = 1 + 4 + 9 = 14
    EXPECT_FLOAT_EQ(result, 14.0f);
}

TEST_F(VectorOperationsTest, DotProduct_WithZero)
{
    constexpr float result = dot(v1, zero_vec);
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST_F(VectorOperationsTest, CrossProduct_BasicVectors)
{
    constexpr vec3f result = cross(v1, v2);
    // cross([1,2,3], [4,5,6]) = [2*6-3*5, 3*4-1*6, 1*5-2*4] = [-3, 6, -3]

    EXPECT_FLOAT_EQ(result.x, -3.0f);
    EXPECT_FLOAT_EQ(result.y, 6.0f);
    EXPECT_FLOAT_EQ(result.z, -3.0f);
}

TEST_F(VectorOperationsTest, CrossProduct_OrthogonalBasis)
{
    constexpr vec3f result = cross(unit_x, unit_y);

    EXPECT_FLOAT_EQ(result.x, 0.0f);
    EXPECT_FLOAT_EQ(result.y, 0.0f);
    EXPECT_FLOAT_EQ(result.z, 1.0f);
}

TEST_F(VectorOperationsTest, CrossProduct_AntiCommutative)
{
    constexpr vec3f result1 = cross(v1, v2);
    constexpr vec3f result2 = cross(v2, v1);

    EXPECT_FLOAT_EQ(result1.x, -result2.x);
    EXPECT_FLOAT_EQ(result1.y, -result2.y);
    EXPECT_FLOAT_EQ(result1.z, -result2.z);
}

TEST_F(VectorOperationsTest, CrossProduct_SameVector)
{
    constexpr vec3f result = cross(v1, v1);

    EXPECT_FLOAT_EQ(result.x, 0.0f);
    EXPECT_FLOAT_EQ(result.y, 0.0f);
    EXPECT_FLOAT_EQ(result.z, 0.0f);
}

TEST_F(VectorOperationsTest, Norm2_BasicVector)
{
    constexpr float result = norm2(v1);
    // 1² + 2² + 3² = 14
    EXPECT_FLOAT_EQ(result, 14.0f);
}

TEST_F(VectorOperationsTest, Norm2_UnitVector)
{
    constexpr float result = norm2(unit_x);
    EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST_F(VectorOperationsTest, Norm2_ZeroVector)
{
    constexpr float result = norm2(zero_vec);
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST_F(VectorOperationsTest, Norm_BasicVector)
{
    float result = norm(v1);
    // sqrt(14)
    EXPECT_FLOAT_EQ(result, std::sqrt(14.0f));
}

TEST_F(VectorOperationsTest, Norm_UnitVector)
{
    float result = norm(unit_x);
    EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST_F(VectorOperationsTest, Norm_ZeroVector)
{
    float result = norm(zero_vec);
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST_F(VectorOperationsTest, Length_SameAsNorm)
{
    float norm_result   = norm(v1);
    float length_result = length(v1);

    EXPECT_FLOAT_EQ(norm_result, length_result);
}

TEST_F(VectorOperationsTest, Normalize_BasicVector)
{
    vec3f result          = normalize(v1);
    float original_length = norm(v1);

    // Normalized vector should have length 1
    EXPECT_NEAR(norm(result), 1.0f, 1e-6f);

    // Should be parallel to original (same direction)
    vec3f expected = v1 / original_length;
    EXPECT_NEAR(result.x, expected.x, 1e-6f);
    EXPECT_NEAR(result.y, expected.y, 1e-6f);
    EXPECT_NEAR(result.z, expected.z, 1e-6f);
}

TEST_F(VectorOperationsTest, Normalize_UnitVector)
{
    vec3f result = normalize(unit_x);

    EXPECT_FLOAT_EQ(result.x, 1.0f);
    EXPECT_FLOAT_EQ(result.y, 0.0f);
    EXPECT_FLOAT_EQ(result.z, 0.0f);
}

TEST_F(VectorOperationsTest, Distance_BasicVectors)
{
    float result = distance(v1, v2);
    // distance([1,2,3], [4,5,6]) = norm([3,3,3]) = sqrt(27) = 3*sqrt(3)

    EXPECT_FLOAT_EQ(result, std::sqrt(27.0f));
}

TEST_F(VectorOperationsTest, Distance_SameVector)
{
    float result = distance(v1, v1);
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST_F(VectorOperationsTest, Distance_WithZero)
{
    float result = distance(v1, zero_vec);
    EXPECT_FLOAT_EQ(result, norm(v1));
}

TEST_F(VectorOperationsTest, Distance_Symmetric)
{
    float result1 = distance(v1, v2);
    float result2 = distance(v2, v1);

    EXPECT_FLOAT_EQ(result1, result2);
}

TEST_F(VectorOperationsTest, Column_ExtractFromMatrix)
{
    constexpr mat3<float> m{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };

    constexpr auto col0 = m.col<0>();
    constexpr vec3f vec_col0{ col0 };

    EXPECT_EQ(vec_col0.x, 1.0f);
    EXPECT_EQ(vec_col0.y, 4.0f);
    EXPECT_EQ(vec_col0.z, 7.0f);

    constexpr auto col1 = m.col<1>();
    constexpr vec3f vec_col1{ col1 };

    EXPECT_EQ(vec_col1.x, 2.0f);
    EXPECT_EQ(vec_col1.y, 5.0f);
    EXPECT_EQ(vec_col1.z, 8.0f);
}

TEST_F(VectorOperationsTest, Row_ExtractFromMatrix)
{
    constexpr mat3<float> m{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };

    constexpr auto row0 = m.row<0>();
    constexpr vec3f vec_row0{ row0 };

    EXPECT_EQ(vec_row0.x, 1.0f);
    EXPECT_EQ(vec_row0.y, 2.0f);
    EXPECT_EQ(vec_row0.z, 3.0f);

    constexpr auto row1 = m.row<1>();
    constexpr vec3f vec_row1{ row1 };

    EXPECT_EQ(vec_row1.x, 4.0f);
    EXPECT_EQ(vec_row1.y, 5.0f);
    EXPECT_EQ(vec_row1.z, 6.0f);
}