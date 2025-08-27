#include "lina.h"
#include <gtest/gtest.h>

using namespace lina;

class VectorTest : public ::testing::Test
{
  protected:
    static constexpr vec3f v1{1.0f, 2.0f, 3.0f};
    static constexpr vec3f v2{4.0f, 5.0f, 6.0f};
    static constexpr vec3f zero_vec{0.0f, 0.0f, 0.0f};
};

TEST_F(VectorTest, DefaultConstructor)
{
    constexpr vec3f v;
    EXPECT_EQ(v.x, 0.0f);
    EXPECT_EQ(v.y, 0.0f);
    EXPECT_EQ(v.z, 0.0f);
}

TEST_F(VectorTest, FillConstructor)
{
    constexpr vec3f v(2.0f);
    EXPECT_EQ(v.x, 2.0f);
    EXPECT_EQ(v.y, 2.0f);
    EXPECT_EQ(v.z, 2.0f);

    constexpr vec3f v2{2.0f};
    EXPECT_EQ(v2.x, 2.0f);
    EXPECT_EQ(v2.y, 0.0f);
    EXPECT_EQ(v2.z, 0.0f);
}

TEST_F(VectorTest, ThreeParameterConstructor)
{
    constexpr vec3f v{1.0f, 2.0f, 3.0f};
    EXPECT_EQ(v.x, 1.0f);
    EXPECT_EQ(v.y, 2.0f);
    EXPECT_EQ(v.z, 3.0f);
}

TEST_F(VectorTest, InitializerListPartial)
{
    constexpr vec3f v{1.0f, 2.0f}; // Only x and y provided
    EXPECT_EQ(v.x, 1.0f);
    EXPECT_EQ(v.y, 2.0f);
    EXPECT_EQ(v.z, 0.0f);
}

TEST_F(VectorTest, CopyConstructor)
{
    constexpr vec3f original{1.0f, 2.0f, 3.0f};
    constexpr vec3f copy(original);

    EXPECT_EQ(copy.x, 1.0f);
    EXPECT_EQ(copy.y, 2.0f);
    EXPECT_EQ(copy.z, 3.0f);
}

TEST_F(VectorTest, AssignmentOperator)
{
    constexpr vec3f v = v1;

    EXPECT_EQ(v.x, 1.0f);
    EXPECT_EQ(v.y, 2.0f);
    EXPECT_EQ(v.z, 3.0f);
}

TEST_F(VectorTest, SelfAssignment)
{
    vec3f v{1.0f, 2.0f, 3.0f};
    v = v; // Self assignment

    EXPECT_EQ(v.x, 1.0f);
    EXPECT_EQ(v.y, 2.0f);
    EXPECT_EQ(v.z, 3.0f);
}

TEST_F(VectorTest, ConversionToMatrix)
{
    constexpr vec3f v{1.0f, 2.0f, 3.0f};

    // Test conversion to column vector
    constexpr auto col_matrix = static_cast<mat<float, 3, 1>>(v);
    EXPECT_EQ(col_matrix(0, 0), 1.0f);
    EXPECT_EQ(col_matrix(1, 0), 2.0f);
    EXPECT_EQ(col_matrix(2, 0), 3.0f);

    // Test conversion to row vector
    constexpr auto row_matrix = static_cast<mat<float, 1, 3>>(v);
    EXPECT_EQ(row_matrix(0, 0), 1.0f);
    EXPECT_EQ(row_matrix(0, 1), 2.0f);
    EXPECT_EQ(row_matrix(0, 2), 3.0f);
}

TEST_F(VectorTest, VectorValueTypeConversion)
{
    constexpr vec3d v = static_cast<vec3d>(v1);

    EXPECT_FLOAT_EQ(v.x, 1.0);
    EXPECT_FLOAT_EQ(v.y, 2.0);
    EXPECT_FLOAT_EQ(v.z, 3.0);
}

TEST_F(VectorTest, GetTemplated)
{
    constexpr vec3f v{1.0f, 2.0f, 3.0f};

    EXPECT_EQ(v.get<0>(), 1.0f);
    EXPECT_EQ(v.get<1>(), 2.0f);
    EXPECT_EQ(v.get<2>(), 3.0f);
}

TEST_F(VectorTest, FunctionCallOperator)
{
    constexpr vec3f v{1.0f, 2.0f, 3.0f};

    EXPECT_EQ(v(0), 1.0f);
    EXPECT_EQ(v(1), 2.0f);
    EXPECT_EQ(v(2), 3.0f);
}

TEST_F(VectorTest, DataAccess)
{
    vec3f v{1.0f, 2.0f, 3.0f};
    float* data = v.data();

    EXPECT_EQ(data[0], 1.0f);
    EXPECT_EQ(data[1], 2.0f);
    EXPECT_EQ(data[2], 3.0f);
}

TEST_F(VectorTest, UnionAccess)
{
    constexpr vec3f v{1.0f, 2.0f, 3.0f};

    EXPECT_EQ(v.x, v.a[0]);
    EXPECT_EQ(v.y, v.a[1]);
    EXPECT_EQ(v.z, v.a[2]);
}

TEST_F(VectorTest, UnaryPlus)
{
    constexpr vec3f result = +v1;

    EXPECT_EQ(result.x, 1.0f);
    EXPECT_EQ(result.y, 2.0f);
    EXPECT_EQ(result.z, 3.0f);
}

TEST_F(VectorTest, UnaryMinus)
{
    constexpr vec3f result = v1.operator-();

    EXPECT_EQ(result.x, -1.0f);
    EXPECT_EQ(result.y, -2.0f);
    EXPECT_EQ(result.z, -3.0f);
}

TEST_F(VectorTest, VectorAddition)
{
    constexpr vec3f result = v1 + v2;

    EXPECT_EQ(result.x, 5.0f);
    EXPECT_EQ(result.y, 7.0f);
    EXPECT_EQ(result.z, 9.0f);
}

TEST_F(VectorTest, VectorSubtraction)
{
    constexpr vec3f result = v2 - v1;

    EXPECT_EQ(result.x, 3.0f);
    EXPECT_EQ(result.y, 3.0f);
    EXPECT_EQ(result.z, 3.0f);
}

TEST_F(VectorTest, ScalarMultiplication)
{
    constexpr vec3f result = v1 * 2.0f;

    EXPECT_EQ(result.x, 2.0f);
    EXPECT_EQ(result.y, 4.0f);
    EXPECT_EQ(result.z, 6.0f);
}

TEST_F(VectorTest, ScalarDivision)
{
    constexpr vec3f result = v1 / 2.0f;

    EXPECT_FLOAT_EQ(result.x, 0.5f);
    EXPECT_FLOAT_EQ(result.y, 1.0f);
    EXPECT_FLOAT_EQ(result.z, 1.5f);
}

TEST_F(VectorTest, VectorAdditionAssignment)
{
    vec3f v = v1;
    v += v2;

    EXPECT_EQ(v.x, 5.0f);
    EXPECT_EQ(v.y, 7.0f);
    EXPECT_EQ(v.z, 9.0f);
}

TEST_F(VectorTest, VectorSubtractionAssignment)
{
    vec3f v = v2;
    v -= v1;

    EXPECT_EQ(v.x, 3.0f);
    EXPECT_EQ(v.y, 3.0f);
    EXPECT_EQ(v.z, 3.0f);
}

TEST_F(VectorTest, ScalarMultiplicationAssignment)
{
    vec3f v = v1;
    v *= 2.0f;

    EXPECT_EQ(v.x, 2.0f);
    EXPECT_EQ(v.y, 4.0f);
    EXPECT_EQ(v.z, 6.0f);
}

TEST_F(VectorTest, ScalarDivisionAssignment)
{
    vec3f v = v1;
    v /= 2.0f;

    EXPECT_FLOAT_EQ(v.x, 0.5f);
    EXPECT_FLOAT_EQ(v.y, 1.0f);
    EXPECT_FLOAT_EQ(v.z, 1.5f);
}