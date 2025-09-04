#include "lina/lina.h"
#include <gtest/gtest.h>

using namespace lina;

class MatrixOperationsTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        mat2f_identity = { 1.0f, 0.0f, 0.0f, 1.0f };
        mat2f_regular  = { 1.0f, 2.0f, 3.0f, 4.0f };
        mat3f_identity = {
            { 1.0f, 0.0f, 0.0f },
            { 0.0f, 1.0f, 0.0f },
            { 0.0f, 0.0f, 1.0f }
        };
        mat3f_regular = {
            { 1.0f, 2.0f, 3.0f },
            { 4.0f, 5.0f, 6.0f },
            { 7.0f, 8.0f, 9.0f }
        };

        // Invertible 3x3 matrix
        mat3f_invertible = { 1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f };
    }

    mat2<float> mat2f_identity, mat2f_regular;
    mat3<float> mat3f_identity, mat3f_regular, mat3f_invertible;
    mat4<float> mat4f_identity;
};

TEST_F(MatrixOperationsTest, AlmostEqual_Scalar)
{
    EXPECT_TRUE(almost_equal(1.0f, 1.000001f, 1e-5f));
    EXPECT_FALSE(almost_equal(1.0f, 1.1f, 1e-5f));
    EXPECT_TRUE(almost_equal(0.0f, 0.0f));
}

TEST_F(MatrixOperationsTest, AlmostEqual_Matrix)
{
    mat2<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
    mat2<float> b{ 1.000001f, 2.000001f, 3.000001f, 4.000001f };

    EXPECT_TRUE(almost_equal(a, b, 1e-5f));
    EXPECT_FALSE(almost_equal(a, b, 1e-8f));
}

TEST_F(MatrixOperationsTest, AlmostZero)
{
    EXPECT_TRUE(almost_zero(0.0f));
    EXPECT_TRUE(almost_zero(1e-7f));
    EXPECT_FALSE(almost_zero(0.1f));
}

TEST_F(MatrixOperationsTest, Identity_2x2)
{
    auto I = identity<float, 2>();

    EXPECT_EQ(I(0, 0), 1.0f);
    EXPECT_EQ(I(0, 1), 0.0f);
    EXPECT_EQ(I(1, 0), 0.0f);
    EXPECT_EQ(I(1, 1), 1.0f);
}

TEST_F(MatrixOperationsTest, Identity_3x3)
{
    auto I = identity<float, 3>();

    EXPECT_EQ(I(0, 0), 1.0f);
    EXPECT_EQ(I(1, 1), 1.0f);
    EXPECT_EQ(I(2, 2), 1.0f);

    // Check off-diagonal elements are zero
    EXPECT_EQ(I(0, 1), 0.0f);
    EXPECT_EQ(I(1, 0), 0.0f);
}

TEST_F(MatrixOperationsTest, Identity_4x4)
{
    auto I = identity<float, 4>();

    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(I(i, i), 1.0f);
        for (size_t j = 0; j < 4; ++j)
        {
            if (i != j)
            {
                EXPECT_EQ(I(i, j), 0.0f);
            }
        }
    }
}

TEST_F(MatrixOperationsTest, Transpose_2x2)
{
    mat2<float> m{ 1.0f, 2.0f, 3.0f, 4.0f };
    auto result = transpose(m);

    EXPECT_EQ(result(0, 0), 1.0f); // m(0,0)
    EXPECT_EQ(result(1, 0), 2.0f); // m(0,1)
    EXPECT_EQ(result(0, 1), 3.0f); // m(1,0)
    EXPECT_EQ(result(1, 1), 4.0f); // m(1,1)
}

TEST_F(MatrixOperationsTest, Transpose_3x3)
{
    auto result = transpose(mat3f_regular);

    EXPECT_EQ(result(0, 0), 1.0f); // m(0,0)
    EXPECT_EQ(result(1, 0), 2.0f); // m(0,1)
    EXPECT_EQ(result(2, 0), 3.0f); // m(0,2)
    EXPECT_EQ(result(0, 1), 4.0f); // m(1,0)
}

TEST_F(MatrixOperationsTest, Transpose_RectangularMatrix)
{
    mat<float, 2, 3> m{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    auto result = transpose(m);

    static_assert(std::is_same_v<decltype(result), mat<float, 3, 2>>);

    EXPECT_EQ(result(0, 0), 1.0f);
    EXPECT_EQ(result(1, 0), 2.0f);
    EXPECT_EQ(result(2, 0), 3.0f);
    EXPECT_EQ(result(0, 1), 4.0f);
    EXPECT_EQ(result(1, 1), 5.0f);
    EXPECT_EQ(result(2, 1), 6.0f);
}

TEST_F(MatrixOperationsTest, Determinant_2x2)
{
    mat2<float> m{ 1.0f, 2.0f, 3.0f, 4.0f };
    float result = det(m);

    // det = 1*4 - 2*3 = -2
    EXPECT_FLOAT_EQ(result, -2.0f);
}

TEST_F(MatrixOperationsTest, Determinant_2x2_Identity)
{
    float result = det(mat2f_identity);
    EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST_F(MatrixOperationsTest, Determinant_3x3_Identity)
{
    float result = det(mat3f_identity);
    EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST_F(MatrixOperationsTest, Determinant_3x3_Zero)
{
    float result = det(mat3f_regular); // This matrix is singular
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST_F(MatrixOperationsTest, Determinant_3x3_Diagonal)
{
    float result = det(mat3f_invertible);
    EXPECT_FLOAT_EQ(result, 6.0f); // 1*2*3 = 6
}

TEST_F(MatrixOperationsTest, Determinant_4x4_Identity)
{
    auto identity_4x4 = identity<float, 4>();
    float result      = det(identity_4x4);
    EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST_F(MatrixOperationsTest, Inverse_2x2_Identity)
{
    auto result = inverse(mat2f_identity);

    EXPECT_FLOAT_EQ(result(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 0.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 1.0f);
}

TEST_F(MatrixOperationsTest, Inverse_2x2_Regular)
{
    mat2<float> m{ 2.0f, 0.0f, 0.0f, 2.0f }; // 2I
    auto result = inverse(m);

    EXPECT_FLOAT_EQ(result(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(result(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 0.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 0.5f);
}

TEST_F(MatrixOperationsTest, ConstexprInverse_2x2)
{
    constexpr auto I      = identity<float, 2>();
    constexpr auto result = c_inverse(I);

    static_assert(result.get<0, 0>() == 1.0f);
    static_assert(result.get<1, 1>() == 1.0f);
    static_assert(result.get<0, 1>() == 0.0f);
    static_assert(result.get<1, 0>() == 0.0f);
}

TEST_F(MatrixOperationsTest, ConstexprInverse_3x3)
{
    constexpr auto I      = identity<float, 3>();
    constexpr auto result = c_inverse(I);

    static_assert(result.get<0, 0>() == 1.0f);
    static_assert(result.get<1, 1>() == 1.0f);
    static_assert(result.get<2, 2>() == 1.0f);
}

TEST_F(MatrixOperationsTest, Inverse_3x3_Identity)
{
    auto result = inverse(mat3f_identity);

    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            EXPECT_FLOAT_EQ(result(i, j), mat3f_identity(i, j));
        }
    }
}

TEST_F(MatrixOperationsTest, Inverse_3x3_Diagonal)
{
    auto result = inverse(mat3f_invertible);

    EXPECT_FLOAT_EQ(result(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 0.5f);        // 1/2
    EXPECT_FLOAT_EQ(result(2, 2), 1.0f / 3.0f); // 1/3

    // Off-diagonal should be zero
    EXPECT_FLOAT_EQ(result(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 0.0f);
}

TEST_F(MatrixOperationsTest, Inverse_SingularMatrix_ThrowsException)
{
    mat2<float> singular{ 1.0f, 2.0f, 2.0f, 4.0f }; // det = 0

    EXPECT_THROW(inverse(singular), std::invalid_argument);
}

TEST_F(MatrixOperationsTest, Inverse_3x3_SingularMatrix_ThrowsException)
{
    EXPECT_THROW(inverse(mat3f_regular), std::invalid_argument);
}