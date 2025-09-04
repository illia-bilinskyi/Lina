#include "lina/lina.h"
#include <gtest/gtest.h>

using namespace lina;

class MatrixTest : public ::testing::Test
{
  protected:
    static constexpr mat2f mat2f_{
        { 1.0f, 2.0f },
        { 3.0f, 4.0f }
    };
    static constexpr mat3f mat3f_{
        { 1.0f, 2.0f, 3.0f },
        { 4.0f, 5.0f, 6.0f },
        { 7.0f, 8.0f, 9.0f }
    };
    static constexpr mat4f mat4f_{
        {  1.0f,  2.0f,  3.0f,  4.0f },
        {  5.0f,  6.0f,  7.0f,  8.0f },
        {  9.0f, 10.0f, 11.0f, 12.0f },
        { 13.0f, 14.0f, 15.0f, 16.0f }
    };
};

TEST_F(MatrixTest, DefaultConstructor)
{
    constexpr mat2f m;
    for (size_t i = 0; i < mat2f::size; ++i)
    {
        EXPECT_EQ(m.data()[i], 0.0f);
    }
}

TEST_F(MatrixTest, FillConstructor)
{
    constexpr mat2f m(5.0f);
    for (size_t i = 0; i < mat2f::size; ++i)
    {
        EXPECT_EQ(m.data()[i], 5.0f);
    }
}

TEST_F(MatrixTest, VariadicTemplateConstructor)
{
    constexpr mat2f m(1.0f, 2.0f, 3.0f, 4.0f);

    EXPECT_EQ(m(0, 0), 1.0f);
    EXPECT_EQ(m(0, 1), 2.0f);
    EXPECT_EQ(m(1, 0), 3.0f);
    EXPECT_EQ(m(1, 1), 4.0f);
}

TEST_F(MatrixTest, InitializerListConstructor)
{
    constexpr mat2f m = mat2f_;
    EXPECT_EQ(m(0, 0), 1.0f);
    EXPECT_EQ(m(0, 1), 2.0f);
    EXPECT_EQ(m(1, 0), 3.0f);
    EXPECT_EQ(m(1, 1), 4.0f);
}

TEST(MatrixRowConstructorTest, InitializerListOfRowMatrices)
{
    // Test 2x3 matrix construction using row matrices
    constexpr mat<float, 2, 3> M{
        { 1.0f, 2.0f, 3.0f },
        { 4.0f, 5.0f, 6.0f }
    };

    // Verify all elements are correctly placed
    EXPECT_EQ(M(0, 0), 1.0f);
    EXPECT_EQ(M(0, 1), 2.0f);
    EXPECT_EQ(M(0, 2), 3.0f);
    EXPECT_EQ(M(1, 0), 4.0f);
    EXPECT_EQ(M(1, 1), 5.0f);
    EXPECT_EQ(M(1, 2), 6.0f);

    // Test that the matrix has correct dimensions
    EXPECT_EQ(M.rows, 2);
    EXPECT_EQ(M.cols, 3);

    // Test constexpr compatibility
    constexpr mat<float, 2, 2> constM{
        { 1.0f, 2.0f },
        { 3.0f, 4.0f }
    };

    static_assert(constM(0, 0) == 1.0f);
    static_assert(constM(1, 1) == 4.0f);

    // Test incomplete number of rows
    constexpr mat<float, 2, 3> M2{
        { 1.0f, 2.0f, 3.0f }
    };
    EXPECT_EQ(M2(0, 0), 1.0f);
    EXPECT_EQ(M2(0, 1), 2.0f);
    EXPECT_EQ(M2(0, 2), 3.0f);
    EXPECT_EQ(M2(1, 0), 0.0f);
    EXPECT_EQ(M2(1, 1), 0.0f);
    EXPECT_EQ(M2(1, 2), 0.0f);
}

TEST_F(MatrixTest, CopyConstructor)
{
    constexpr mat<float, 2, 3> original{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 9.0f };
    constexpr decltype(original) copy(original);

    for (size_t i = 0; i < mat2f::size; ++i)
    {
        EXPECT_EQ(original.data()[i], copy.data()[i]);
    }
}

TEST_F(MatrixTest, CopyConstructorSquareMatrix)
{
    constexpr mat2f original = mat2f_;
    constexpr mat3f copy(original);

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
        {
            if (i < 2 && j < 2)
                EXPECT_EQ(original(i, j), copy(i, j));
            else
                EXPECT_EQ(0, copy(i, j));
        }
}

TEST_F(MatrixTest, MatrixValueTypeConversion)
{
    constexpr mat2d m = static_cast<mat2d>(mat2f_);

    constexpr decltype(m) expected{ 1.0, 2.0, 3.0, 4.0 };
    EXPECT_TRUE(almost_equal(m, expected));
}

TEST_F(MatrixTest, GetTemplated)
{
    constexpr mat2f m  = mat2f_;
    constexpr auto a00 = m.get<0, 0>();
    constexpr auto a10 = m.get<1, 0>();
    constexpr auto a01 = m.get<0, 1>();
    constexpr auto a11 = m.get<1, 1>();
    EXPECT_EQ(a00, mat2f_(0, 0));
    EXPECT_EQ(a10, mat2f_(1, 0));
    EXPECT_EQ(a01, mat2f_(0, 1));
    EXPECT_EQ(a11, mat2f_(1, 1));
}

TEST_F(MatrixTest, DataAccess)
{
    constexpr mat2f m = mat2f_;
    const float* data = m.data();
    EXPECT_EQ(data[0], 1.0f);
    EXPECT_EQ(data[1], 2.0f);
    EXPECT_EQ(data[2], 3.0f);
    EXPECT_EQ(data[3], 4.0f);
}

TEST_F(MatrixTest, FunctionCallOperator)
{
    constexpr mat2f m = mat2f_;
    EXPECT_EQ(m(0, 0), 1.0f);
    EXPECT_EQ(m(0, 1), 2.0f);
    EXPECT_EQ(m(1, 0), 3.0f);
    EXPECT_EQ(m(1, 1), 4.0f);
}

TEST_F(MatrixTest, ColumnExtraction)
{
    constexpr mat3f m   = mat3f_;
    constexpr auto col0 = m.col<0>();
    EXPECT_EQ(col0(0, 0), 1.0f);
    EXPECT_EQ(col0(1, 0), 4.0f);
    EXPECT_EQ(col0(2, 0), 7.0f);
}

TEST_F(MatrixTest, RowExtraction)
{
    constexpr mat3f m   = mat3f_;
    constexpr auto row0 = m.row<0>();
    EXPECT_EQ(row0(0, 0), 1.0f);
    EXPECT_EQ(row0(0, 1), 2.0f);
    EXPECT_EQ(row0(0, 2), 3.0f);
}

TEST_F(MatrixTest, UnaryPlus)
{
    constexpr mat2f m     = mat2f_;
    constexpr auto result = +m;
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(m.data()[i], result.data()[i]);
    }
}

TEST_F(MatrixTest, UnaryMinus)
{
    constexpr mat2f m     = mat2f_;
    constexpr auto result = -m;

    constexpr decltype(m) expected{ -1.0f, -2.0f, -3.0f, -4.0f };
    EXPECT_TRUE(almost_equal(result, expected));
    EXPECT_TRUE(almost_equal(m, mat2f_));
}

TEST_F(MatrixTest, ScalarMultiplication)
{
    constexpr mat2f m     = mat2f_;
    constexpr auto result = m * 2.0f;

    constexpr decltype(m) expected{ 2.0f, 4.0f, 6.0f, 8.0f };
    EXPECT_TRUE(almost_equal(result, expected));
}

TEST_F(MatrixTest, ScalarDivision)
{
    constexpr mat2f m{ 2.0f, 4.0f, 6.0f, 8.0f };
    constexpr auto result = m / 2.0f;

    constexpr decltype(m) expected{ 1.0f, 2.0f, 3.0f, 4.0f };
    EXPECT_TRUE(almost_equal(result, expected));
}

TEST_F(MatrixTest, ScalarMultiplicationAssignment)
{
    mat2f m{ 1.0f, 2.0f, 3.0f, 4.0f };
    m *= 2.0f;

    constexpr decltype(m) expected{ 2.0f, 4.0f, 6.0f, 8.0f };
    EXPECT_TRUE(almost_equal(m, expected));
}

TEST_F(MatrixTest, ScalarDivisionAssignment)
{
    mat2f m{ 2.0f, 4.0f, 6.0f, 8.0f };
    m /= 2.0f;

    constexpr decltype(m) expected{ 1.0f, 2.0f, 3.0f, 4.0f };
    EXPECT_TRUE(almost_equal(m, expected));
}

TEST_F(MatrixTest, MatrixAddition)
{
    constexpr mat2f a{ 1.0f, 2.0f, 3.0f, 4.0f };
    constexpr mat2f b{ 5.0f, 6.0f, 7.0f, 8.0f };
    constexpr auto result = a + b;

    constexpr decltype(result) expected{ 6.0f, 8.0f, 10.0f, 12.0f };
    EXPECT_TRUE(almost_equal(result, expected));
}

TEST_F(MatrixTest, MatrixSubtraction)
{
    constexpr mat2f a{ 5.0f, 6.0f, 7.0f, 8.0f };
    constexpr mat2f b{ 1.0f, 2.0f, 3.0f, 4.0f };
    constexpr auto result = a - b;

    constexpr decltype(result) expected(4.0f);
    EXPECT_TRUE(almost_equal(result, expected));
}

TEST_F(MatrixTest, MatrixAdditionAssignment)
{
    mat2f a{ 1.0f, 2.0f, 3.0f, 4.0f };
    mat2f b{ 5.0f, 6.0f, 7.0f, 8.0f };
    a += b;

    constexpr decltype(a) expected{ 6.0f, 8.0f, 10.0f, 12.0f };
    EXPECT_TRUE(almost_equal(a, expected));
}

TEST_F(MatrixTest, MatrixSubtractionAssignment)
{
    mat2f a{ 5.0f, 6.0f, 7.0f, 8.0f };
    mat2f b{ 1.0f, 2.0f, 3.0f, 4.0f };
    a -= b;

    constexpr decltype(a) expected(4.0f);
    EXPECT_TRUE(almost_equal(a, expected));
}

TEST_F(MatrixTest, MatrixMultiplication)
{
    constexpr mat2f a{ 1.0f, 2.0f, 3.0f, 4.0f };
    constexpr mat2f b{ 5.0f, 6.0f, 7.0f, 8.0f };
    constexpr auto result = a * b;

    constexpr decltype(result) expected{ 19.0f, 22.0f, 43.0f, 50.0f };
    EXPECT_TRUE(almost_equal(result, expected));

    constexpr mat<float, 4, 2> A{ 8, 7, 6, 5, 4, 3, 2, 1 };
    constexpr mat<float, 2, 3> B{ 3, 4, 5, 6, 7, 8 };

    constexpr mat<float, 4, 3> result2 = A * B;
    constexpr decltype(result2) expected2{ 66, 81, 96, 48, 59, 70, 30, 37, 44, 12, 15, 18 };
    EXPECT_TRUE(almost_equal(result2, expected2));
}

// Test the assignment capability that was missing
TEST_F(MatrixTest, ElementAssignment)
{
    mat2f m;
    m(0, 0) = 1.0f;
    m(0, 1) = 2.0f;
    m(1, 0) = 3.0f;
    m(1, 1) = 4.0f;

    constexpr decltype(m) expected{ 1.0f, 2.0f, 3.0f, 4.0f };
    EXPECT_TRUE(almost_equal(m, expected));
}