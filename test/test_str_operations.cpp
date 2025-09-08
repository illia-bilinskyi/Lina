#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include "lina/lina.h" // Assuming the header file is named lina.h

using namespace lina;

// Test fixture for matrix stream operators
class MatStreamOperatorTest : public ::testing::Test {
protected:
    std::ostringstream oss;
    
    void SetUp() override {
        oss.str("");
        oss.clear();
    }
};

// Test fixture for vec3 stream operators
class Vec3StreamOperatorTest : public ::testing::Test {
protected:
    std::ostringstream oss;
    
    void SetUp() override {
        oss.str("");
        oss.clear();
    }
};

// Matrix Stream Operator Tests

TEST_F(MatStreamOperatorTest, Mat2x2IntegerValues) {
    mat<int, 2, 2> m{1, 2, 3, 4};
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 1, 2 } { 3, 4 } }");
}

TEST_F(MatStreamOperatorTest, Mat2x2FloatValues) {
    mat<float, 2, 2> m{1.5f, 2.5f, 3.5f, 4.5f};
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 1.5, 2.5 } { 3.5, 4.5 } }");
}

TEST_F(MatStreamOperatorTest, Mat3x3IntegerValues) {
    mat<int, 3, 3> m{1, 2, 3, 4, 5, 6, 7, 8, 9};
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 1, 2, 3 } { 4, 5, 6 } { 7, 8, 9 } }");
}

TEST_F(MatStreamOperatorTest, Mat4x4IntegerValues) {
    mat<int, 4, 4> m{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 1, 2, 3, 4 } { 5, 6, 7, 8 } { 9, 10, 11, 12 } { 13, 14, 15, 16 } }");
}

TEST_F(MatStreamOperatorTest, Mat1x3RowVector) {
    mat<int, 1, 3> m{1, 2, 3};
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 1, 2, 3 } }");
}

TEST_F(MatStreamOperatorTest, Mat3x1ColumnVector) {
    mat<int, 3, 1> m{1, 2, 3};
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 1 } { 2 } { 3 } }");
}

TEST_F(MatStreamOperatorTest, Mat1x1SingleElement) {
    mat<int, 1, 1> m{42};
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 42 } }");
}

TEST_F(MatStreamOperatorTest, Mat2x2ZeroMatrix) {
    mat<int, 2, 2> m{}; // Default initialized to zeros
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 0, 0 } { 0, 0 } }");
}

TEST_F(MatStreamOperatorTest, Mat2x2NegativeValues) {
    mat<int, 2, 2> m{-1, -2, -3, -4};
    oss << m;
    EXPECT_EQ(oss.str(), "{ { -1, -2 } { -3, -4 } }");
}

TEST_F(MatStreamOperatorTest, Mat2x2DoubleValues) {
    mat<double, 2, 2> m{1.123, 2.456, 3.789, 4.012};
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 1.123, 2.456 } { 3.789, 4.012 } }");
}

TEST_F(MatStreamOperatorTest, Mat3x2RectangularMatrix) {
    mat<int, 3, 2> m{1, 2, 3, 4, 5, 6};
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 1, 2 } { 3, 4 } { 5, 6 } }");
}

TEST_F(MatStreamOperatorTest, Mat2x3RectangularMatrix) {
    mat<int, 2, 3> m{1, 2, 3, 4, 5, 6};
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 1, 2, 3 } { 4, 5, 6 } }");
}

// Vec3 Stream Operator Tests

TEST_F(Vec3StreamOperatorTest, Vec3IntegerValues) {
    vec3<int> v{1, 2, 3};
    oss << v;
    EXPECT_EQ(oss.str(), "{ 1, 2, 3 }");
}

TEST_F(Vec3StreamOperatorTest, Vec3FloatValues) {
    vec3<float> v{1.5f, 2.5f, 3.5f};
    oss << v;
    EXPECT_EQ(oss.str(), "{ 1.5, 2.5, 3.5 }");
}

TEST_F(Vec3StreamOperatorTest, Vec3DoubleValues) {
    vec3<double> v{1.123, 2.456, 3.789};
    oss << v;
    EXPECT_EQ(oss.str(), "{ 1.123, 2.456, 3.789 }");
}

TEST_F(Vec3StreamOperatorTest, Vec3ZeroVector) {
    vec3<int> v{}; // Default initialized to zeros
    oss << v;
    EXPECT_EQ(oss.str(), "{ 0, 0, 0 }");
}

TEST_F(Vec3StreamOperatorTest, Vec3NegativeValues) {
    vec3<int> v{-1, -2, -3};
    oss << v;
    EXPECT_EQ(oss.str(), "{ -1, -2, -3 }");
}

TEST_F(Vec3StreamOperatorTest, Vec3MixedSignValues) {
    vec3<int> v{-1, 2, -3};
    oss << v;
    EXPECT_EQ(oss.str(), "{ -1, 2, -3 }");
}

TEST_F(Vec3StreamOperatorTest, Vec3LargeValues) {
    vec3<int> v{1000000, 2000000, 3000000};
    oss << v;
    EXPECT_EQ(oss.str(), "{ 1000000, 2000000, 3000000 }");
}

TEST_F(Vec3StreamOperatorTest, Vec3SmallDoubleValues) {
    vec3<double> v{0.001, 0.002, 0.003};
    oss << v;
    EXPECT_EQ(oss.str(), "{ 0.001, 0.002, 0.003 }");
}

// Vec3 str() Method Tests

TEST(Vec3StrMethodTest, Vec3StrIntegerValues) {
    vec3<int> v{1, 2, 3};
    EXPECT_EQ(v.str(), "{ 1, 2, 3 }");
}

TEST(Vec3StrMethodTest, Vec3StrFloatValues) {
    vec3<float> v{1.5f, 2.5f, 3.5f};
    EXPECT_EQ(v.str(), "{ 1.5, 2.5, 3.5 }");
}

TEST(Vec3StrMethodTest, Vec3StrDoubleValues) {
    vec3<double> v{1.123, 2.456, 3.789};
    EXPECT_EQ(v.str(), "{ 1.123, 2.456, 3.789 }");
}

TEST(Vec3StrMethodTest, Vec3StrZeroVector) {
    vec3<int> v{};
    EXPECT_EQ(v.str(), "{ 0, 0, 0 }");
}

TEST(Vec3StrMethodTest, Vec3StrNegativeValues) {
    vec3<int> v{-1, -2, -3};
    EXPECT_EQ(v.str(), "{ -1, -2, -3 }");
}

TEST(Vec3StrMethodTest, Vec3StrMixedSignValues) {
    vec3<int> v{-1, 2, -3};
    EXPECT_EQ(v.str(), "{ -1, 2, -3 }");
}

TEST(Vec3StrMethodTest, Vec3StrConsistencyWithStreamOperator) {
    vec3<double> v{42.42, -13.13, 99.99};
    std::ostringstream oss;
    oss << v;
    EXPECT_EQ(v.str(), oss.str());
}

// Additional edge case tests

TEST_F(Vec3StreamOperatorTest, Vec3ScientificNotation) {
    vec3<double> v{1e-10, 1e10, 1e-5};
    oss << v;
    // Note: The exact format depends on the compiler and locale settings
    // This test verifies that the output is well-formed
    std::string result = oss.str();
    EXPECT_TRUE(result.find("{ ") == 0);
    EXPECT_TRUE(result.find(" }") == result.length() - 2);
    EXPECT_TRUE(result.find(", ") != std::string::npos);
}

TEST(Vec3StrMethodTest, Vec3StrScientificNotation) {
    vec3<double> v{1e-10, 1e10, 1e-5};
    std::string result = v.str();
    EXPECT_TRUE(result.find("{ ") == 0);
    EXPECT_TRUE(result.find(" }") == result.length() - 2);
    EXPECT_TRUE(result.find(", ") != std::string::npos);
}

TEST_F(MatStreamOperatorTest, MatIdentityMatrix) {
    auto m = identity<int, 3>();
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 1, 0, 0 } { 0, 1, 0 } { 0, 0, 1 } }");
}

TEST_F(MatStreamOperatorTest, MatFillConstructor) {
    mat<int, 2, 2> m{42}; // Fill with 42
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 42, 42 } { 42, 42 } }");
}



TEST_F(MatStreamOperatorTest, MatHighPrecision) {
    mat<double, 2, 2> m{}; // Fill with 42
    m(0, 0) = 0.0000001;
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 1e-07, 0 } { 0, 0 } }");
}

// Test with type aliases
TEST_F(Vec3StreamOperatorTest, Vec3fAlias) {
    vec3f v{1.0f, 2.0f, 3.0f};
    oss << v;
    EXPECT_EQ(oss.str(), "{ 1, 2, 3 }");
}

TEST_F(Vec3StreamOperatorTest, Vec3dAlias) {
    vec3d v{1.0, 2.0, 3.0};
    oss << v;
    EXPECT_EQ(oss.str(), "{ 1, 2, 3 }");
}

TEST_F(MatStreamOperatorTest, Mat2fAlias) {
    mat2f m{1.0f, 2.0f, 3.0f, 4.0f};
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 1, 2 } { 3, 4 } }");
}

TEST_F(MatStreamOperatorTest, Mat3dAlias) {
    mat3d m{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 1, 2, 3 } { 4, 5, 6 } { 7, 8, 9 } }");
}

TEST_F(MatStreamOperatorTest, Mat4fAlias) {
    mat4f m{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    oss << m;
    EXPECT_EQ(oss.str(), "{ { 1, 2, 3, 4 } { 5, 6, 7, 8 } { 9, 10, 11, 12 } { 13, 14, 15, 16 } }");
}