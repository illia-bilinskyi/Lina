#include "lina.h"
#include <gtest/gtest.h>

using namespace lina;

class VectorMatrixOperationsTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Test vectors
        v1     = vec3f{1.0f, 2.0f, 3.0f};
        v2     = vec3f{4.0f, 5.0f, 6.0f};
        unit_x = vec3f{1.0f, 0.0f, 0.0f};
        unit_y = vec3f{0.0f, 1.0f, 0.0f};
        unit_z = vec3f{0.0f, 0.0f, 1.0f};

        // Test matrices
        identity_3x3 = identity<float, 3>();
        identity_4x4 = identity<float, 4>();

        // Scale matrices
        scale_2x  = scale(vec3f{2.0f, 2.0f, 2.0f});
        scale_xyz = scale(vec3f{1.0f, 2.0f, 3.0f});

        // Translation matrix
        translate_123 = translation(vec3f{1.0f, 2.0f, 3.0f});
    }

    vec3f v1, v2, unit_x, unit_y, unit_z;
    mat3f identity_3x3;
    mat4f identity_4x4, scale_2x, scale_xyz, translate_123;
};

// ===== Homogeneous Coordinate Tests =====

TEST_F(VectorMatrixOperationsTest, ToHomogeneous_DefaultW)
{
    mat<float, 1, 4> homogeneous = to_homogeneous(v1);

    EXPECT_EQ(homogeneous(0, 0), 1.0f);
    EXPECT_EQ(homogeneous(0, 1), 2.0f);
    EXPECT_EQ(homogeneous(0, 2), 3.0f);
    EXPECT_EQ(homogeneous(0, 3), 1.0f); // Default w = 1
}

TEST_F(VectorMatrixOperationsTest, ToHomogeneous_CustomW)
{
    mat<float, 1, 4> homogeneous = to_homogeneous(v1, 2.5f);

    EXPECT_EQ(homogeneous(0, 0), 1.0f);
    EXPECT_EQ(homogeneous(0, 1), 2.0f);
    EXPECT_EQ(homogeneous(0, 2), 3.0f);
    EXPECT_EQ(homogeneous(0, 3), 2.5f);
}

TEST_F(VectorMatrixOperationsTest, FromHomogeneous_UnitW)
{
    mat<float, 1, 4> homogeneous{2.0f, 4.0f, 6.0f, 1.0f};
    vec3f result = from_homogeneous(homogeneous);

    EXPECT_EQ(result.x, 2.0f);
    EXPECT_EQ(result.y, 4.0f);
    EXPECT_EQ(result.z, 6.0f);
}

TEST_F(VectorMatrixOperationsTest, FromHomogeneous_NonUnitW)
{
    mat<float, 1, 4> homogeneous{4.0f, 8.0f, 12.0f, 2.0f};
    vec3f result = from_homogeneous(homogeneous);

    EXPECT_EQ(result.x, 2.0f); // 4/2
    EXPECT_EQ(result.y, 4.0f); // 8/2
    EXPECT_EQ(result.z, 6.0f); // 12/2
}

TEST_F(VectorMatrixOperationsTest, HomogeneousRoundTrip)
{
    mat<float, 1, 4> homogeneous = to_homogeneous(v1, 3.0f);
    vec3f result                 = from_homogeneous(homogeneous);

    // Should get back original vector divided by w
    EXPECT_NEAR(result.x, v1.x / 3.0f, 1e-6f);
    EXPECT_NEAR(result.y, v1.y / 3.0f, 1e-6f);
    EXPECT_NEAR(result.z, v1.z / 3.0f, 1e-6f);
}

// ===== Matrix3 × Vector3 Tests =====

TEST_F(VectorMatrixOperationsTest, Matrix3_Vector_Identity)
{
    vec3f result = identity_3x3 * v1;

    EXPECT_EQ(result.x, v1.x);
    EXPECT_EQ(result.y, v1.y);
    EXPECT_EQ(result.z, v1.z);
}

TEST_F(VectorMatrixOperationsTest, Matrix3_Vector_Scale)
{
    mat3f scale_matrix{2.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 2.0f};
    vec3f result = scale_matrix * v1;

    EXPECT_EQ(result.x, 2.0f * v1.x);
    EXPECT_EQ(result.y, 2.0f * v1.y);
    EXPECT_EQ(result.z, 2.0f * v1.z);
}

TEST_F(VectorMatrixOperationsTest, Matrix3_Vector_Rotation90Z)
{
    // 90-degree rotation around Z-axis
    mat3f rot_z{0.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    vec3f result = rot_z * unit_x;

    EXPECT_NEAR(result.x, 0.0f, 1e-6f);
    EXPECT_NEAR(result.y, 1.0f, 1e-6f); // X becomes Y
    EXPECT_NEAR(result.z, 0.0f, 1e-6f);
}

TEST_F(VectorMatrixOperationsTest, Matrix3_Vector_GeneralTransform)
{
    mat3f transform{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    vec3f result = transform * vec3f{1.0f, 1.0f, 1.0f};

    // Result should be sum of rows: [1+2+3, 4+5+6, 7+8+9]
    EXPECT_EQ(result.x, 6.0f);
    EXPECT_EQ(result.y, 15.0f);
    EXPECT_EQ(result.z, 24.0f);
}

// ===== Matrix4 × Vector3 Tests =====

TEST_F(VectorMatrixOperationsTest, Matrix4_Vector_Identity)
{
    vec3f result = identity_4x4 * v1;

    EXPECT_EQ(result.x, v1.x);
    EXPECT_EQ(result.y, v1.y);
    EXPECT_EQ(result.z, v1.z);
}

TEST_F(VectorMatrixOperationsTest, Matrix4_Vector_Translation)
{
    vec3f result = translate_123 * v1;

    EXPECT_EQ(result.x, v1.x + 1.0f);
    EXPECT_EQ(result.y, v1.y + 2.0f);
    EXPECT_EQ(result.z, v1.z + 3.0f);
}

TEST_F(VectorMatrixOperationsTest, Matrix4_Vector_UniformScale)
{
    vec3f result = scale_2x * v1;

    EXPECT_EQ(result.x, 2.0f * v1.x);
    EXPECT_EQ(result.y, 2.0f * v1.y);
    EXPECT_EQ(result.z, 2.0f * v1.z);
}

TEST_F(VectorMatrixOperationsTest, Matrix4_Vector_NonUniformScale)
{
    vec3f result = scale_xyz * v1;

    EXPECT_EQ(result.x, 1.0f * v1.x); // Scale by 1
    EXPECT_EQ(result.y, 2.0f * v1.y); // Scale by 2
    EXPECT_EQ(result.z, 3.0f * v1.z); // Scale by 3
}

TEST_F(VectorMatrixOperationsTest, Matrix4_Vector_CompositeTransform)
{
    mat4f composite = translate_123 * scale_2x;
    vec3f result    = composite * v1;

    // Should scale first, then translate
    EXPECT_EQ(result.x, 2.0f * v1.x + 1.0f);
    EXPECT_EQ(result.y, 2.0f * v1.y + 2.0f);
    EXPECT_EQ(result.z, 2.0f * v1.z + 3.0f);
}

TEST_F(VectorMatrixOperationsTest, Matrix4_Vector_ZeroWHandling)
{
    // Create a matrix that produces w=0 (degenerate case)
    mat4f degenerate_matrix{};
    // Set only the spatial components, leave w row as zero
    degenerate_matrix(0, 0) = 1.0f;
    degenerate_matrix(1, 1) = 1.0f;
    degenerate_matrix(2, 2) = 1.0f;
    // w component will be 0

    vec3f result = degenerate_matrix * v1;

    // Should return zero vector when w is zero
    EXPECT_EQ(result.x, 0.0f);
    EXPECT_EQ(result.y, 0.0f);
    EXPECT_EQ(result.z, 0.0f);
}

// ===== Vector3 × Matrix3 Tests =====

TEST_F(VectorMatrixOperationsTest, Vector_Matrix3_Identity)
{
    vec3f result = v1 * identity_3x3;

    EXPECT_EQ(result.x, v1.x);
    EXPECT_EQ(result.y, v1.y);
    EXPECT_EQ(result.z, v1.z);
}

TEST_F(VectorMatrixOperationsTest, Vector_Matrix3_ColumnOperation)
{
    mat3f transform{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    vec3f result = vec3f{1.0f, 1.0f, 1.0f} * transform;

    // Result should be sum of columns: [1+4+7, 2+5+8, 3+6+9]
    EXPECT_EQ(result.x, 12.0f);
    EXPECT_EQ(result.y, 15.0f);
    EXPECT_EQ(result.z, 18.0f);
}

// ===== Vector3 × Matrix4 Tests =====

TEST_F(VectorMatrixOperationsTest, Vector_Matrix4_Identity)
{
    vec3f result = identity_4x4 * v1;

    EXPECT_EQ(result.x, v1.x);
    EXPECT_EQ(result.y, v1.y);
    EXPECT_EQ(result.z, v1.z);
}

TEST_F(VectorMatrixOperationsTest, Vector_Matrix4_Translation)
{
    vec3f result = translate_123 * v1;

    EXPECT_EQ(result.x, v1.x + 1.0f);
    EXPECT_EQ(result.y, v1.y + 2.0f);
    EXPECT_EQ(result.z, v1.z + 3.0f);
}

TEST_F(VectorMatrixOperationsTest, Vector_Matrix4_ZeroWHandling)
{
    // Create a matrix that produces w=0
    mat4f degenerate_matrix{};
    degenerate_matrix(0, 0) = 1.0f;
    degenerate_matrix(1, 1) = 1.0f;
    degenerate_matrix(2, 2) = 1.0f;
    // Fourth row remains zero

    vec3f result = degenerate_matrix * v1;

    // Should return zero vector when w is zero
    EXPECT_EQ(result.x, 0.0f);
    EXPECT_EQ(result.y, 0.0f);
    EXPECT_EQ(result.z, 0.0f);
}

// ===== Cross-Operation Consistency Tests =====

TEST_F(VectorMatrixOperationsTest, Matrix_Vector_vs_Vector_Transpose_Consistency)
{
    mat3f test_matrix{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    // These should be different due to different operations
    vec3f matrix_vector = test_matrix * v1;
    vec3f vector_matrix = v1 * test_matrix;

    // They should not be equal (different mathematical operations)
    EXPECT_NE(matrix_vector.x, vector_matrix.x);
    EXPECT_NE(matrix_vector.y, vector_matrix.y);
    EXPECT_NE(matrix_vector.z, vector_matrix.z);
}

TEST_F(VectorMatrixOperationsTest, Homogeneous_Coordinate_Precision)
{
    vec3f original{1.234567f, 2.345678f, 3.456789f};

    // Test with various w values
    for (float w : {0.1f, 1.0f, 2.0f, 10.0f, 100.0f})
    {
        mat<float, 1, 4> homogeneous = to_homogeneous(original, w);
        vec3f recovered              = from_homogeneous(homogeneous);

        EXPECT_NEAR(recovered.x, original.x / w, 1e-6f);
        EXPECT_NEAR(recovered.y, original.y / w, 1e-6f);
        EXPECT_NEAR(recovered.z, original.z / w, 1e-6f);
    }
}

// ===== Edge Cases and Error Conditions =====

TEST_F(VectorMatrixOperationsTest, Zero_Vector_Operations)
{
    vec3f zero{0.0f, 0.0f, 0.0f};

    auto result_mat3 = identity_3x3 * zero;
    EXPECT_EQ(result_mat3.x, 0.0f);
    EXPECT_EQ(result_mat3.y, 0.0f);
    EXPECT_EQ(result_mat3.z, 0.0f);

    auto result_mat4 = identity_4x4 * zero;
    EXPECT_EQ(result_mat4.x, 0.0f);
    EXPECT_EQ(result_mat4.y, 0.0f);
    EXPECT_EQ(result_mat4.z, 0.0f);
}

TEST_F(VectorMatrixOperationsTest, Large_Values_Stability)
{
    vec3f large{1e6f, 2e6f, 3e6f};

    auto result_mat3 = identity_3x3 * large;
    EXPECT_NEAR(result_mat3.x, large.x, 1e-2f); // Allow for floating point precision
    EXPECT_NEAR(result_mat3.y, large.y, 1e-2f);
    EXPECT_NEAR(result_mat3.z, large.z, 1e-2f);

    auto result_mat4 = identity_4x4 * large;
    EXPECT_NEAR(result_mat4.x, large.x, 1e-2f);
    EXPECT_NEAR(result_mat4.y, large.y, 1e-2f);
    EXPECT_NEAR(result_mat4.z, large.z, 1e-2f);
}

TEST_F(VectorMatrixOperationsTest, Type_Consistency_Static_Assertions)
{
    // Verify return types are correct
    static_assert(std::is_same_v<decltype(identity_3x3 * v1), vec3f>);
    static_assert(std::is_same_v<decltype(identity_4x4 * v1), vec3f>);
    static_assert(std::is_same_v<decltype(v1 * identity_3x3), vec3f>);
    static_assert(std::is_same_v<decltype(v1 * identity_4x4), vec3f>);
    static_assert(std::is_same_v<decltype(to_homogeneous(v1)), mat<float, 1, 4>>);
    static_assert(std::is_same_v<decltype(from_homogeneous(std::declval<mat<float, 1, 4>>())), vec3f>);
}
