#include "lina.h"
#include <gtest/gtest.h>

using namespace lina;

class TransformTest : public ::testing::Test
{
  protected:
    static constexpr vec3f translation_vec{1.0f, 2.0f, 3.0f};
    static constexpr vec3f scale_vec{2.0f, 3.0f, 4.0f};
    static constexpr vec3f unit_x{1.0f, 0.0f, 0.0f};
    static constexpr vec3f unit_y{0.0f, 1.0f, 0.0f};
    static constexpr vec3f unit_z{0.0f, 0.0f, 1.0f};
    static constexpr mat3f identity_mat3 = identity<float, 3>();
    static constexpr mat4f identity_mat4 = identity<float, 4>();
};

TEST_F(TransformTest, Translation_Matrix)
{
    mat4f result = translation(translation_vec);

    // Check translation components
    EXPECT_EQ(result(0, 3), 1.0f);
    EXPECT_EQ(result(1, 3), 2.0f);
    EXPECT_EQ(result(2, 3), 3.0f);
    EXPECT_EQ(result(3, 3), 1.0f);

    // Check that upper-left 3x3 is identity
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            if (i == j)
            {
                EXPECT_EQ(result(i, j), 1.0f);
            }
            else
            {
                EXPECT_EQ(result(i, j), 0.0f);
            }
        }
    }
}

TEST_F(TransformTest, Rotation_FromMatrix3)
{
    mat4f result = rotation(identity_mat3);

    // Should create a 4x4 with the 3x3 in upper-left and [0,0,0,1] bottom row
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            EXPECT_EQ(result(i, j), identity_mat3(i, j));
        }
    }

    // Check bottom row and right column
    EXPECT_EQ(result(3, 3), 1.0f);
    EXPECT_EQ(result(0, 3), 0.0f);
    EXPECT_EQ(result(1, 3), 0.0f);
    EXPECT_EQ(result(2, 3), 0.0f);
    EXPECT_EQ(result(3, 0), 0.0f);
    EXPECT_EQ(result(3, 1), 0.0f);
    EXPECT_EQ(result(3, 2), 0.0f);
}

TEST_F(TransformTest, Scale_Matrix)
{
    mat4f result = scale(scale_vec);

    EXPECT_EQ(result(0, 0), 2.0f);
    EXPECT_EQ(result(1, 1), 3.0f);
    EXPECT_EQ(result(2, 2), 4.0f);
    EXPECT_EQ(result(3, 3), 1.0f);

    // All off-diagonal elements should be zero
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            if (i != j)
            {
                EXPECT_EQ(result(i, j), 0.0f);
            }
        }
    }
}

TEST_F(TransformTest, Transform_TRS)
{
    mat4f result = transform(translation_vec, identity_mat3, scale_vec);

    // This should be T * R * S
    // Since R is identity, result should be T * S
    mat4f expected = translation(translation_vec) * scale(scale_vec);

    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            EXPECT_FLOAT_EQ(result(i, j), expected(i, j));
        }
    }
}

TEST_F(TransformTest, RotationX_ZeroAngle)
{
    mat4f result = rotation_x(0.0f);

    // Should be identity matrix
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            EXPECT_FLOAT_EQ(result(i, j), identity_mat4(i, j));
        }
    }
}

TEST_F(TransformTest, RotationX_90Degrees)
{
    mat4f result = rotation_x(pi<float> / 2.0f);

    // After 90-degree rotation around X, Y becomes Z and Z becomes -Y
    EXPECT_NEAR(result(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(result(1, 1), 0.0f, 1e-6f);
    EXPECT_NEAR(result(1, 2), -1.0f, 1e-6f);
    EXPECT_NEAR(result(2, 1), 1.0f, 1e-6f);
    EXPECT_NEAR(result(2, 2), 0.0f, 1e-6f);
    EXPECT_NEAR(result(3, 3), 1.0f, 1e-6f);
}

TEST_F(TransformTest, RotationY_90Degrees)
{
    mat4f result = rotation_y(pi<float> / 2.0f);

    // After 90-degree rotation around Y, Z becomes X and X becomes -Z
    EXPECT_NEAR(result(0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(result(0, 2), 1.0f, 1e-6f);
    EXPECT_NEAR(result(1, 1), 1.0f, 1e-6f);
    EXPECT_NEAR(result(2, 0), -1.0f, 1e-6f);
    EXPECT_NEAR(result(2, 2), 0.0f, 1e-6f);
    EXPECT_NEAR(result(3, 3), 1.0f, 1e-6f);
}

TEST_F(TransformTest, RotationZ_90Degrees)
{
    mat4f result = rotation_z(pi<float> / 2.0f);

    // After 90-degree rotation around Z, X becomes Y and Y becomes -X
    EXPECT_NEAR(result(0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(result(0, 1), -1.0f, 1e-6f);
    EXPECT_NEAR(result(1, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(result(1, 1), 0.0f, 1e-6f);
    EXPECT_NEAR(result(2, 2), 1.0f, 1e-6f);
    EXPECT_NEAR(result(3, 3), 1.0f, 1e-6f);
}

TEST_F(TransformTest, Rotation_ArbitraryAxis_Identity)
{
    mat4f result = rotation(unit_x, 0.0f);

    // Zero rotation should give identity
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            EXPECT_NEAR(result(i, j), identity_mat4(i, j), 1e-6f);
        }
    }
}

TEST_F(TransformTest, Rotation_ArbitraryAxis_XAxis)
{
    mat4f result   = rotation(unit_x, pi<float> / 2.0f);
    mat4f expected = rotation_x(pi<float> / 2.0f);

    // Should be same as rotation_x
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            EXPECT_NEAR(result(i, j), expected(i, j), 1e-6f);
        }
    }
}

TEST_F(TransformTest, Rotate_Vector_ZeroAngle)
{
    vec3f v{1.0f, 2.0f, 3.0f};
    vec3f result = rotate(v, unit_x, 0.0f);

    EXPECT_NEAR(result.x, v.x, 1e-6f);
    EXPECT_NEAR(result.y, v.y, 1e-6f);
    EXPECT_NEAR(result.z, v.z, 1e-6f);
}

TEST_F(TransformTest, Rotate_Vector_90DegreesX)
{
    vec3f v{0.0f, 1.0f, 0.0f}; // Y axis
    vec3f result = rotate(v, unit_x, pi<float> / 2.0f);

    // Y should become Z after 90-degree rotation around X
    EXPECT_NEAR(result.x, 0.0f, 1e-6f);
    EXPECT_NEAR(result.y, 0.0f, 1e-6f);
    EXPECT_NEAR(result.z, 1.0f, 1e-6f);
}

TEST_F(TransformTest, IsRotationMatrix_Identity)
{
    EXPECT_TRUE(is_rotation_valid(identity_mat3));
}

TEST_F(TransformTest, IsRotationMatrix_ScaledMatrix)
{
    mat3f scaled_identity = identity_mat3 * 2.0f;
    EXPECT_FALSE(is_rotation_valid(scaled_identity));
}

TEST_F(TransformTest, IsRotationMatrix_ZeroMatrix)
{
    constexpr mat3f zero_matrix{};
    EXPECT_FALSE(is_rotation_valid(zero_matrix));
}

TEST_F(TransformTest, IsScaleValid_ValidScale)
{
    vec3f valid_scale{1.0f, 2.0f, 3.0f};
    EXPECT_TRUE(is_scale_valid(valid_scale));
}

TEST_F(TransformTest, IsScaleValid_ZeroScale)
{
    vec3f invalid_scale{0.0f, 2.0f, 3.0f};
    EXPECT_FALSE(is_scale_valid(invalid_scale));
}

TEST_F(TransformTest, GetTranslation)
{
    mat4f transform_matrix = translation(translation_vec);
    vec3f result           = get_translation(transform_matrix);

    EXPECT_FLOAT_EQ(result.x, 1.0f);
    EXPECT_FLOAT_EQ(result.y, 2.0f);
    EXPECT_FLOAT_EQ(result.z, 3.0f);
}

TEST_F(TransformTest, GetScale_UniformScale)
{
    mat4f scale_matrix = scale(scale_vec);
    vec3f result       = get_scale(scale_matrix);

    EXPECT_FLOAT_EQ(result.x, 2.0f);
    EXPECT_FLOAT_EQ(result.y, 3.0f);
    EXPECT_FLOAT_EQ(result.z, 4.0f);
}

TEST_F(TransformTest, GetRotation_Identity)
{
    mat4f transform_matrix = rotation(identity_mat3);
    mat3f result           = get_rotation(transform_matrix);

    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(result(i, j), identity_mat3(i, j), 1e-6f);
        }
    }
}

TEST_F(TransformTest, Decompose_TRS)
{
    mat4f original_transform = transform(translation_vec, identity_mat3, scale_vec);

    vec3f t, s;
    mat3f r;
    decompose(original_transform, t, r, s);

    EXPECT_NEAR(t.x, translation_vec.x, 1e-6f);
    EXPECT_NEAR(t.y, translation_vec.y, 1e-6f);
    EXPECT_NEAR(t.z, translation_vec.z, 1e-6f);

    EXPECT_NEAR(s.x, scale_vec.x, 1e-6f);
    EXPECT_NEAR(s.y, scale_vec.y, 1e-6f);
    EXPECT_NEAR(s.z, scale_vec.z, 1e-6f);
}

TEST_F(TransformTest, InverseTransform_Identity)
{
    mat4f result = inverse_transform(identity_mat4);

    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            EXPECT_NEAR(result(i, j), identity_mat4(i, j), 1e-6f);
        }
    }
}

TEST_F(TransformTest, ToHomogeneous)
{
    constexpr vec3f v{1.0f, 2.0f, 3.0f};
    constexpr mat<float, 1, 4> result = to_homogeneous(v, 1.0f);

    EXPECT_EQ(result(0, 0), 1.0f);
    EXPECT_EQ(result(0, 1), 2.0f);
    EXPECT_EQ(result(0, 2), 3.0f);
    EXPECT_EQ(result(0, 3), 1.0f);
}

TEST_F(TransformTest, FromHomogeneous)
{
    constexpr mat<float, 4, 1> homogeneous{2.0f, 4.0f, 6.0f, 2.0f};
    constexpr auto result = from_homogeneous(transpose(homogeneous));

    EXPECT_FLOAT_EQ(result.x, 1.0f); // 2/2
    EXPECT_FLOAT_EQ(result.y, 2.0f); // 4/2
    EXPECT_FLOAT_EQ(result.z, 3.0f); // 6/2
}

TEST_F(TransformTest, FromMat_Conversion)
{
    constexpr mat<float, 3, 1> matrix_vec{1.0f, 2.0f, 3.0f};
    constexpr vec3f result = static_cast<vec3f>(matrix_vec);

    EXPECT_EQ(result.x, 1.0f);
    EXPECT_EQ(result.y, 2.0f);
    EXPECT_EQ(result.z, 3.0f);
}

// ===== Complex Rotation Tests =====

TEST_F(TransformTest, Complex_Rotation_30_60_145_Degrees) {
    // Convert degrees to radians
    float angle_30 = 30.0f * pi<float> / 180.0f;
    float angle_60 = 60.0f * pi<float> / 180.0f;
    float angle_145 = 145.0f * pi<float> / 180.0f;
    
    // Create rotation matrices
    mat4f rot_x_30 = rotation_x(angle_30);
    mat4f rot_y_60 = rotation_y(angle_60);
    mat4f rot_z_145 = rotation_z(angle_145);
    
    // Test individual rotations
    vec3f test_vector{1.0f, 0.0f, 0.0f}; // Unit X vector
    
    // 30-degree rotation around X-axis (should not affect X component)
    vec3f result_x = rot_x_30 * test_vector;
    EXPECT_NEAR(result_x.x, 1.0f, 1e-6f);
    EXPECT_NEAR(result_x.y, 0.0f, 1e-6f);
    EXPECT_NEAR(result_x.z, 0.0f, 1e-6f);
    
    // 60-degree rotation around Y-axis 
    vec3f result_y = rot_y_60 * test_vector;
    EXPECT_NEAR(result_y.x, std::cos(angle_60), 1e-6f);  // cos(60°) = 0.5
    EXPECT_NEAR(result_y.y, 0.0f, 1e-6f);
    EXPECT_NEAR(result_y.z, -std::sin(angle_60), 1e-6f);  // sin(60°) ≈ 0.866
    
    // 145-degree rotation around Z-axis
    vec3f result_z = rot_z_145 * test_vector;
    EXPECT_NEAR(result_z.x, std::cos(angle_145), 1e-6f);  // cos(145°) ≈ -0.819
    EXPECT_NEAR(result_z.y, std::sin(angle_145), 1e-6f);  // sin(145°) ≈ 0.574
    EXPECT_NEAR(result_z.z, 0.0f, 1e-6f);
}

TEST_F(TransformTest, Complex_Composite_Rotation_XYZ) {
    // Convert degrees to radians
    float angle_30 = 30.0f * pi<float> / 180.0f;
    float angle_60 = 60.0f * pi<float> / 180.0f;
    float angle_145 = 145.0f * pi<float> / 180.0f;
    
    // Create individual rotation matrices
    mat4f rot_x = rotation_x(angle_30);
    mat4f rot_y = rotation_y(angle_60);
    mat4f rot_z = rotation_z(angle_145);
    
    // Create composite rotation: Z * Y * X (applied in reverse order)
    mat4f composite_rotation = rot_z * rot_y * rot_x;
    
    // Test with unit vectors
    vec3f unit_vectors[] = {
        {1.0f, 0.0f, 0.0f},  // X-axis
        {0.0f, 1.0f, 0.0f},  // Y-axis
        {0.0f, 0.0f, 1.0f},  // Z-axis
        {1.0f, 1.0f, 1.0f}   // Diagonal
    };
    
    for (const auto& test_vec : unit_vectors) {
        vec3f result = composite_rotation * test_vec;
        
        // Verify the result has the same length (rotation preserves length)
        float original_length = norm(test_vec);
        float result_length = norm(result);
        EXPECT_NEAR(result_length, original_length, 1e-5f);
        
        // Verify the transformation is not identity (should change the vector)
        if (original_length > 1e-6f) {  // Skip zero vectors
            float dot_product = dot(test_vec, result) / (original_length * result_length);
            // For these specific angles, the vectors should be significantly rotated
            EXPECT_LT(std::abs(dot_product), 0.99f); // Not too close to parallel
        }
    }
}

TEST_F(TransformTest, Complex_Rotation_Orthogonality_Check) {
    float angle_30 = 30.0f * pi<float> / 180.0f;
    float angle_60 = 60.0f * pi<float> / 180.0f;
    float angle_145 = 145.0f * pi<float> / 180.0f;
    
    mat4f rot_x = rotation_x(angle_30);
    mat4f rot_y = rotation_y(angle_60);
    mat4f rot_z = rotation_z(angle_145);
    mat4f composite = rot_z * rot_y * rot_x;
    
    // Extract the 3x3 rotation part
    mat3f rotation_3x3{};
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            rotation_3x3(i, j) = composite(i, j);
        }
    }
    
    // Test orthogonality: R * R^T should equal identity
    mat3f transpose_rot = transpose(rotation_3x3);
    mat3f should_be_identity = rotation_3x3 * transpose_rot;
    mat3f actual_identity = identity<float, 3>();
    
    EXPECT_TRUE(almost_equal(should_be_identity, actual_identity, 1e-5f));
    
    // Test determinant should be 1 (proper rotation, not reflection)
    float determinant = det(rotation_3x3);
    EXPECT_NEAR(determinant, 1.0f, 1e-6f);
}

TEST_F(TransformTest, Complex_Rotation_Specific_Results) {
    // Test the exact transformation of a known vector
    float angle_30 = 30.0f * pi<float> / 180.0f;
    float angle_60 = 60.0f * pi<float> / 180.0f;
    float angle_145 = 145.0f * pi<float> / 180.0f;
    
    mat4f rot_x = rotation_x(angle_30);
    mat4f rot_y = rotation_y(angle_60);
    mat4f rot_z = rotation_z(angle_145);
    mat4f composite = rot_x * rot_y * rot_z;
    
    // Transform a specific test vector
    vec3f test_vec{1.0f, 2.0f, 3.0f};
    vec3f result = composite * test_vec;
    
    // Verify length preservation
    EXPECT_NEAR(norm(result), norm(test_vec), 1e-5f);
    
    // Store the result for regression testing (these are the expected values
    // for this specific rotation sequence applied to this specific vector)
    // Note: These values were computed and verified to be mathematically correct
    float expected_x = 1.614924f;  // Approximate expected values
    float expected_y = -2.523516f;
    float expected_z = 2.241403f;
    
    // Allow reasonable tolerance for floating-point arithmetic
    EXPECT_NEAR(result.x, expected_x, 1e-6f);
    EXPECT_NEAR(result.y, expected_y, 1e-6f);
    EXPECT_NEAR(result.z, expected_z, 1e-6f);
}

TEST_F(TransformTest, Complex_Rotation_Arbitrary_Axis) {
    // Test rotation around arbitrary axis using the same angles
    float angle_30 = 30.0f * pi<float> / 180.0f;
    float angle_60 = 60.0f * pi<float> / 180.0f;
    float angle_145 = 145.0f * pi<float> / 180.0f;
    
    // Define arbitrary normalized axes
    vec3f axis_1 = normalize(vec3f{1.0f, 1.0f, 0.0f});
    vec3f axis_2 = normalize(vec3f{0.0f, 1.0f, 1.0f});
    vec3f axis_3 = normalize(vec3f{1.0f, 0.0f, 1.0f});
    
    mat4f rot_1 = rotation(axis_1, angle_30);
    mat4f rot_2 = rotation(axis_2, angle_60);
    mat4f rot_3 = rotation(axis_3, angle_145);
    
    mat4f composite = rot_3 * rot_2 * rot_1;
    
    vec3f test_vec{1.0f, 2.0f, 3.0f};
    vec3f result = composite * test_vec;
    
    // Verify fundamental rotation properties
    EXPECT_NEAR(norm(result), norm(test_vec), 1e-5f); // Length preservation
    
    // Extract 3x3 part and verify it's a valid rotation matrix
    mat3f rot_part{};
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            rot_part(i, j) = composite(i, j);
        }
    }
    
    EXPECT_TRUE(is_rotation_valid(rot_part));
}