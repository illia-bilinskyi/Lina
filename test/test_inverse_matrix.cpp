#include "lina/lina.h"
#include <gtest/gtest.h>

using namespace lina;

const float EPSILON    = 1e-5f;
const double EPSILON_D = 1e-12;

// ========================= 4x4 Matrix Inverse Tests =========================

class Mat4InverseTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Common test matrices
        identity_mat = identity<float, 4>();

        // Simple scaling matrix
        scale_mat = mat4f{
            2, 0, 0, 0, //
            0, 3, 0, 0, //
            0, 0, 4, 0, //
            0, 0, 0, 1  //
        };

        // Translation matrix
        translation_mat = mat4f{
            1, 0, 0, 5,  //
            0, 1, 0, 3,  //
            0, 0, 1, -2, //
            0, 0, 0, 1   //
        };

        // General invertible matrix
        general_mat = mat4f{
            1, 2, 0, 1, //
            0, 1, 1, 2, //
            2, 0, 1, 0, //
            1, 1, 2, 1  //
        };

        // Singular matrix (rank < 4)
        singular_mat = mat4f{ 1, 2, 3, 4, 2, 4, 6, 8, // 2x first row
                              0, 1, 2, 3, 1, 0, 1, 2 };
    }

    // Helper function to check if A * A^(-1) ≈ I
    static bool verify_inverse(const mat4f& matrix, const mat4f& inverse, float epsilon = EPSILON)
    {
        mat4f product        = matrix * inverse;
        mat4f identity_check = identity<float, 4>();

        return almost_equal(product, identity_check, epsilon);
    }

    mat4f identity_mat, scale_mat, translation_mat, general_mat, singular_mat;
};

TEST_F(Mat4InverseTest, IdentityMatrixInverse)
{
    mat4f inv = inverse(identity_mat);

    // Inverse of identity should be identity
    EXPECT_TRUE(almost_equal(inv, identity_mat, EPSILON));

    // Verify A * A^(-1) = I
    EXPECT_TRUE(verify_inverse(identity_mat, inv));
}

TEST_F(Mat4InverseTest, DiagonalMatrixInverse)
{
    mat4f inv = inverse(scale_mat);

    // Expected inverse: diagonal elements should be reciprocals
    mat4f expected{
        0.5f, 0,           0,     0, //
        0,    1.0f / 3.0f, 0,     0, //
        0,    0,           0.25f, 0, //
        0,    0,           0,     1  //
    };

    EXPECT_TRUE(almost_equal(inv, expected, EPSILON));
    EXPECT_TRUE(verify_inverse(scale_mat, inv));

    // Check specific diagonal elements
    EXPECT_NEAR(inv(0, 0), 0.5f, EPSILON);
    EXPECT_NEAR(inv(1, 1), 1.0f / 3.0f, EPSILON);
    EXPECT_NEAR(inv(2, 2), 0.25f, EPSILON);
    EXPECT_NEAR(inv(3, 3), 1.0f, EPSILON);
}

TEST_F(Mat4InverseTest, TranslationMatrixInverse)
{
    mat4f inv = inverse(translation_mat);

    // Translation inverse should negate the translation part
    mat4f expected{
        1, 0, 0, -5, //
        0, 1, 0, -3, //
        0, 0, 1, 2,  //
        0, 0, 0, 1   //
    };

    EXPECT_TRUE(almost_equal(inv, expected, EPSILON));
    EXPECT_TRUE(verify_inverse(translation_mat, inv));

    // Check translation components specifically
    EXPECT_NEAR(inv(0, 3), -5.0f, EPSILON);
    EXPECT_NEAR(inv(1, 3), -3.0f, EPSILON);
    EXPECT_NEAR(inv(2, 3), 2.0f, EPSILON);
}

TEST_F(Mat4InverseTest, GeneralMatrixInverse)
{
    mat4f inv = inverse(general_mat);

    // Verify A * A^(-1) = I (this is the key test for general matrices)
    EXPECT_TRUE(verify_inverse(general_mat, inv));

    // Also verify A^(-1) * A = I
    mat4f product_reverse = inv * general_mat;
    mat4f identity_check  = identity<float, 4>();
    EXPECT_TRUE(almost_equal(product_reverse, identity_check, EPSILON));
}

TEST_F(Mat4InverseTest, SingularMatrixThrows)
{
    // Singular matrix should throw exception
    EXPECT_THROW(inverse(singular_mat), std::invalid_argument);

    // Test with a matrix of all zeros
    mat4f zero_mat{};
    EXPECT_THROW(inverse(zero_mat), std::invalid_argument);

    // Test with a matrix where one row is all zeros
    mat4f one_zero_row{
        1, 2,  3,  4, //
        0, 0,  0,  0, // Zero row
        5, 6,  7,  8, //
        9, 10, 11, 12 //
    };
    EXPECT_THROW(inverse(one_zero_row), std::invalid_argument);
}

TEST_F(Mat4InverseTest, RotationMatrixInverse)
{
    // Create a rotation matrix around Z-axis (90 degrees)
    float angle    = pi<float> / 2.0f; // 90 degrees
    mat4f rotation = rotation_z(angle);

    mat4f inv = inverse(rotation);

    // Verify the inverse
    EXPECT_TRUE(verify_inverse(rotation, inv));

    // For rotation matrices, inverse should equal transpose (for the 3x3 part)
    mat4f rotation_transpose = transpose(rotation);

    // Check that the rotation part (3x3) of inverse equals transpose
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(inv(i, j), rotation_transpose(i, j), EPSILON);
        }
    }
}

TEST_F(Mat4InverseTest, TransformationMatrixInverse)
{
    // Create a complex transformation: translation + rotation + scaling
    vec3f trans_vec{ 2.0f, -1.0f, 3.0f };
    vec3f scale_vec{ 0.5f, 2.0f, 1.5f };

    mat4f T = translation(trans_vec);
    mat4f S = scale(scale_vec);
    mat4f R = rotation_z(pi<float> / 4.0f); // 45 degrees

    mat4f combined = T * R * S;
    mat4f inv      = inverse(combined);

    EXPECT_TRUE(verify_inverse(combined, inv));

    // Test point transformation round-trip
    vec3f original_point{ 1.0f, 2.0f, 3.0f };
    vec3f transformed      = combined * original_point;
    vec3f back_to_original = inv * transformed;

    EXPECT_TRUE(almost_equal(original_point, back_to_original, EPSILON * 10));
}

TEST_F(Mat4InverseTest, DoublePrecisionInverse)
{
    // Test with double precision
    mat4d double_mat{
        1.0, 2.0, 0.0, 1.0, //
        0.0, 1.0, 1.0, 2.0, //
        2.0, 0.0, 1.0, 0.0, //
        1.0, 1.0, 2.0, 1.0  //
    };

    mat4d inv = inverse(double_mat);

    // Verify with higher precision
    mat4d product    = double_mat * inv;
    mat4d identity_d = identity<double, 4>();

    EXPECT_TRUE(almost_equal(product, identity_d, EPSILON_D * 100));
}

TEST_F(Mat4InverseTest, ConstexprInverse)
{
    // Test compile-time inverse computation
    constexpr mat4f const_scale{
        2, 0, 0, 0, //
        0, 3, 0, 0, //
        0, 0, 4, 0, //
        0, 0, 0, 1  //
    };

    constexpr mat4f const_inv = c_inverse(const_scale);

    // Static assertions for compile-time verification
    static_assert(almost_equal(const_inv(0, 0), 0.5f, 1e-6f));
    static_assert(almost_equal(const_inv(1, 1), 1.0f / 3.0f, 1e-6f));
    static_assert(almost_equal(const_inv(2, 2), 0.25f, 1e-6f));
    static_assert(almost_equal(const_inv(3, 3), 1.0f, 1e-6f));

    // Runtime verification
    EXPECT_NEAR(const_inv(0, 0), 0.5f, EPSILON);
    EXPECT_NEAR(const_inv(1, 1), 1.0f / 3.0f, EPSILON);
    EXPECT_NEAR(const_inv(2, 2), 0.25f, EPSILON);
    EXPECT_NEAR(const_inv(3, 3), 1.0f, EPSILON);
}

TEST_F(Mat4InverseTest, OrthogonalMatrixInverse)
{
    // Create an orthogonal matrix (rotation)
    mat4f ortho{
        0,  1, 0, 0, //
        -1, 0, 0, 0, //
        0,  0, 1, 0, //
        0,  0, 0, 1  //
    };

    mat4f inv = inverse(ortho);

    // For orthogonal matrices, inverse should equal transpose
    mat4f ortho_transpose = transpose(ortho);
    EXPECT_TRUE(almost_equal(inv, ortho_transpose, EPSILON));

    EXPECT_TRUE(verify_inverse(ortho, inv));
}

TEST_F(Mat4InverseTest, NearSingularMatrix)
{
    // Test with a matrix that's close to singular but still invertible
    mat4f near_singular{
        1, 2, 3, 4,      //
        2, 4, 6, 8.001f, // Almost 2x the first row
        0, 1, 2, 3,      //
        1, 0, 1, 2       //
    };

    // Should not throw (determinant is very small but non-zero)
    mat4f inv;
    EXPECT_NO_THROW(inv = inverse(near_singular));

    // Verification might be less accurate due to numerical issues
    EXPECT_TRUE(verify_inverse(near_singular, inv, EPSILON * 1000));
}

TEST_F(Mat4InverseTest, UpperTriangularMatrix)
{
    mat4f upper_tri{
        2, 1, 3, 4, //
        0, 3, 2, 1, //
        0, 0, 4, 5, //
        0, 0, 0, 2  //
    };

    mat4f inv = inverse(upper_tri);
    EXPECT_TRUE(verify_inverse(upper_tri, inv));

    // Upper triangular inverse should also be upper triangular
    EXPECT_NEAR(inv(1, 0), 0.0f, EPSILON);
    EXPECT_NEAR(inv(2, 0), 0.0f, EPSILON);
    EXPECT_NEAR(inv(2, 1), 0.0f, EPSILON);
    EXPECT_NEAR(inv(3, 0), 0.0f, EPSILON);
    EXPECT_NEAR(inv(3, 1), 0.0f, EPSILON);
    EXPECT_NEAR(inv(3, 2), 0.0f, EPSILON);
}

TEST_F(Mat4InverseTest, LowerTriangularMatrix)
{
    mat4f lower_tri{
        2, 0, 0, 0, //
        1, 3, 0, 0, //
        3, 2, 4, 0, //
        4, 1, 5, 2  //
    };

    mat4f inv = inverse(lower_tri);
    EXPECT_TRUE(verify_inverse(lower_tri, inv));

    // Lower triangular inverse should also be lower triangular
    EXPECT_NEAR(inv(0, 1), 0.0f, EPSILON);
    EXPECT_NEAR(inv(0, 2), 0.0f, EPSILON);
    EXPECT_NEAR(inv(0, 3), 0.0f, EPSILON);
    EXPECT_NEAR(inv(1, 2), 0.0f, EPSILON);
    EXPECT_NEAR(inv(1, 3), 0.0f, EPSILON);
    EXPECT_NEAR(inv(2, 3), 0.0f, EPSILON);
}

TEST_F(Mat4InverseTest, InverseOfInverse)
{
    // Test that (A^(-1))^(-1) = A
    mat4f inv1 = inverse(general_mat);
    mat4f inv2 = inverse(inv1);

    EXPECT_TRUE(almost_equal(inv2, general_mat, EPSILON * 10));
}

TEST_F(Mat4InverseTest, DeterminantConsistency)
{
    // Test that det(A^(-1)) = 1/det(A)
    float det_original = det(general_mat);
    mat4f inv          = inverse(general_mat);
    float det_inverse  = det(inv);

    EXPECT_NEAR(det_inverse * det_original, 1.0f, EPSILON * 10);
}

TEST_F(Mat4InverseTest, LookAtMatrixInverse)
{
    // Test inverse of a look-at matrix
    vec3f eye{ 5.0f, 3.0f, 2.0f };
    vec3f center{ 0.0f, 0.0f, 0.0f };
    vec3f up{ 0.0f, 1.0f, 0.0f };

    mat4f view = look_at(eye, center, up);
    mat4f inv  = inverse(view);

    EXPECT_TRUE(verify_inverse(view, inv));

    // Test that applying view then inverse gets back to original
    vec3f world_point{ 1.0f, 2.0f, 3.0f };
    vec3f view_point    = view * world_point;
    vec3f back_to_world = inv * view_point;

    EXPECT_TRUE(almost_equal(world_point, back_to_world, EPSILON * 10));
}

TEST_F(Mat4InverseTest, StressTest)
{
    // Test multiple random-ish matrices for robustness
    for (int i = 1; i <= 10; ++i)
    {
        auto scale = static_cast<float>(i);

        mat4f test_matrix{
            scale,     2 * scale, 0,         scale,     //
            0,         scale,     scale,     2 * scale, //
            2 * scale, 0,         scale,     0,         //
            scale,     scale,     2 * scale, scale      //
        };

        // Ensure it's not singular by adding to diagonal
        test_matrix(0, 0) += 0.1f;
        test_matrix(1, 1) += 0.1f;
        test_matrix(2, 2) += 0.1f;
        test_matrix(3, 3) += 0.1f;

        mat4f inv = inverse(test_matrix);
        EXPECT_TRUE(verify_inverse(test_matrix, inv, EPSILON * scale * 10));
    }
}