#include "lina/lina.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace lina;

constexpr float EPSILON    = 1e-5f;
constexpr double EPSILON_D = 1e-12;

// ========================= look_at Function Tests =========================

class LookAtTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Common test vectors
        origin = vec3f{ 0.0f, 0.0f, 0.0f };
        x_axis = vec3f{ 1.0f, 0.0f, 0.0f };
        y_axis = vec3f{ 0.0f, 1.0f, 0.0f };
        z_axis = vec3f{ 0.0f, 0.0f, 1.0f };
        neg_z  = vec3f{ 0.0f, 0.0f, -1.0f };
    }

    // Helper function to verify matrix orthogonality
    static bool is_orthogonal(const mat4f& matrix, float epsilon = EPSILON)
    {
        vec3f row0{ matrix(0, 0), matrix(0, 1), matrix(0, 2) };
        vec3f row1{ matrix(1, 0), matrix(1, 1), matrix(1, 2) };
        vec3f row2{ matrix(2, 0), matrix(2, 1), matrix(2, 2) };

        // Check if rows are orthogonal
        bool orthogonal = almost_zero(dot(row0, row1), epsilon) && //
                          almost_zero(dot(row0, row2), epsilon) && //
                          almost_zero(dot(row1, row2), epsilon);

        // Check if rows are normalized
        bool normalized = almost_equal(norm(row0), 1.0f, epsilon) && //
                          almost_equal(norm(row1), 1.0f, epsilon) && //
                          almost_equal(norm(row2), 1.0f, epsilon);

        return orthogonal && normalized;
    }

    // Helper to check if homogeneous row is correct
    static bool has_correct_homogeneous_row(const mat4f& matrix, float epsilon = EPSILON)
    {
        return almost_zero(matrix(3, 0), epsilon) && //
               almost_zero(matrix(3, 1), epsilon) && //
               almost_zero(matrix(3, 2), epsilon) && //
               almost_equal(matrix(3, 3), 1.0f, epsilon);
    }

    vec3f origin, x_axis, y_axis, z_axis, neg_z;
};

TEST_F(LookAtTest, IdentityConfiguration)
{
    // Camera at origin looking down negative Z (standard OpenGL setup)
    vec3f eye{ 0.0f, 0.0f, 0.0f };
    vec3f center{ 0.0f, 0.0f, -1.0f };
    vec3f up{ 0.0f, 1.0f, 0.0f };

    mat4f view = look_at(eye, center, up);

    // Should be very close to identity matrix
    mat4f identity_matrix = identity<float, 4>();

    EXPECT_TRUE(almost_equal(view, identity_matrix, 1e-4f))
        << "look_at with standard configuration should produce near-identity matrix";

    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));
}

TEST_F(LookAtTest, CameraAlongPositiveZ)
{
    // Camera behind origin looking toward origin
    vec3f eye{ 0.0f, 0.0f, 5.0f };
    vec3f center{ 0.0f, 0.0f, 0.0f };
    vec3f up{ 0.0f, 1.0f, 0.0f };

    mat4f view = look_at(eye, center, up);

    // Check that matrix is orthogonal
    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));

    // Forward vector should point toward negative Z
    vec3f forward          = forward_vec(view);
    vec3f expected_forward = normalize(center - eye); // Should be (0, 0, -1)

    EXPECT_NEAR(forward.x, expected_forward.x, EPSILON);
    EXPECT_NEAR(forward.y, expected_forward.y, EPSILON);
    EXPECT_NEAR(forward.z, expected_forward.z, EPSILON);

    // Right vector should be positive X
    EXPECT_NEAR(view(0, 0), 1.0f, EPSILON); // Right X component
    EXPECT_NEAR(view(0, 1), 0.0f, EPSILON); // Right Y component
    EXPECT_NEAR(view(0, 2), 0.0f, EPSILON); // Right Z component

    // Up vector should be positive Y
    EXPECT_NEAR(view(1, 0), 0.0f, EPSILON); // Up X component
    EXPECT_NEAR(view(1, 1), 1.0f, EPSILON); // Up Y component
    EXPECT_NEAR(view(1, 2), 0.0f, EPSILON); // Up Z component
}

TEST_F(LookAtTest, CameraAlongPositiveX)
{
    // Camera to the right, looking at origin
    vec3f eye{ 5.0f, 0.0f, 0.0f };
    vec3f center{ 0.0f, 0.0f, 0.0f };
    vec3f up{ 0.0f, 1.0f, 0.0f };

    mat4f view = look_at(eye, center, up);

    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));

    // Forward vector should point toward negative X
    vec3f forward          = forward_vec(view);
    vec3f expected_forward = normalize(center - eye); // Should be (-1, 0, 0)

    EXPECT_NEAR(forward.x, expected_forward.x, EPSILON);
    EXPECT_NEAR(forward.y, expected_forward.y, EPSILON);
    EXPECT_NEAR(forward.z, expected_forward.z, EPSILON);

    // Right vector should point toward positive Z
    EXPECT_NEAR(view(2, 0), 1.0f, EPSILON);

    // Up vector should remain Y
    EXPECT_NEAR(view(1, 1), 1.0f, EPSILON);
}

TEST_F(LookAtTest, CameraAboveLookingDown)
{
    // Camera above, looking down at origin
    vec3f eye{ 0.0f, 5.0f, 0.0f };
    vec3f center{ 0.0f, 0.0f, 0.0f };
    vec3f up{ 0.0f, 0.0f, -1.0f }; // Up is now negative Z

    mat4f view = look_at(eye, center, up);

    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));

    // Forward vector should point toward negative Y
    vec3f forward = forward_vec(view);

    EXPECT_NEAR(forward.x, 0.0f, EPSILON);
    EXPECT_NEAR(forward.y, -1.0f, EPSILON);
    EXPECT_NEAR(forward.z, 0.0f, EPSILON);
}

TEST_F(LookAtTest, ArbitraryCameraPosition)
{
    // Camera at arbitrary position
    vec3f eye{ 3.0f, 4.0f, 5.0f };
    vec3f center{ 1.0f, 2.0f, 1.0f };
    vec3f up{ 0.0f, 1.0f, 0.0f };

    mat4f view = look_at(eye, center, up);

    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));

    // Check that forward direction is correct
    vec3f forward          = forward_vec(view);
    vec3f expected_forward = normalize(center - eye);

    EXPECT_NEAR(forward.x, expected_forward.x, EPSILON);
    EXPECT_NEAR(forward.y, expected_forward.y, EPSILON);
    EXPECT_NEAR(forward.z, expected_forward.z, EPSILON);
}

TEST_F(LookAtTest, TransformationCorrectness)
{
    // Test that the look_at matrix correctly transforms points
    vec3f eye{ 0.0f, 0.0f, 5.0f };
    vec3f center{ 0.0f, 0.0f, 0.0f };
    vec3f up{ 0.0f, 1.0f, 0.0f };

    mat4f view = look_at(eye, center, up);

    // A point at the center should map to (0, 0, -5) in view space
    vec3f world_center{ 0.0f, 0.0f, 0.0f };
    vec3f view_center = view * world_center;

    EXPECT_NEAR(view_center.x, 0.0f, EPSILON);
    EXPECT_NEAR(view_center.y, 0.0f, EPSILON);
    EXPECT_NEAR(view_center.z, -5.0f, EPSILON);

    // The eye position should map to origin in view space
    vec3f view_eye = view * eye;
    EXPECT_NEAR(view_eye.x, 0.0f, EPSILON);
    EXPECT_NEAR(view_eye.y, 0.0f, EPSILON);
    EXPECT_NEAR(view_eye.z, 0.0f, EPSILON);

    // A point to the right should map to positive X in view space
    vec3f world_right{ 1.0f, 0.0f, 0.0f };
    vec3f view_right = view * world_right;
    EXPECT_GT(view_right.x, 0.0f); // Should be positive
}

TEST_F(LookAtTest, UpVectorNormalization)
{
    // Test with non-normalized up vector
    vec3f eye{ 0.0f, 0.0f, 5.0f };
    vec3f center{ 0.0f, 0.0f, 0.0f };
    vec3f up{ 0.0f, 3.0f, 4.0f }; // Length = 5, not normalized

    mat4f view = look_at(eye, center, up);

    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));

    // The resulting matrix should still be valid despite non-normalized input
    vec3f up_result{ view(1, 0), view(1, 1), view(1, 2) };
    EXPECT_NEAR(norm(up_result), 1.0f, EPSILON);
}

TEST_F(LookAtTest, NearParallelUpVector)
{
    // Test case where up vector is nearly parallel to forward vector
    vec3f eye{ 0.0f, 0.0f, 5.0f };
    vec3f center{ 0.0f, 0.0f, 0.0f };
    vec3f up{ 0.1f, 0.0f, -1.0f }; // Nearly parallel to forward direction

    mat4f view = look_at(eye, center, up);

    // Should still produce orthogonal matrix (though up vector will be adjusted)
    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));
}

TEST_F(LookAtTest, DoublesPrecision)
{
    // Test with double precision
    vec3d eye_d{ 0.0, 0.0, 5.0 };
    vec3d center_d{ 0.0, 0.0, 0.0 };
    vec3d up_d{ 0.0, 1.0, 0.0 };

    mat<double, 4, 4> view_d = look_at(eye_d, center_d, up_d);

    // Check orthogonality with higher precision
    vec3d row0{ view_d(0, 0), view_d(0, 1), view_d(0, 2) };
    vec3d row1{ view_d(1, 0), view_d(1, 1), view_d(1, 2) };
    vec3d row2{ view_d(2, 0), view_d(2, 1), view_d(2, 2) };

    EXPECT_NEAR(dot(row0, row1), 0.0, EPSILON_D);
    EXPECT_NEAR(dot(row0, row2), 0.0, EPSILON_D);
    EXPECT_NEAR(dot(row1, row2), 0.0, EPSILON_D);

    EXPECT_NEAR(norm(row0), 1.0, EPSILON_D);
    EXPECT_NEAR(norm(row1), 1.0, EPSILON_D);
    EXPECT_NEAR(norm(row2), 1.0, EPSILON_D);
}

TEST_F(LookAtTest, ConsistentWithManualCalculation)
{
    // Test against manually calculated expected values
    vec3f eye{ 1.0f, 2.0f, 3.0f };
    vec3f center{ 4.0f, 5.0f, 6.0f };
    vec3f up{ 0.0f, 1.0f, 0.0f };

    // Manually calculate expected basis vectors
    vec3f forward   = normalize(center - eye); // (3,3,3) normalized = (√3/3, √3/3, √3/3)
    vec3f right     = normalize(cross(forward, up));
    vec3f camera_up = cross(right, forward);

    mat4f view = look_at(eye, center, up);

    // Check basis vectors match
    EXPECT_NEAR(view(0, 0), right.x, EPSILON);
    EXPECT_NEAR(view(0, 1), right.y, EPSILON);
    EXPECT_NEAR(view(0, 2), right.z, EPSILON);

    EXPECT_NEAR(view(1, 0), camera_up.x, EPSILON);
    EXPECT_NEAR(view(1, 1), camera_up.y, EPSILON);
    EXPECT_NEAR(view(1, 2), camera_up.z, EPSILON);

    EXPECT_NEAR(view(2, 0), -forward.x, EPSILON);
    EXPECT_NEAR(view(2, 1), -forward.y, EPSILON);
    EXPECT_NEAR(view(2, 2), -forward.z, EPSILON);

    // Check translation components
    EXPECT_NEAR(view(0, 3), -dot(right, eye), EPSILON);
    EXPECT_NEAR(view(1, 3), -dot(camera_up, eye), EPSILON);
    EXPECT_NEAR(view(2, 3), dot(forward, eye), EPSILON);
}

TEST_F(LookAtTest, InvertibilityTest)
{
    // Test that the look_at matrix can be conceptually inverted
    vec3f eye{ 2.0f, 3.0f, 4.0f };
    vec3f center{ 0.0f, 0.0f, 0.0f };
    vec3f up{ 0.0f, 1.0f, 0.0f };

    mat4f view = look_at(eye, center, up);

    // The eye position transformed to view space should be at origin
    vec3f transformed_eye = view * eye;

    EXPECT_NEAR(transformed_eye.x, 0.0f, EPSILON);
    EXPECT_NEAR(transformed_eye.y, 0.0f, EPSILON);
    EXPECT_NEAR(transformed_eye.z, 0.0f, EPSILON);
}

TEST_F(LookAtTest, NonUnitUpVectorHandling)
{
    // Test various non-unit up vectors
    vec3f eye{ 0.0f, 0.0f, 5.0f };
    vec3f center{ 0.0f, 0.0f, 0.0f };

    // Test very small up vector
    vec3f tiny_up{ 0.0f, 0.001f, 0.0f };
    mat4f view1 = look_at(eye, center, tiny_up);
    EXPECT_TRUE(is_orthogonal(view1));

    // Test very large up vector
    vec3f large_up{ 0.0f, 1000.0f, 0.0f };
    mat4f view2 = look_at(eye, center, large_up);
    EXPECT_TRUE(is_orthogonal(view2));

    // Both should produce similar results (after normalization)
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            EXPECT_NEAR(view1(i, j), view2(i, j), EPSILON * 10);
        }
    }
}

// Performance/stress test
TEST_F(LookAtTest, StressTest)
{
    // Test with many different configurations to ensure robustness
    constexpr int num_tests = 100;

    for (int i = 0; i < num_tests; ++i)
    {
        float angle = 2.0f * pi<float> * static_cast<float>(i) / num_tests;

        vec3f eye{ 5.0f * std::cos(angle), 2.0f, 5.0f * std::sin(angle) };
        vec3f center{ 0.0f, 0.0f, 0.0f };
        vec3f up{ 0.0f, 1.0f, 0.0f };

        mat4f view = look_at(eye, center, up);

        // Each matrix should be orthogonal
        EXPECT_TRUE(is_orthogonal(view));
        EXPECT_TRUE(has_correct_homogeneous_row(view));
    }
}

// ========================= Orthographic Projection Tests =========================

class OrthoProjectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test parameters
        left = -2.0f;
        right = 2.0f;
        bottom = -1.5f;
        top = 1.5f;
        near_plane = 0.1f;
        far_plane = 10.0f;

        // Standard orthographic matrix
        ortho_matrix = ortho(left, right, bottom, top, near_plane, far_plane);

        // Symmetric orthographic matrix
        symmetric_ortho = ortho(-1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 5.0f);
    }

    float left, right, bottom, top, near_plane, far_plane;
    mat4f ortho_matrix, symmetric_ortho;
};

TEST_F(OrthoProjectionTest, MatrixStructure) {
    // Test the diagonal elements
    EXPECT_NEAR(ortho_matrix(0, 0), 2.0f / (right - left), EPSILON);  // 0.5
    EXPECT_NEAR(ortho_matrix(1, 1), 2.0f / (top - bottom), EPSILON);  // 2/3
    EXPECT_NEAR(ortho_matrix(2, 2), -2.0f / (far_plane - near_plane), EPSILON); // -2/9.9
    EXPECT_NEAR(ortho_matrix(3, 3), 1.0f, EPSILON);

    // Test the translation elements
    EXPECT_NEAR(ortho_matrix(0, 3), -(right + left) / (right - left), EPSILON);  // 0
    EXPECT_NEAR(ortho_matrix(1, 3), -(top + bottom) / (top - bottom), EPSILON);  // 0
    EXPECT_NEAR(ortho_matrix(2, 3), -(far_plane + near_plane) / (far_plane - near_plane), EPSILON);

    // Test that other elements are zero
    EXPECT_NEAR(ortho_matrix(0, 1), 0.0f, EPSILON);
    EXPECT_NEAR(ortho_matrix(0, 2), 0.0f, EPSILON);
    EXPECT_NEAR(ortho_matrix(1, 0), 0.0f, EPSILON);
    EXPECT_NEAR(ortho_matrix(1, 2), 0.0f, EPSILON);
    EXPECT_NEAR(ortho_matrix(2, 0), 0.0f, EPSILON);
    EXPECT_NEAR(ortho_matrix(2, 1), 0.0f, EPSILON);
    EXPECT_NEAR(ortho_matrix(3, 0), 0.0f, EPSILON);
    EXPECT_NEAR(ortho_matrix(3, 1), 0.0f, EPSILON);
    EXPECT_NEAR(ortho_matrix(3, 2), 0.0f, EPSILON);
}

TEST_F(OrthoProjectionTest, CornerPointProjection) {
    // Test all 8 corners of the orthographic frustum

    // Near plane corners
    vec3f near_bottom_left{left, bottom, -near_plane};
    vec3f near_bottom_right{right, bottom, -near_plane};
    vec3f near_top_left{left, top, -near_plane};
    vec3f near_top_right{right, top, -near_plane};

    // Far plane corners
    vec3f far_bottom_left{left, bottom, -far_plane};
    vec3f far_bottom_right{right, bottom, -far_plane};
    vec3f far_top_left{left, top, -far_plane};
    vec3f far_top_right{right, top, -far_plane};

    // Project all corners
    vec3f proj_nbl = ortho_matrix * near_bottom_left;
    vec3f proj_nbr = ortho_matrix * near_bottom_right;
    vec3f proj_ntl = ortho_matrix * near_top_left;
    vec3f proj_ntr = ortho_matrix * near_top_right;
    vec3f proj_fbl = ortho_matrix * far_bottom_left;
    vec3f proj_fbr = ortho_matrix * far_bottom_right;
    vec3f proj_ftl = ortho_matrix * far_top_left;
    vec3f proj_ftr = ortho_matrix * far_top_right;

    // All corners should map to the canonical view volume [-1,1]³

    // Near plane should map to z = -1
    EXPECT_NEAR(proj_nbl.z, -1.0f, EPSILON);
    EXPECT_NEAR(proj_nbr.z, -1.0f, EPSILON);
    EXPECT_NEAR(proj_ntl.z, -1.0f, EPSILON);
    EXPECT_NEAR(proj_ntr.z, -1.0f, EPSILON);

    // Far plane should map to z = 1
    EXPECT_NEAR(proj_fbl.z, 1.0f, EPSILON);
    EXPECT_NEAR(proj_fbr.z, 1.0f, EPSILON);
    EXPECT_NEAR(proj_ftl.z, 1.0f, EPSILON);
    EXPECT_NEAR(proj_ftr.z, 1.0f, EPSILON);

    // Left edge should map to x = -1
    EXPECT_NEAR(proj_nbl.x, -1.0f, EPSILON);
    EXPECT_NEAR(proj_ntl.x, -1.0f, EPSILON);
    EXPECT_NEAR(proj_fbl.x, -1.0f, EPSILON);
    EXPECT_NEAR(proj_ftl.x, -1.0f, EPSILON);

    // Right edge should map to x = 1
    EXPECT_NEAR(proj_nbr.x, 1.0f, EPSILON);
    EXPECT_NEAR(proj_ntr.x, 1.0f, EPSILON);
    EXPECT_NEAR(proj_fbr.x, 1.0f, EPSILON);
    EXPECT_NEAR(proj_ftr.x, 1.0f, EPSILON);

    // Bottom edge should map to y = -1
    EXPECT_NEAR(proj_nbl.y, -1.0f, EPSILON);
    EXPECT_NEAR(proj_nbr.y, -1.0f, EPSILON);
    EXPECT_NEAR(proj_fbl.y, -1.0f, EPSILON);
    EXPECT_NEAR(proj_fbr.y, -1.0f, EPSILON);

    // Top edge should map to y = 1
    EXPECT_NEAR(proj_ntl.y, 1.0f, EPSILON);
    EXPECT_NEAR(proj_ntr.y, 1.0f, EPSILON);
    EXPECT_NEAR(proj_ftl.y, 1.0f, EPSILON);
    EXPECT_NEAR(proj_ftr.y, 1.0f, EPSILON);
}

TEST_F(OrthoProjectionTest, CenterPointProjection) {
    // Center of the frustum should map to origin
    vec3f center{(left + right) / 2.0f, (bottom + top) / 2.0f, -(near_plane + far_plane) / 2.0f};
    vec3f projected = ortho_matrix * center;

    EXPECT_NEAR(projected.x, 0.0f, EPSILON);
    EXPECT_NEAR(projected.y, 0.0f, EPSILON);
    EXPECT_NEAR(projected.z, 0.0f, EPSILON);
}

TEST_F(OrthoProjectionTest, ParallelLinesRemainParallel) {
    // Orthographic projection should preserve parallel lines

    // Two parallel lines at different depths
    vec3f line1_start{0.0f, 0.0f, -1.0f};
    vec3f line1_end{1.0f, 1.0f, -1.0f};
    vec3f line2_start{0.0f, 0.0f, -5.0f};
    vec3f line2_end{1.0f, 1.0f, -5.0f};

    vec3f proj_line1_start = ortho_matrix * line1_start;
    vec3f proj_line1_end = ortho_matrix * line1_end;
    vec3f proj_line2_start = ortho_matrix * line2_start;
    vec3f proj_line2_end = ortho_matrix * line2_end;

    // Calculate direction vectors
    vec3f dir1 = proj_line1_end - proj_line1_start;
    vec3f dir2 = proj_line2_end - proj_line2_start;

    // Direction vectors should be the same (parallel)
    EXPECT_NEAR(dir1.x, dir2.x, EPSILON);
    EXPECT_NEAR(dir1.y, dir2.y, EPSILON);
    // Z components will be different due to different depths
}

TEST_F(OrthoProjectionTest, SymmetricMatrix) {
    // Test symmetric orthographic matrix (left=-right, bottom=-top)

    // For symmetric matrix, translation components should be zero
    EXPECT_NEAR(symmetric_ortho(0, 3), 0.0f, EPSILON);
    EXPECT_NEAR(symmetric_ortho(1, 3), 0.0f, EPSILON);

    // Diagonal components should be simple
    EXPECT_NEAR(symmetric_ortho(0, 0), 1.0f, EPSILON);  // 2/2
    EXPECT_NEAR(symmetric_ortho(1, 1), 1.0f, EPSILON);  // 2/2
    EXPECT_NEAR(symmetric_ortho(2, 2), -0.5f, EPSILON); // -2/4
}

TEST_F(OrthoProjectionTest, DoublePrecision) {
    // Test with double precision
    double left_d = -3.0, right_d = 3.0, bottom_d = -2.0, top_d = 2.0;
    double near_d = 0.5, far_d = 20.0;

    mat<double, 4, 4> ortho_d = ortho(left_d, right_d, bottom_d, top_d, near_d, far_d);

    // Test diagonal elements with higher precision
    EXPECT_NEAR(ortho_d(0, 0), 2.0 / (right_d - left_d), EPSILON_D * 100);
    EXPECT_NEAR(ortho_d(1, 1), 2.0 / (top_d - bottom_d), EPSILON_D * 100);
    EXPECT_NEAR(ortho_d(2, 2), -2.0 / (far_d - near_d), EPSILON_D * 100);
    EXPECT_NEAR(ortho_d(3, 3), 1.0, EPSILON_D);
}

TEST_F(OrthoProjectionTest, ConstexprSupport) {
    // Test compile-time evaluation
    constexpr mat4f const_ortho = ortho(-1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 3.0f);

    static_assert(const_ortho(0, 0) == 1.0f);  // 2/2
    static_assert(const_ortho(1, 1) == 1.0f);  // 2/2
    static_assert(const_ortho(2, 2) == -1.0f); // -2/2
    static_assert(const_ortho(3, 3) == 1.0f);
    static_assert(const_ortho(0, 3) == 0.0f);  // Symmetric case
    static_assert(const_ortho(1, 3) == 0.0f);  // Symmetric case

    EXPECT_EQ(const_ortho(0, 0), 1.0f);
    EXPECT_EQ(const_ortho(3, 3), 1.0f);
}

TEST_F(OrthoProjectionTest, DepthMapping) {
    // Test depth mapping specifically

    // Points at near and far planes
    vec3f at_near{0.0f, 0.0f, -near_plane};
    vec3f at_far{0.0f, 0.0f, -far_plane};
    vec3f at_middle{0.0f, 0.0f, -(near_plane + far_plane) / 2.0f};

    vec3f proj_near = ortho_matrix * at_near;
    vec3f proj_far = ortho_matrix * at_far;
    vec3f proj_middle = ortho_matrix * at_middle;

    // Near plane maps to z = -1
    EXPECT_NEAR(proj_near.z, -1.0f, EPSILON);

    // Far plane maps to z = 1
    EXPECT_NEAR(proj_far.z, 1.0f, EPSILON);

    // Middle should map to z = 0
    EXPECT_NEAR(proj_middle.z, 0.0f, EPSILON);

    // X and Y should be unchanged for points on the Z axis
    EXPECT_NEAR(proj_near.x, 0.0f, EPSILON);
    EXPECT_NEAR(proj_near.y, 0.0f, EPSILON);
    EXPECT_NEAR(proj_far.x, 0.0f, EPSILON);
    EXPECT_NEAR(proj_far.y, 0.0f, EPSILON);
}

TEST_F(OrthoProjectionTest, ScalingBehavior) {
    // Test that orthographic projection scales objects uniformly

    // Square in world space
    vec3f square_corners[4] = {
        {-0.5f, -0.5f, -2.0f},
        { 0.5f, -0.5f, -2.0f},
        { 0.5f,  0.5f, -2.0f},
        {-0.5f,  0.5f, -2.0f}
    };

    // Same square at different depth
    vec3f square_far[4] = {
        {-0.5f, -0.5f, -8.0f},
        { 0.5f, -0.5f, -8.0f},
        { 0.5f,  0.5f, -8.0f},
        {-0.5f,  0.5f, -8.0f}
    };

    // Project both squares
    vec3f proj_near[4], proj_far[4];
    for (int i = 0; i < 4; ++i) {
        proj_near[i] = ortho_matrix * square_corners[i];
        proj_far[i] = ortho_matrix * square_far[i];
    }

    // Calculate edge lengths for both projected squares
    float near_edge_x = std::abs(proj_near[1].x - proj_near[0].x);
    float near_edge_y = std::abs(proj_near[2].y - proj_near[1].y);
    float far_edge_x = std::abs(proj_far[1].x - proj_far[0].x);
    float far_edge_y = std::abs(proj_far[2].y - proj_far[1].y);

    // Edge lengths should be the same (no perspective distortion)
    EXPECT_NEAR(near_edge_x, far_edge_x, EPSILON);
    EXPECT_NEAR(near_edge_y, far_edge_y, EPSILON);
}

TEST_F(OrthoProjectionTest, NonSymmetricProjection) {
    // Test off-center projection
    mat4f off_center = ortho(-1.0f, 3.0f, -2.0f, 1.0f, 1.0f, 5.0f);

    // Center of frustum should not map to origin
    vec3f frustum_center{1.0f, -0.5f, -3.0f}; // center = (left+right)/2, (bottom+top)/2, -(near+far)/2
    vec3f projected_center = off_center * frustum_center;

    EXPECT_NEAR(projected_center.x, 0.0f, EPSILON);
    EXPECT_NEAR(projected_center.y, 0.0f, EPSILON);
    EXPECT_NEAR(projected_center.z, 0.0f, EPSILON);
}

TEST_F(OrthoProjectionTest, InverseProperty) {
    // Test that the orthographic matrix is invertible and behaves correctly
    mat4f ortho_inv = inverse(symmetric_ortho);
    mat4f identity_check = symmetric_ortho * ortho_inv;
    mat4f identity_expected = identity<float, 4>();

    EXPECT_TRUE(almost_equal(identity_check, identity_expected, EPSILON * 10));

    // Test point round-trip
    vec3f original{0.5f, -0.3f, -2.0f};
    vec3f projected = symmetric_ortho * original;
    vec3f back_to_original = ortho_inv * projected;

    EXPECT_TRUE(almost_equal(original, back_to_original, EPSILON * 10));
}