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
    constexpr vec3f eye{ 0.0f, 0.0f, 0.0f };
    constexpr vec3f center{ 0.0f, 0.0f, -1.0f };
    constexpr vec3f up{ 0.0f, 1.0f, 0.0f };

    constexpr mat4f view = look_at(eye, center, up);

    // Should be very close to identity matrix
    constexpr mat4f identity_matrix = identity<float, 4>();

    EXPECT_TRUE(almost_equal(view, identity_matrix, 1e-4f))
        << "look_at with standard configuration should produce near-identity matrix";

    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));
}

TEST_F(LookAtTest, CameraAlongPositiveZ)
{
    // Camera behind origin looking toward origin
    constexpr vec3f eye{ 0.0f, 0.0f, 5.0f };
    constexpr vec3f center{ 0.0f, 0.0f, 0.0f };
    constexpr vec3f up{ 0.0f, 1.0f, 0.0f };

    constexpr mat4f view = look_at(eye, center, up);

    // Check that matrix is orthogonal
    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));

    // Forward vector should point toward negative Z
    constexpr vec3f forward          = forward_vec(view);
    constexpr vec3f expected_forward = normalize(center - eye); // Should be (0, 0, -1)

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
    constexpr vec3f eye{ 5.0f, 0.0f, 0.0f };
    constexpr vec3f center{ 0.0f, 0.0f, 0.0f };
    constexpr vec3f up{ 0.0f, 1.0f, 0.0f };

    constexpr mat4f view = look_at(eye, center, up);

    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));

    // Forward vector should point toward negative X
    constexpr vec3f forward          = forward_vec(view);
    constexpr vec3f expected_forward = normalize(center - eye); // Should be (-1, 0, 0)

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
    constexpr vec3f eye{ 0.0f, 5.0f, 0.0f };
    constexpr vec3f center{ 0.0f, 0.0f, 0.0f };
    constexpr vec3f up{ 0.0f, 0.0f, -1.0f }; // Up is now negative Z

    constexpr mat4f view = look_at(eye, center, up);

    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));

    // Forward vector should point toward negative Y
    constexpr vec3f forward = forward_vec(view);

    EXPECT_NEAR(forward.x, 0.0f, EPSILON);
    EXPECT_NEAR(forward.y, -1.0f, EPSILON);
    EXPECT_NEAR(forward.z, 0.0f, EPSILON);
}

TEST_F(LookAtTest, ArbitraryCameraPosition)
{
    // Camera at arbitrary position
    constexpr vec3f eye{ 3.0f, 4.0f, 5.0f };
    constexpr vec3f center{ 1.0f, 2.0f, 1.0f };
    constexpr vec3f up{ 0.0f, 1.0f, 0.0f };

    constexpr mat4f view = look_at(eye, center, up);

    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));

    // Check that forward direction is correct
    constexpr vec3f forward          = forward_vec(view);
    constexpr vec3f expected_forward = normalize(center - eye);

    EXPECT_NEAR(forward.x, expected_forward.x, EPSILON);
    EXPECT_NEAR(forward.y, expected_forward.y, EPSILON);
    EXPECT_NEAR(forward.z, expected_forward.z, EPSILON);
}

TEST_F(LookAtTest, TransformationCorrectness)
{
    // Test that the look_at matrix correctly transforms points
    constexpr vec3f eye{ 0.0f, 0.0f, 5.0f };
    constexpr vec3f center{ 0.0f, 0.0f, 0.0f };
    constexpr vec3f up{ 0.0f, 1.0f, 0.0f };

    constexpr mat4f view = look_at(eye, center, up);

    // A point at the center should map to (0, 0, -5) in view space
    constexpr vec3f world_center{ 0.0f, 0.0f, 0.0f };
    constexpr vec3f view_center = view * world_center;

    EXPECT_NEAR(view_center.x, 0.0f, EPSILON);
    EXPECT_NEAR(view_center.y, 0.0f, EPSILON);
    EXPECT_NEAR(view_center.z, -5.0f, EPSILON);

    // The eye position should map to origin in view space
    constexpr vec3f view_eye = view * eye;
    EXPECT_NEAR(view_eye.x, 0.0f, EPSILON);
    EXPECT_NEAR(view_eye.y, 0.0f, EPSILON);
    EXPECT_NEAR(view_eye.z, 0.0f, EPSILON);

    // A point to the right should map to positive X in view space
    constexpr vec3f world_right{ 1.0f, 0.0f, 0.0f };
    constexpr vec3f view_right = view * world_right;
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

class PerspectiveProjectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test parameters
        fov_45 = pi<float> / 4.0f;  // 45 degrees in radians
        aspect_ratio = 16.0f / 9.0f;
        near_plane = 0.1f;
        far_plane = 100.0f;

        // Standard perspective matrix
        persp_matrix = perspective(fov_45, aspect_ratio, near_plane, far_plane);

        // Square aspect ratio for simpler calculations
        persp_square = perspective(fov_45, 1.0f, 1.0f, 10.0f);
    }

    float fov_45, aspect_ratio, near_plane, far_plane;
    mat4f persp_matrix, persp_square;
};

TEST_F(PerspectiveProjectionTest, MatrixStructure) {
    float tan_half_fov = std::tan(fov_45 / 2.0f);

    // Test the key matrix elements
    EXPECT_NEAR(persp_matrix(0, 0), 1.0f / (aspect_ratio * tan_half_fov), EPSILON);
    EXPECT_NEAR(persp_matrix(1, 1), 1.0f / tan_half_fov, EPSILON);
    EXPECT_NEAR(persp_matrix(2, 2), -(far_plane + near_plane) / (far_plane - near_plane), EPSILON);
    EXPECT_NEAR(persp_matrix(2, 3), -(2.0f * far_plane * near_plane) / (far_plane - near_plane), EPSILON);
    EXPECT_NEAR(persp_matrix(3, 2), -1.0f, EPSILON);

    // Test that other elements are zero
    EXPECT_NEAR(persp_matrix(0, 1), 0.0f, EPSILON);
    EXPECT_NEAR(persp_matrix(0, 2), 0.0f, EPSILON);
    EXPECT_NEAR(persp_matrix(0, 3), 0.0f, EPSILON);
    EXPECT_NEAR(persp_matrix(1, 0), 0.0f, EPSILON);
    EXPECT_NEAR(persp_matrix(1, 2), 0.0f, EPSILON);
    EXPECT_NEAR(persp_matrix(1, 3), 0.0f, EPSILON);
    EXPECT_NEAR(persp_matrix(2, 0), 0.0f, EPSILON);
    EXPECT_NEAR(persp_matrix(2, 1), 0.0f, EPSILON);
    EXPECT_NEAR(persp_matrix(3, 0), 0.0f, EPSILON);
    EXPECT_NEAR(persp_matrix(3, 1), 0.0f, EPSILON);
    EXPECT_NEAR(persp_matrix(3, 3), 0.0f, EPSILON);
}

TEST_F(PerspectiveProjectionTest, PerspectiveDivide) {
    // Test that the perspective divide works correctly
    vec3f point{1.0f, 1.0f, -5.0f};
    vec3f projected = persp_square * point;

    // After perspective projection, we need to divide by w (which is -z)
    float w = 5.0f;  // -(-5.0f)
    float final_x = projected.x / w;
    float final_y = projected.y / w;
    float final_z = projected.z / w;

    // Points farther away should have smaller final coordinates
    EXPECT_LT(std::abs(final_x), 1.0f);
    EXPECT_LT(std::abs(final_y), 1.0f);

    // Test that closer points are larger
    vec3f closer_point{1.0f, 1.0f, -2.0f};
    vec3f closer_projected = persp_square * closer_point;
    float closer_w = 2.0f;
    float closer_final_x = closer_projected.x / closer_w;

    EXPECT_GT(std::abs(closer_final_x), std::abs(final_x));
}

TEST_F(PerspectiveProjectionTest, NearFarPlaneMapping) {
    // Test depth mapping at near and far planes

    // Point at near plane
    vec3f at_near{0.0f, 0.0f, -near_plane};
    vec3f proj_near = persp_matrix * at_near;

    // Point at far plane
    vec3f at_far{0.0f, 0.0f, -far_plane};
    vec3f proj_far = persp_matrix * at_far;

    // Near plane should map to NDC z = -1
    EXPECT_NEAR(proj_near.z, -1.0f, EPSILON * 10);

    // Far plane should map to NDC z = 1
    EXPECT_NEAR(proj_far.z, 1.0f, EPSILON * 10);
}

TEST_F(PerspectiveProjectionTest, CenterAxisPoints) {
    // Points on the center axis (x=0, y=0) should remain on axis after projection
    vec3f center_points[] = {
        {0.0f, 0.0f, -1.0f},
        {0.0f, 0.0f, -5.0f},
        {0.0f, 0.0f, -50.0f}
    };

    for (const auto& point : center_points) {
        vec3f projected = persp_matrix * point;
        float w = -point.z;
        float final_x = projected.x / w;
        float final_y = projected.y / w;

        EXPECT_NEAR(final_x, 0.0f, EPSILON);
        EXPECT_NEAR(final_y, 0.0f, EPSILON);
    }
}

TEST_F(PerspectiveProjectionTest, FieldOfViewEffects) {
    // Test different FOV values
    float fov_30 = radians(30.0f);
    float fov_60 = radians(60.0f);
    float fov_90 = radians(90.0f);

    mat4f persp_30 = perspective(fov_30, 1.0f, 1.0f, 10.0f);
    mat4f persp_60 = perspective(fov_60, 1.0f, 1.0f, 10.0f);
    mat4f persp_90 = perspective(fov_90, 1.0f, 1.0f, 10.0f);

    vec3f test_point{1.0f, 1.0f, -5.0f};

    vec3f proj_30 = persp_30 * test_point;
    vec3f proj_60 = persp_60 * test_point;
    vec3f proj_90 = persp_90 * test_point;

    float w = 5.0f;
    float final_30_x = proj_30.x / w;
    float final_60_x = proj_60.x / w;
    float final_90_x = proj_90.x / w;

    // Smaller FOV should produce larger projected coordinates (telephoto effect)
    EXPECT_GT(std::abs(final_30_x), std::abs(final_60_x));
    EXPECT_GT(std::abs(final_60_x), std::abs(final_90_x));
}

TEST_F(PerspectiveProjectionTest, AspectRatioEffects) {
    // Test different aspect ratios
    float aspect_1_1 = 1.0f;
    float aspect_4_3 = 4.0f / 3.0f;
    float aspect_16_9 = 16.0f / 9.0f;

    mat4f persp_1_1 = perspective(fov_45, aspect_1_1, 1.0f, 10.0f);
    mat4f persp_4_3 = perspective(fov_45, aspect_4_3, 1.0f, 10.0f);
    mat4f persp_16_9 = perspective(fov_45, aspect_16_9, 1.0f, 10.0f);

    vec3f test_point{2.0f, 1.0f, -5.0f};

    vec3f proj_1_1 = persp_1_1 * test_point;
    vec3f proj_4_3 = persp_4_3 * test_point;
    vec3f proj_16_9 = persp_16_9 * test_point;

    float w = 5.0f;
    float final_1_1_x = proj_1_1.x / w;
    float final_4_3_x = proj_4_3.x / w;
    float final_16_9_x = proj_16_9.x / w;

    // Higher aspect ratio should compress X coordinates
    EXPECT_GT(std::abs(final_1_1_x), std::abs(final_4_3_x));
    EXPECT_GT(std::abs(final_4_3_x), std::abs(final_16_9_x));

    // Y coordinates should be the same (aspect only affects X)
    float final_1_1_y = proj_1_1.y / w;
    float final_4_3_y = proj_4_3.y / w;
    float final_16_9_y = proj_16_9.y / w;

    EXPECT_NEAR(final_1_1_y, final_4_3_y, EPSILON);
    EXPECT_NEAR(final_4_3_y, final_16_9_y, EPSILON);
}

TEST_F(PerspectiveProjectionTest, DepthNonLinearity) {
    // Test that perspective projection creates non-linear depth distribution

    // Points at regular intervals
    vec3f points[] = {
        {0.0f, 0.0f, -4.0f},
        {0.0f, 0.0f, -6.0f},
        {0.0f, 0.0f, -8.0f},
        {0.0f, 0.0f, -10.0f}
    };

    float ndc_z_values[4];

    for (int i = 0; i < 4; ++i) {
        vec3f projected = persp_square * points[i];
        float w = -points[i].z;
        ndc_z_values[i] = projected.z / w;
    }

    // Calculate intervals between consecutive NDC Z values
    float interval1 = lina::abs(ndc_z_values[1] - ndc_z_values[0]);
    float interval2 = lina::abs(ndc_z_values[2] - ndc_z_values[1]);
    float interval3 = lina::abs(ndc_z_values[3] - ndc_z_values[2]);

    // Intervals should decrease (non-linear distribution)
    // More precision near the camera, less precision far away
    EXPECT_GT(interval1, interval2);
    EXPECT_GT(interval2, interval3);
}

TEST_F(PerspectiveProjectionTest, PerspectiveVsOrthographic) {
    // Compare perspective projection with orthographic
    mat4f ortho_proj = ortho(-2.0f, 2.0f, -2.0f, 2.0f, 1.0f, 10.0f);

    // Same points at different depths
    vec3f near_point{1.0f, 1.0f, -2.0f};
    vec3f far_point{1.0f, 1.0f, -8.0f};

    // Perspective projections
    vec3f persp_near = persp_square * near_point;
    vec3f persp_far = persp_square * far_point;

    // Orthographic projections
    vec3f ortho_near = ortho_proj * near_point;
    vec3f ortho_far = ortho_proj * far_point;

    // Apply perspective divide for perspective projection
    float persp_near_x = persp_near.x / 2.0f;
    float persp_far_x = persp_far.x / 8.0f;

    // In perspective: far objects should be smaller
    EXPECT_GT(std::abs(persp_near_x), std::abs(persp_far_x));

    // In orthographic: objects should be the same size
    EXPECT_NEAR(ortho_near.x, ortho_far.x, EPSILON);
}

TEST_F(PerspectiveProjectionTest, SymmetricFrustum) {
    // Test that our perspective matrix creates a symmetric frustum

    // Points at same distance but opposite sides
    vec3f left_point{-1.0f, 0.0f, -5.0f};
    vec3f right_point{1.0f, 0.0f, -5.0f};
    vec3f bottom_point{0.0f, -1.0f, -5.0f};
    vec3f top_point{0.0f, 1.0f, -5.0f};

    vec3f proj_left = persp_square * left_point;
    vec3f proj_right = persp_square * right_point;
    vec3f proj_bottom = persp_square * bottom_point;
    vec3f proj_top = persp_square * top_point;

    float w = 5.0f;

    // Should be symmetric around origin
    EXPECT_NEAR(proj_left.x / w, -(proj_right.x / w), EPSILON);
    EXPECT_NEAR(proj_bottom.y / w, -(proj_top.y / w), EPSILON);
}

TEST_F(PerspectiveProjectionTest, DoublePrecision) {
    // Test with double precision
    double fov_d = pi<double> / 4.0;
    double aspect_d = 16.0 / 9.0;
    double near_d = 0.1;
    double far_d = 1000.0;

    mat<double, 4, 4> persp_d = perspective(fov_d, aspect_d, near_d, far_d);

    double tan_half_fov = std::tan(fov_d / 2.0);

    // Test with higher precision
    EXPECT_NEAR(persp_d(0, 0), 1.0 / (aspect_d * tan_half_fov), EPSILON_D * 1000);
    EXPECT_NEAR(persp_d(1, 1), 1.0 / tan_half_fov, EPSILON_D * 1000);
    EXPECT_NEAR(persp_d(3, 2), -1.0, EPSILON_D);
}

TEST_F(PerspectiveProjectionTest, ExtremeFOVValues) {
    // Test with very small and very large FOV values
    float fov_small = radians(5.0f);   // Very narrow
    float fov_large = radians(160.0f); // Very wide

    mat4f persp_small = perspective(fov_small, 1.0f, 1.0f, 10.0f);
    mat4f persp_large = perspective(fov_large, 1.0f, 1.0f, 10.0f);

    // Both should produce valid matrices
    EXPECT_FALSE(std::isnan(persp_small(0, 0)));
    EXPECT_FALSE(std::isnan(persp_small(1, 1)));
    EXPECT_FALSE(std::isnan(persp_large(0, 0)));
    EXPECT_FALSE(std::isnan(persp_large(1, 1)));

    // Small FOV should have large scaling factors (telephoto effect)
    EXPECT_GT(persp_small(1, 1), persp_large(1, 1));
}

TEST_F(PerspectiveProjectionTest, ViewingVolumeSize) {
    // Test that the viewing volume size changes correctly with parameters

    vec3f edge_point{1.0f, 1.0f, -5.0f};  // Point near the edge

    // Wide FOV should include more of the scene (smaller final coordinates)
    mat4f wide_fov = perspective(radians(90.0f), 1.0f, 1.0f, 10.0f);
    mat4f narrow_fov = perspective(radians(30.0f), 1.0f, 1.0f, 10.0f);

    vec3f wide_proj = wide_fov * edge_point;
    vec3f narrow_proj = narrow_fov * edge_point;

    float w = 5.0f;
    float wide_final = wide_proj.x / w;
    float narrow_final = narrow_proj.x / w;

    // Wide FOV should produce smaller coordinates (fits more in view)
    EXPECT_LT(std::abs(wide_final), std::abs(narrow_final));
}

TEST_F(PerspectiveProjectionTest, ZeroWCoordinateHandling) {
    // Test points very close to the camera (potential division by very small numbers)
    vec3f very_close{1.0f, 1.0f, -0.01f};  // Very close to camera

    // This should still work and not produce infinite values
    vec3f projected = persp_matrix * very_close;

    EXPECT_FALSE(std::isinf(projected.x));
    EXPECT_FALSE(std::isinf(projected.y));
    EXPECT_FALSE(std::isinf(projected.z));
    EXPECT_FALSE(std::isnan(projected.x));
    EXPECT_FALSE(std::isnan(projected.y));
    EXPECT_FALSE(std::isnan(projected.z));
}