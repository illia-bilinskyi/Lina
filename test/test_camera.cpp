#include "lina/lina.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace lina;

constexpr float EPSILON = 1e-5f;
constexpr double EPSILON_D = 1e-12;

// ========================= look_at Function Tests =========================

class LookAtTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test vectors
        origin = vec3f{0.0f, 0.0f, 0.0f};
        x_axis = vec3f{1.0f, 0.0f, 0.0f};
        y_axis = vec3f{0.0f, 1.0f, 0.0f};
        z_axis = vec3f{0.0f, 0.0f, 1.0f};
        neg_z = vec3f{0.0f, 0.0f, -1.0f};
    }

    // Helper function to verify matrix orthogonality
    static bool is_orthogonal(const mat4f& matrix, float epsilon = EPSILON) {
        vec3f row0{matrix(0,0), matrix(0,1), matrix(0,2)};
        vec3f row1{matrix(1,0), matrix(1,1), matrix(1,2)};
        vec3f row2{matrix(2,0), matrix(2,1), matrix(2,2)};

        // Check if rows are orthogonal
        bool orthogonal = almost_zero(dot(row0, row1), epsilon) &&
                         almost_zero(dot(row0, row2), epsilon) &&
                         almost_zero(dot(row1, row2), epsilon);

        // Check if rows are normalized
        bool normalized = almost_equal(norm(row0), 1.0f, epsilon) &&
                         almost_equal(norm(row1), 1.0f, epsilon) &&
                         almost_equal(norm(row2), 1.0f, epsilon);

        return orthogonal && normalized;
    }

    // Helper to check if homogeneous row is correct
    static bool has_correct_homogeneous_row(const mat4f& matrix, float epsilon = EPSILON) {
        return almost_zero(matrix(3,0), epsilon) &&
               almost_zero(matrix(3,1), epsilon) &&
               almost_zero(matrix(3,2), epsilon) &&
               almost_equal(matrix(3,3), 1.0f, epsilon);
    }

    vec3f origin, x_axis, y_axis, z_axis, neg_z;
};

TEST_F(LookAtTest, IdentityConfiguration) {
    // Camera at origin looking down negative Z (standard OpenGL setup)
    vec3f eye{0.0f, 0.0f, 0.0f};
    vec3f center{0.0f, 0.0f, -1.0f};
    vec3f up{0.0f, 1.0f, 0.0f};

    mat4f view = look_at(eye, center, up);

    // Should be very close to identity matrix
    mat4f identity_matrix = identity<float, 4>();

    EXPECT_TRUE(almost_equal(view, identity_matrix, 1e-4f))
        << "look_at with standard configuration should produce near-identity matrix";

    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));
}

TEST_F(LookAtTest, CameraAlongPositiveZ) {
    // Camera behind origin looking toward origin
    vec3f eye{0.0f, 0.0f, 5.0f};
    vec3f center{0.0f, 0.0f, 0.0f};
    vec3f up{0.0f, 1.0f, 0.0f};

    mat4f view = look_at(eye, center, up);

    // Check that matrix is orthogonal
    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));

    // Forward vector should point toward negative Z
    vec3f forward = forward_vec(view);
    vec3f expected_forward = normalize(center - eye); // Should be (0, 0, -1)

    EXPECT_NEAR(forward.x, expected_forward.x, EPSILON);
    EXPECT_NEAR(forward.y, expected_forward.y, EPSILON);
    EXPECT_NEAR(forward.z, expected_forward.z, EPSILON);

    // Right vector should be positive X
    EXPECT_NEAR(view(0,0), 1.0f, EPSILON);  // Right X component
    EXPECT_NEAR(view(0,1), 0.0f, EPSILON);  // Right Y component
    EXPECT_NEAR(view(0,2), 0.0f, EPSILON);  // Right Z component

    // Up vector should be positive Y
    EXPECT_NEAR(view(1,0), 0.0f, EPSILON);  // Up X component
    EXPECT_NEAR(view(1,1), 1.0f, EPSILON);  // Up Y component
    EXPECT_NEAR(view(1,2), 0.0f, EPSILON);  // Up Z component
}

TEST_F(LookAtTest, CameraAlongPositiveX) {
    // Camera to the right, looking at origin
    vec3f eye{5.0f, 0.0f, 0.0f};
    vec3f center{0.0f, 0.0f, 0.0f};
    vec3f up{0.0f, 1.0f, 0.0f};

    mat4f view = look_at(eye, center, up);

    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));

    // Forward vector should point toward negative X
    vec3f forward = forward_vec(view);
    vec3f expected_forward = normalize(center - eye); // Should be (-1, 0, 0)

    EXPECT_NEAR(forward.x, expected_forward.x, EPSILON);
    EXPECT_NEAR(forward.y, expected_forward.y, EPSILON);
    EXPECT_NEAR(forward.z, expected_forward.z, EPSILON);

    // Right vector should point toward positive Z
    EXPECT_NEAR(view(2,0), 1.0f, EPSILON);

    // Up vector should remain Y
    EXPECT_NEAR(view(1,1), 1.0f, EPSILON);
}

TEST_F(LookAtTest, CameraAboveLookingDown) {
    // Camera above, looking down at origin
    vec3f eye{0.0f, 5.0f, 0.0f};
    vec3f center{0.0f, 0.0f, 0.0f};
    vec3f up{0.0f, 0.0f, -1.0f};  // Up is now negative Z

    mat4f view = look_at(eye, center, up);

    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));

    // Forward vector should point toward negative Y
    vec3f forward = forward_vec(view);

    EXPECT_NEAR(forward.x, 0.0f, EPSILON);
    EXPECT_NEAR(forward.y, -1.0f, EPSILON);
    EXPECT_NEAR(forward.z, 0.0f, EPSILON);
}

TEST_F(LookAtTest, ArbitraryCameraPosition) {
    // Camera at arbitrary position
    vec3f eye{3.0f, 4.0f, 5.0f};
    vec3f center{1.0f, 2.0f, 1.0f};
    vec3f up{0.0f, 1.0f, 0.0f};

    mat4f view = look_at(eye, center, up);

    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));

    // Check that forward direction is correct
    vec3f forward = forward_vec(view);
    vec3f expected_forward = normalize(center - eye);

    EXPECT_NEAR(forward.x, expected_forward.x, EPSILON);
    EXPECT_NEAR(forward.y, expected_forward.y, EPSILON);
    EXPECT_NEAR(forward.z, expected_forward.z, EPSILON);
}

TEST_F(LookAtTest, TransformationCorrectness) {
    // Test that the look_at matrix correctly transforms points
    vec3f eye{0.0f, 0.0f, 5.0f};
    vec3f center{0.0f, 0.0f, 0.0f};
    vec3f up{0.0f, 1.0f, 0.0f};

    mat4f view = look_at(eye, center, up);

    // A point at the center should map to (0, 0, -5) in view space
    vec3f world_center{0.0f, 0.0f, 0.0f};
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
    vec3f world_right{1.0f, 0.0f, 0.0f};
    vec3f view_right = view * world_right;
    EXPECT_GT(view_right.x, 0.0f);  // Should be positive
}

TEST_F(LookAtTest, UpVectorNormalization) {
    // Test with non-normalized up vector
    vec3f eye{0.0f, 0.0f, 5.0f};
    vec3f center{0.0f, 0.0f, 0.0f};
    vec3f up{0.0f, 3.0f, 4.0f};  // Length = 5, not normalized

    mat4f view = look_at(eye, center, up);

    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));

    // The resulting matrix should still be valid despite non-normalized input
    vec3f up_result{view(1,0), view(1,1), view(1,2)};
    EXPECT_NEAR(norm(up_result), 1.0f, EPSILON);
}

TEST_F(LookAtTest, NearParallelUpVector) {
    // Test case where up vector is nearly parallel to forward vector
    vec3f eye{0.0f, 0.0f, 5.0f};
    vec3f center{0.0f, 0.0f, 0.0f};
    vec3f up{0.1f, 0.0f, -1.0f};  // Nearly parallel to forward direction

    mat4f view = look_at(eye, center, up);

    // Should still produce orthogonal matrix (though up vector will be adjusted)
    EXPECT_TRUE(is_orthogonal(view));
    EXPECT_TRUE(has_correct_homogeneous_row(view));
}

TEST_F(LookAtTest, DoublesPrecision) {
    // Test with double precision
    vec3<double> eye_d{0.0, 0.0, 5.0};
    vec3<double> center_d{0.0, 0.0, 0.0};
    vec3<double> up_d{0.0, 1.0, 0.0};

    mat<double, 4, 4> view_d = look_at(eye_d, center_d, up_d);

    // Check orthogonality with higher precision
    vec3<double> row0{view_d(0,0), view_d(0,1), view_d(0,2)};
    vec3<double> row1{view_d(1,0), view_d(1,1), view_d(1,2)};
    vec3<double> row2{view_d(2,0), view_d(2,1), view_d(2,2)};

    EXPECT_NEAR(dot(row0, row1), 0.0, EPSILON_D);
    EXPECT_NEAR(dot(row0, row2), 0.0, EPSILON_D);
    EXPECT_NEAR(dot(row1, row2), 0.0, EPSILON_D);

    EXPECT_NEAR(norm(row0), 1.0, EPSILON_D);
    EXPECT_NEAR(norm(row1), 1.0, EPSILON_D);
    EXPECT_NEAR(norm(row2), 1.0, EPSILON_D);
}

TEST_F(LookAtTest, ConsistentWithManualCalculation) {
    // Test against manually calculated expected values
    vec3f eye{1.0f, 2.0f, 3.0f};
    vec3f center{4.0f, 5.0f, 6.0f};
    vec3f up{0.0f, 1.0f, 0.0f};

    // Manually calculate expected basis vectors
    vec3f forward = normalize(center - eye);  // (3,3,3) normalized = (√3/3, √3/3, √3/3)
    vec3f right = normalize(cross(forward, up));
    vec3f camera_up = cross(right, forward);

    mat4f view = look_at(eye, center, up);

    // Check basis vectors match
    EXPECT_NEAR(view(0,0), right.x, EPSILON);
    EXPECT_NEAR(view(0,1), right.y, EPSILON);
    EXPECT_NEAR(view(0,2), right.z, EPSILON);

    EXPECT_NEAR(view(1,0), camera_up.x, EPSILON);
    EXPECT_NEAR(view(1,1), camera_up.y, EPSILON);
    EXPECT_NEAR(view(1,2), camera_up.z, EPSILON);

    EXPECT_NEAR(view(2,0), -forward.x, EPSILON);
    EXPECT_NEAR(view(2,1), -forward.y, EPSILON);
    EXPECT_NEAR(view(2,2), -forward.z, EPSILON);

    // Check translation components
    EXPECT_NEAR(view(0,3), -dot(right, eye), EPSILON);
    EXPECT_NEAR(view(1,3), -dot(camera_up, eye), EPSILON);
    EXPECT_NEAR(view(2,3), dot(forward, eye), EPSILON);
}

TEST_F(LookAtTest, InvertibilityTest) {
    // Test that the look_at matrix can be conceptually inverted
    vec3f eye{2.0f, 3.0f, 4.0f};
    vec3f center{0.0f, 0.0f, 0.0f};
    vec3f up{0.0f, 1.0f, 0.0f};

    mat4f view = look_at(eye, center, up);

    // The eye position transformed to view space should be at origin
    vec3f transformed_eye = view * eye;

    EXPECT_NEAR(transformed_eye.x, 0.0f, EPSILON);
    EXPECT_NEAR(transformed_eye.y, 0.0f, EPSILON);
    EXPECT_NEAR(transformed_eye.z, 0.0f, EPSILON);
}

TEST_F(LookAtTest, NonUnitUpVectorHandling) {
    // Test various non-unit up vectors
    vec3f eye{0.0f, 0.0f, 5.0f};
    vec3f center{0.0f, 0.0f, 0.0f};

    // Test very small up vector
    vec3f tiny_up{0.0f, 0.001f, 0.0f};
    mat4f view1 = look_at(eye, center, tiny_up);
    EXPECT_TRUE(is_orthogonal(view1));

    // Test very large up vector
    vec3f large_up{0.0f, 1000.0f, 0.0f};
    mat4f view2 = look_at(eye, center, large_up);
    EXPECT_TRUE(is_orthogonal(view2));

    // Both should produce similar results (after normalization)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(view1(i,j), view2(i,j), EPSILON * 10);
        }
    }
}

// Performance/stress test
TEST_F(LookAtTest, StressTest) {
    // Test with many different configurations to ensure robustness
    constexpr int num_tests = 100;

    for (int i = 0; i < num_tests; ++i) {
        float angle = 2.0f * pi<float> * i / num_tests;

        vec3f eye{5.0f * std::cos(angle), 2.0f, 5.0f * std::sin(angle)};
        vec3f center{0.0f, 0.0f, 0.0f};
        vec3f up{0.0f, 1.0f, 0.0f};

        mat4f view = look_at(eye, center, up);

        // Each matrix should be orthogonal
        EXPECT_TRUE(is_orthogonal(view));
        EXPECT_TRUE(has_correct_homogeneous_row(view));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}