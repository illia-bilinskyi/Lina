#include "lina/lina.h"
#include <iostream>

// Test compilation of all library functions
// This test ensures all functions compile correctly across different standards and platforms

int main()
{
    using namespace lina;
    
    // ========================= Basic Types Test =========================
    
    // Matrix types
    mat2f m2f;
    mat3f m3f; 
    mat4f m4f;
    mat2d m2d;
    mat3d m3d;
    mat4d m4d;
    
    // Vector types
    vec3f v3f;
    vec3d v3d;
    
    // Generic matrix
    mat<float, 2, 3> m23f;
    mat<double, 3, 2> m32d;
    
    // ========================= Matrix Constructors =========================
    
    // Default constructor
    mat3f m_default;
    
    // Fill constructor
    mat3f m_fill(2.5f);
    
    // Variadic constructor
    mat2f m_variadic(1.0f, 2.0f, 3.0f, 4.0f);
    
    // Initializer list constructor (matrix of row matrices)
    mat3f m_init_list{
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };
    
    // Copy constructor
    mat3f m_copy(m3f);
    
    // Square matrix copy constructor
    mat4f m4_from_m3(mat3f{});
    
    // Type conversion constructor
    mat3d m3d_from_m3f(m3f);
    
    // ========================= Vector Constructors =========================
    
    // Default constructor
    vec3f v_default;
    
    // Three parameter constructor
    vec3f v_params(1.0f, 2.0f, 3.0f);
    
    // Fill constructor
    vec3f v_fill(2.5f);
    
    // Initializer list constructor
    vec3f v_init_list{1.0f, 2.0f, 3.0f};
    
    // Copy constructor
    vec3f v_copy(v3f);
    
    // Type conversion constructor
    vec3d v3d_from_v3f(v3f);
    
    // Matrix to vector conversion
    vec3f v_from_mat3(mat<float, 3, 1>{1.0f, 2.0f, 3.0f});
    vec3f v_from_mat1(mat<float, 1, 3>{1.0f, 2.0f, 3.0f});
    
    // Vector to matrix conversion
    auto mat31_from_v = static_cast<mat<float, 3, 1>>(v3f);
    auto mat13_from_v = static_cast<mat<float, 1, 3>>(v3f);
    
    // ========================= Matrix Access Methods =========================
    
    // Element access
    float elem1 = m3f(0, 1);
    m3f(1, 2) = 5.0f;
    
    // Linear access
    float elem2 = m3f[3];
    m3f[4] = 6.0f;
    
    // Single index access for vectors
    mat<float, 1, 3> row_vec;
    float row_elem = row_vec(1);
    row_vec(2) = 7.0f;
    
    // Template access
    float templated_elem = m3f.get<1, 2>();
    m3f.get<0, 0>() = 8.0f;
    
    // Data access
    float* mat_data = m3f.data();
    const float* const_mat_data = static_cast<const mat3f&>(m3f).data();
    
    // Row and column extraction
    auto row0 = m3f.row(0);
    auto col1 = m3f.col(1);
    auto templated_row = m3f.row<1>();
    auto templated_col = m3f.col<2>();
    
    // ========================= Vector Access Methods =========================
    
    // Named access
    float x = v3f.x;
    float y = v3f.y;
    float z = v3f.z;
    v3f.x = 1.0f;
    v3f.y = 2.0f;
    v3f.z = 3.0f;
    
    // Index access
    float v_elem1 = v3f[0];
    v3f[1] = 4.0f;
    
    // Template access
    float v_templated = v3f.get<2>();
    v3f.get<0>() = 5.0f;
    
    // Data access
    float* vec_data = v3f.data();
    const float* const_vec_data = static_cast<const vec3f&>(v3f).data();
    
    // ========================= Matrix Operators =========================
    
    // Unary operators
    mat3f m_positive = +m3f;
    mat3f m_negative = -m3f;
    
    // Scalar operations
    mat3f m_scaled = m3f * 2.0f;
    mat3f m_divided = m3f / 2.0f;
    m3f *= 3.0f;
    m3f /= 1.5f;
    
    // Matrix elementwise operations (same type)
    mat3f m3f_copy = mat3f{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    mat3f m_added = m3f + m3f_copy;
    mat3f m_subtracted = m3f - m3f_copy;
    m3f += m3f_copy;
    m3f -= m3f_copy;
    
    // Matrix multiplication (same type)
    mat3f m_multiplied = m3f * m3f_copy;
    mat<float, 3, 2> m_mult_rect = m3f * mat<float, 3, 2>{};
    
    // ========================= Vector Operators =========================
    
    // Unary operators
    vec3f v_positive = +v3f;
    vec3f v_negative = -v3f;
    
    // Vector-vector operations (same type)
    vec3f v3f_copy = vec3f{1.0f, 2.0f, 3.0f};
    vec3f v_added = v3f + v3f_copy;
    vec3f v_subtracted = v3f - v3f_copy;
    
    // Vector-scalar operations
    vec3f v_scaled = v3f * 2.0f;
    vec3f v_divided = v3f / 2.0f;
    
    // Assignment operations (same type)
    v3f += v3f_copy;
    v3f -= v3f_copy;
    v3f *= 2.5f;
    v3f /= 1.5f;
    
    // ========================= Constexpr Math Functions =========================
    
    // Absolute value
    float abs_result = abs(-5.0f);
    double abs_result_d = abs(-5.0);
    
    // Trigonometric functions
    float sin_result = sin(pi<float> / 6.0f);
    float cos_result = cos(pi<float> / 6.0f);
    float tan_result = tan(pi<float> / 4.0f);
    
    double sin_result_d = sin(pi<double> / 6.0);
    double cos_result_d = cos(pi<double> / 6.0);
    double tan_result_d = tan(pi<double> / 4.0);
    
    // Square root
    float sqrt_result = sqrt(4.0f);
    double sqrt_result_d = sqrt(4.0);
    
    // Constants access
    float pi_f = pi<float>;
    double pi_d = pi<double>;
    float eps_f = COMPARE_EPSILON_DEFAULT<float>;
    double eps_d = COMPARE_EPSILON_DEFAULT<double>;
    
    // ========================= Comparison Functions =========================
    
    // Scalar comparisons
    bool almost_eq_scalar = almost_equal(1.0f, 1.000001f);
    bool almost_eq_scalar_custom = almost_equal(1.0, 1.000001, 1e-5);
    bool almost_zero_scalar = almost_zero(1e-8f);
    bool almost_zero_scalar_custom = almost_zero(1e-8, 1e-7);
    
    // Vector comparisons (same type)
    bool almost_eq_vec = almost_equal(v3f, v3f_copy);
    bool almost_eq_vec_custom = almost_equal(v3f, v3f_copy, 1e-5f);
    bool almost_zero_vec = almost_zero(vec3f{1e-8f, 1e-9f, 1e-7f});
    bool almost_zero_vec_custom = almost_zero(vec3f{1e-8f, 1e-9f, 1e-7f}, 1e-6f);
    
    // Matrix comparisons (same type)
    bool almost_eq_mat = almost_equal(m3f, m3f_copy);
    bool almost_eq_mat_custom = almost_equal(m3f, m3f_copy, 1e-5f);
    bool almost_zero_mat = almost_zero(mat3f{});
    bool almost_zero_mat_custom = almost_zero(mat3f{}, 1e-7f);
    
    // ========================= Matrix Operations =========================
    
    // Identity matrices
    mat2f identity2 = identity<float, 2>();
    mat3f identity3 = identity<float, 3>();
    mat4f identity4 = identity<float, 4>();
    mat2d identity2d = identity<double, 2>();
    mat3d identity3d = identity<double, 3>();
    mat4d identity4d = identity<double, 4>();
    
    // Transpose
    mat<float, 3, 2> m32_transposed = transpose(m23f);
    mat2f m2_transposed = transpose(m2f);
    mat3f m3_transposed = transpose(m3f);
    mat4f m4_transposed = transpose(m4f);
    
    // Determinant
    float det2 = det(m2f);
    float det3 = det(m3f);
    float det4 = det(m4f);
    double det2d = det(m2d);
    double det3d = det(m3d);
    double det4d = det(m4d);
    
    // Constexpr inverse (no exception throwing)
    mat2f inv2_constexpr = c_inverse(mat2f{1, 2, 3, 5});
    mat3f inv3_constexpr = c_inverse(mat3f{1, 0, 0, 0, 1, 0, 0, 0, 1});
    mat4f inv4_constexpr = c_inverse(identity<float, 4>());
    
    // Runtime inverse (with validation) - using try-catch to avoid exceptions in build test
    try {
        mat2f inv2_runtime = inverse(mat2f{1, 2, 3, 5});
        mat3f inv3_runtime = inverse(identity<float, 3>());
        mat4f inv4_runtime = inverse(identity<float, 4>());
        (void)inv2_runtime; (void)inv3_runtime; (void)inv4_runtime;
    } catch (...) {
        // Handle potential exceptions gracefully
    }
    
    // Helper determinant function
    float det3x3_helper = det3x3(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
    
    // ========================= Vector Operations =========================
    
    vec3f test_v1{1.0f, 2.0f, 3.0f};
    vec3f test_v2{4.0f, 5.0f, 6.0f};
    
    // Dot product
    float dot_result = dot(test_v1, test_v2);
    double dot_result_d = dot(vec3d{1.0, 2.0, 3.0}, vec3d{4.0, 5.0, 6.0});
    
    // Cross product
    vec3f cross_result = cross(test_v1, test_v2);
    vec3d cross_result_d = cross(vec3d{1.0, 2.0, 3.0}, vec3d{4.0, 5.0, 6.0});
    
    // Norm operations
    float norm2_result = norm2(test_v1);
    float norm_result = norm(test_v1);
    float length_result = length(test_v1);
    double norm2_result_d = norm2(vec3d{3.0, 4.0, 0.0});
    double norm_result_d = norm(vec3d{3.0, 4.0, 0.0});
    double length_result_d = length(vec3d{3.0, 4.0, 0.0});
    
    // Normalize
    vec3f normalized = normalize(test_v1);
    vec3d normalized_d = normalize(vec3d{3.0, 4.0, 5.0});
    
    // Distance
    float distance_result = distance(test_v1, test_v2);
    double distance_result_d = distance(vec3d{1.0, 2.0, 3.0}, vec3d{4.0, 5.0, 6.0});
    
    // ========================= Vector-Matrix Operations =========================
    
    // Homogeneous conversions
    mat<float, 1, 4> homog_point = to_homogeneous(test_v1, 1.0f);
    mat<float, 1, 4> homog_vector = to_homogeneous(test_v1, 0.0f);
    vec3f from_homog = from_homogeneous(homog_point);
    
    mat<double, 1, 4> homog_point_d = to_homogeneous(vec3d{1.0, 2.0, 3.0}, 1.0);
    mat<double, 1, 4> homog_vector_d = to_homogeneous(vec3d{1.0, 2.0, 3.0}, 0.0);
    vec3d from_homog_d = from_homogeneous(homog_point_d);
    
    // Matrix-vector multiplication
    vec3f mat3_vec_mult = m3f * test_v1;
    vec3f mat4_vec_mult = m4f * test_v1;  // Homogeneous
    vec3d mat3_vec_mult_d = m3d * vec3d{1.0, 2.0, 3.0};
    vec3d mat4_vec_mult_d = m4d * vec3d{1.0, 2.0, 3.0};
    
    // Vector-matrix multiplication
    vec3f vec_mat3_mult = test_v1 * m3f;
    vec3f vec_mat4_mult = test_v1 * m4f;
    vec3d vec_mat3_mult_d = vec3d{1.0, 2.0, 3.0} * m3d;
    vec3d vec_mat4_mult_d = vec3d{1.0, 2.0, 3.0} * m4d;
    
    // ========================= 3D Transformations =========================
    
    vec3f trans_vec{1.0f, 2.0f, 3.0f};
    vec3f scale_vec{2.0f, 2.0f, 2.0f};
    float angle = pi<float> / 4.0f;
    
    // Basic transformations
    mat4f translation_mat = translation(trans_vec);
    mat4f scale_mat = scale(scale_vec);
    mat4f rotation_from_mat3 = rotation(identity<float, 3>());
    
    mat4d translation_mat_d = translation(vec3d{1.0, 2.0, 3.0});
    mat4d scale_mat_d = scale(vec3d{2.0, 2.0, 2.0});
    mat4d rotation_from_mat3_d = rotation(identity<double, 3>());
    
    // Axis rotations
    mat4f rot_x = rotation_x(angle);
    mat4f rot_y = rotation_y(angle);
    mat4f rot_z = rotation_z(angle);
    mat4d rot_x_d = rotation_x(pi<double> / 4.0);
    mat4d rot_y_d = rotation_y(pi<double> / 4.0);
    mat4d rot_z_d = rotation_z(pi<double> / 4.0);
    
    // Arbitrary axis rotation
    vec3f axis{0.0f, 1.0f, 0.0f};
    mat4f rot_arbitrary = rotation(axis, angle);
    mat4d rot_arbitrary_d = rotation(vec3d{0.0, 1.0, 0.0}, pi<double> / 4.0);
    
    // Euler rotation
    mat4f euler_rot = rotation(angle, angle / 2.0f, angle / 3.0f);
    mat4d euler_rot_d = rotation(pi<double> / 4.0, pi<double> / 6.0, pi<double> / 8.0);
    
    // Combined transformation
    mat3f rot3x3 = identity<float, 3>();
    mat4f combined_transform = transform(trans_vec, rot3x3, scale_vec);
    mat4d combined_transform_d = transform(vec3d{1.0, 2.0, 3.0}, identity<double, 3>(), vec3d{2.0, 2.0, 2.0});
    
    // Rodrigues rotation
    vec3f rotated_vec = rotate(test_v1, axis, angle);
    vec3d rotated_vec_d = rotate(vec3d{1.0, 2.0, 3.0}, vec3d{0.0, 1.0, 0.0}, pi<double> / 4.0);
    
    // ========================= Transform Analysis =========================
    
    mat4f test_transform = translation(vec3f{1.0f, 2.0f, 3.0f}) * rotation_x(angle) * scale(vec3f{2.0f, 2.0f, 2.0f});
    
    // Validation functions
    bool rot_valid = is_rotation_valid(identity<float, 3>());
    bool scale_valid = is_scale_valid(scale_vec);
    bool rot_valid_d = is_rotation_valid(identity<double, 3>());
    bool scale_valid_d = is_scale_valid(vec3d{2.0, 2.0, 2.0});
    
    // Extraction functions
    vec3f extracted_translation = get_translation(test_transform);
    vec3f extracted_scale = get_scale(test_transform);
    mat3f extracted_rotation_c = c_get_rotation(test_transform);  // Constexpr version
    
    vec3d extracted_translation_d = get_translation(mat4d{});
    vec3d extracted_scale_d = get_scale(mat4d{});
    mat3d extracted_rotation_c_d = c_get_rotation(mat4d{});
    
    // Runtime extraction (with validation)
    try {
        mat3f extracted_rotation = get_rotation(test_transform);
        (void)extracted_rotation;
    } catch (...) {
        // Handle potential exceptions
    }
    
    // Decomposition
    vec3f decomp_t, decomp_s;
    mat3f decomp_r;
    try {
        decompose(test_transform, decomp_t, decomp_r, decomp_s);
    } catch (...) {
        // Handle potential exceptions
    }
    
    // Inverse transforms
    mat4f inv_transform_c = c_inverse_transform(test_transform);  // Constexpr version
    try {
        mat4f inv_transform = inverse_transform(test_transform);   // Runtime version
        (void)inv_transform;
    } catch (...) {
        // Handle potential exceptions
    }
    
    // ========================= Camera Operations =========================
    
    vec3f eye{0.0f, 0.0f, 5.0f};
    vec3f center{0.0f, 0.0f, 0.0f};
    vec3f up{0.0f, 1.0f, 0.0f};
    
    // Look-at matrix
    mat4f view_matrix = look_at(eye, center, up);
    mat4d view_matrix_d = look_at(vec3d{0.0, 0.0, 5.0}, vec3d{0.0, 0.0, 0.0}, vec3d{0.0, 1.0, 0.0});
    
    // Direction vectors from view matrix
    vec3f right_vector = right_vec(view_matrix);
    vec3f up_vector = up_vec(view_matrix);
    vec3f forward_vector = forward_vec(view_matrix);
    vec3f translation_vector = translation_vec(view_matrix);
    
    vec3d right_vector_d = right_vec(view_matrix_d);
    vec3d up_vector_d = up_vec(view_matrix_d);
    vec3d forward_vector_d = forward_vec(view_matrix_d);
    vec3d translation_vector_d = translation_vec(view_matrix_d);
    
    // Projection matrices
    float fovy = pi<float> / 4.0f;
    float aspect = 16.0f / 9.0f;
    float near_plane = 0.1f;
    float far_plane = 100.0f;
    
    mat4f perspective_mat = perspective(fovy, aspect, near_plane, far_plane);
    mat4f orthographic_mat = ortho(-10.0f, 10.0f, -10.0f, 10.0f, near_plane, far_plane);
    
    mat4d perspective_mat_d = perspective(pi<double> / 4.0, 16.0 / 9.0, 0.1, 100.0);
    mat4d orthographic_mat_d = ortho(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
    
    // Utility functions
    float degrees_to_radians = radians(45.0f);
    double degrees_to_radians_d = radians(45.0);
    
    // ========================= String Operations =========================
    
    // Matrix to string
    std::string mat_str = m3f.str();
    
    // Vector to string  
    std::string vec_str = v3f.str();
    
    // ========================= Prevent Optimization =========================
    
    // Use volatile to prevent complete optimization away
    volatile float optimization_preventer = 
        elem1 + elem2 + abs_result + sin_result + sqrt_result + 
        dot_result + norm_result + det2 + distance_result + 
        degrees_to_radians + static_cast<float>(almost_eq_scalar);
    
    volatile double optimization_preventer_d = 
        abs_result_d + sin_result_d + sqrt_result_d + dot_result_d + 
        norm_result_d + det2d + distance_result_d + degrees_to_radians_d + 
        static_cast<double>(almost_eq_vec);
    
    // Prevent unused variable warnings
    (void)optimization_preventer;
    (void)optimization_preventer_d;
    (void)mat_str;
    (void)vec_str;
    
    // ========================= Success Output =========================
    
    std::cout << "✅ All Lina library functions compiled successfully!" << std::endl;
    std::cout << "📊 Functions tested include:" << std::endl;
    std::cout << "   • Matrix and vector constructors & operators" << std::endl;
    std::cout << "   • Mathematical functions (sin, cos, sqrt, etc.)" << std::endl;
    std::cout << "   • Matrix operations (transpose, determinant, inverse)" << std::endl;
    std::cout << "   • Vector operations (dot, cross, normalize)" << std::endl;
    std::cout << "   • 3D transformations and camera operations" << std::endl;
    std::cout << "   • Comparison and utility functions" << std::endl;
    std::cout << "🚀 Library is ready for use!" << std::endl;
    
    return 0;
}