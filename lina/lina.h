#pragma once
#include <cmath>
#include <exception>
#include <initializer_list>
#include <stdexcept>

namespace lina
{
template <typename T, std::size_t R, std::size_t C>
struct mat
{
    static_assert(std::is_arithmetic_v<T>, "Matrix supports only arithmetic types.");

    T a[R * C]{ 0 };

    static constexpr std::size_t rows = R;
    static constexpr std::size_t cols = C;
    static constexpr std::size_t size = rows * cols;

    constexpr mat() = default;

    // Fill constructor
    constexpr explicit mat(T fill)
    {
        for (std::size_t i = 0; i < R; i++)
            for (std::size_t j = 0; j < C; j++)
                operator()(i, j) = fill;
    }

    template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == R * C>>
    constexpr mat(Args... args)
    {
        std::size_t i = 0;
        ((operator[](i++) = args), ...);
    }

    constexpr mat(std::initializer_list<mat<T, 1, C>> rows)
    {
        std::size_t i = 0;
        for (auto row : rows)
        {
            for (std::size_t j = 0; j < C; j++)
                operator()(i, j) = row(j);
            i++;
        }
    }

    // Copy constructor for square matrices
    template <std::size_t N>
    constexpr explicit mat(const mat<T, N, N>& m)
    {
        static_assert(R == C);
        const std::size_t minSize = (R < N) ? R : N;
        // Copy overlapping part
        for (std::size_t i = 0; i < minSize; i++)
            for (std::size_t j = 0; j < minSize; j++)
                operator()(i, j) = m(i, j);
    }

    template <typename T2>
    constexpr explicit mat(const mat<T2, R, C>& m)
    {
        for (std::size_t i = 0; i < R; i++)
            for (std::size_t j = 0; j < C; j++)
                operator()(i, j) = static_cast<T>(m(i, j));
    }

    template <std::size_t r, std::size_t c>
    constexpr T& get()
    {
        static_assert(r < R, "Row index out of bounds");
        static_assert(c < C, "Column index out of bounds");
        return operator()(r, c);
    }

    template <std::size_t r, std::size_t c>
    constexpr const T& get() const
    {
        static_assert(r < R, "Row index out of bounds");
        static_assert(c < C, "Column index out of bounds");
        return operator()(r, c);
    }

    constexpr T* data() { return a; }
    constexpr const T* data() const { return a; }

    constexpr T& operator()(std::size_t r, std::size_t c) { return a[c + r * C]; }
    constexpr const T& operator()(std::size_t r, std::size_t c) const { return a[c + r * C]; }
    constexpr T& operator[](std::size_t i) { return a[i]; }
    constexpr const T& operator[](std::size_t i) const { return a[i]; }

    constexpr T& operator()(std::size_t i)
    {
        static_assert(R == 1 || C == 1);
        return operator[](i);
    }

    constexpr const T& operator()(std::size_t i) const
    {
        static_assert(R == 1 || C == 1);
        return operator[](i);
    }

    constexpr mat<T, R, 1> col(std::size_t c) const
    {
        mat<T, R, 1> result{};
        for (std::size_t i = 0; i < R; i++)
            result(i) = operator()(i, c);
        return result;
    }

    constexpr mat<T, 1, C> row(std::size_t r) const
    {
        mat<T, 1, C> result{};
        for (std::size_t i = 0; i < C; i++)
            result(i) = operator()(r, i);
        return result;
    }

    template <std::size_t r>
    constexpr mat<T, 1, C> row() const
    {
        static_assert(r < R, "Row index out of bounds");
        return row(r);
    }

    template <std::size_t c>
    constexpr mat<T, R, 1> col() const
    {
        static_assert(c < C, "Column index out of bounds");
        return col(c);
    }

    // ===== Unary operators =====

    constexpr mat operator+() const { return *this; }

    constexpr mat operator-() const
    {
        mat res{};
        for (std::size_t i = 0; i < R; i++)
            for (std::size_t j = 0; j < C; j++)
                res(i, j) = -operator()(i, j);
        return res;
    }

    // ===== Scalar ops =====

    constexpr mat operator*(T s) const
    {
        mat res{};
        for (std::size_t i = 0; i < R; i++)
            for (std::size_t j = 0; j < C; j++)
                res(i, j) = operator()(i, j) * s;
        return res;
    }

    constexpr mat operator/(T s) const { return *this * (T{ 1 } / s); }
    constexpr mat& operator*=(T s) { return *this = *this * s; }
    constexpr mat& operator/=(T s) { return *this = *this / s; }

    // ===== Matrix elementwise ops =====

    constexpr mat operator+(const mat& other) const
    {
        mat res{};
        for (std::size_t i = 0; i < R; i++)
            for (std::size_t j = 0; j < C; j++)
                res(i, j) = operator()(i, j) + other(i, j);
        return res;
    }

    constexpr mat operator-(const mat& other) const { return *this + (-other); }
    constexpr mat& operator+=(const mat& other) { return *this = *this + other; }
    constexpr mat& operator-=(const mat& other) { return *this += -other; }

    // ===== Matrix multiplication =====

    template <std::size_t K>
    constexpr mat<T, R, K> operator*(const mat<T, C, K>& B) const
    {
        mat<T, R, K> res{};
        for (std::size_t i = 0; i < R; i++)
        {
            for (std::size_t j = 0; j < K; j++)
            {
                T sum{};
                for (std::size_t k = 0; k < C; k++)
                    sum += (*this)(i, k) * B(k, j);
                res(i, j) = sum;
            }
        }
        return res;
    }
};

template <typename T>
struct vec3
{
    static_assert(std::is_arithmetic_v<T>, "Vector supports only arithmetic types.");

    T x, y, z;

    constexpr vec3()
        : x{ 0 }
        , y{ 0 }
        , z{ 0 }
    {}

    constexpr vec3(T x_, T y_, T z_)
        : x{ x_ }
        , y{ y_ }
        , z{ z_ }
    {}

    // Fill constructor
    constexpr explicit vec3(T value)
        : x{ value }
        , y{ value }
        , z{ value }
    {}

    constexpr vec3(std::initializer_list<T> init)
        : vec3()
    {
        auto it = init.begin();
        x       = (it != init.end()) ? *it++ : T{ 0 };
        y       = (it != init.end()) ? *it++ : T{ 0 };
        z       = (it != init.end()) ? *it : T{ 0 };
    }

    // Copy constructor
    constexpr vec3(const vec3&) = default; // let compiler handle constexpr correctly

    // Assignment operator
    constexpr vec3& operator=(const vec3& other) = default;

    // Conversion operator into matrix
    constexpr explicit operator mat<T, 3, 1>() const { return mat<T, 3, 1>{ x, y, z }; }
    constexpr explicit operator mat<T, 1, 3>() const { return mat<T, 1, 3>{ x, y, z }; }

    constexpr explicit vec3(const mat<T, 1, 3>& m)
        : vec3(m(0), m(1), m(2))
    {}

    constexpr explicit vec3(const mat<T, 3, 1>& m)
        : vec3(m(0), m(1), m(2))
    {}

    template <typename T2>
    constexpr vec3(const vec3<T2>& other)
        : x{ static_cast<T>(other.x) }
        , y{ static_cast<T>(other.y) }
        , z{ static_cast<T>(other.z) }
    {}

    T* data() { return &x; }
    const T* data() const { return &x; }

    template <std::size_t i>
    constexpr T& get()
    {
        static_assert(i < 3, "Index out of bounds");
        return data()[i];
    }

    template <std::size_t i>
    constexpr const T& get() const
    {
        static_assert(i < 3, "Index out of bounds");
        return data()[i];
    }

    constexpr T& operator[](std::size_t i) { return *(&x + i); }
    constexpr const T& operator[](const std::size_t i) const { return *(&x + i); }

    // ===== Unary Arithmetic Operations =====

    constexpr vec3 operator+() const { return *this; }
    constexpr vec3 operator-() const { return { -x, -y, -z }; }

    // ===== Binary Arithmetic Operations (vector + vector) =====

    constexpr vec3 operator+(const vec3& other) const { return { x + other.x, y + other.y, z + other.z }; }
    constexpr vec3 operator-(const vec3& other) const { return *this + (-other); }

    // ===== Binary Arithmetic Operations (vector + scalar) =====

    constexpr vec3 operator*(T scalar) const { return { x * scalar, y * scalar, z * scalar }; }
    constexpr vec3 operator/(T scalar) const { return *this * (T{ 1 } / scalar); }

    // ===== Compound Assignment Operations (vector + vector) =====

    vec3& operator+=(const vec3& other) { return *this = *this + other; }
    vec3& operator-=(const vec3& other) { return *this += -other; }

    // ===== Compound Assignment Operations (vector + scalar) =====

    vec3& operator*=(T scalar) { return *this = *this * scalar; }
    vec3& operator/=(T scalar) { return *this = *this / scalar; }
};

// ========================= Aliases =========================

using vec3f = vec3<float>;
using vec3d = vec3<double>;

template <typename T>
using mat2 = mat<T, 2, 2>;
template <typename T>
using mat3 = mat<T, 3, 3>;
template <typename T>
using mat4 = mat<T, 4, 4>;

using mat2f = mat2<float>;
using mat3f = mat3<float>;
using mat4f = mat4<float>;

using mat2d = mat2<double>;
using mat3d = mat3<double>;
using mat4d = mat4<double>;

// ========================= Constants =========================

template <typename T>
constexpr T COMPARE_EPSILON_DEFAULT = static_cast<T>(1e-6);

template <typename T>
constexpr T pi = static_cast<T>(3.141592653589793L);

// ========================= Comparison =========================

template <typename T>
constexpr std::enable_if_t<std::is_arithmetic_v<T>, T> abs(T x) noexcept
{
    return x < T(0) ? -x : x;
}

template <typename T>
constexpr bool almost_equal(T a, T b, T eps = COMPARE_EPSILON_DEFAULT<T>)
{
    return abs(a - b) <= eps;
}

template <typename T, std::size_t R, std::size_t C>
constexpr bool almost_equal(const mat<T, R, C>& A, const mat<T, R, C>& B, T eps = COMPARE_EPSILON_DEFAULT<T>)
{
    for (std::size_t i = 0; i < R * C; i++)
    {
        if (!almost_equal(A[i], B[i], eps))
            return false;
    }
    return true;
}

template <typename T>
constexpr bool almost_equal(const vec3<T>& a, const vec3<T>& b, T eps = COMPARE_EPSILON_DEFAULT<T>)
{
    for (std::size_t i = 0; i < 3; i++)
    {
        if (!almost_equal(a[i], b[i], eps))
            return false;
    }
    return true;
}

template <typename T>
constexpr bool almost_zero(T a, T eps = COMPARE_EPSILON_DEFAULT<T>)
{
    return almost_equal(a, T(0), eps);
}

template <typename T>
constexpr bool almost_zero(const vec3<T>& a, T eps = COMPARE_EPSILON_DEFAULT<T>)
{
    return almost_equal(a, vec3<T>{}, eps);
}

template <typename T, std::size_t R, std::size_t C>
constexpr bool almost_zero(const mat<T, R, C>& a, T eps = COMPARE_EPSILON_DEFAULT<T>)
{
    return almost_equal(a, mat<T, R, C>{}, eps);
}

// ========================= Matrix Operations =========================

/**
 * Create an identity matrix
 * @tparam T
 * @tparam N
 * @return
 */
template <typename T, std::size_t N>
constexpr mat<T, N, N> identity()
{
    mat<T, N, N> I{};
    for (std::size_t i = 0; i < N; i++)
        I(i, i) = T(1);
    return I;
}

template <typename T, std::size_t R, std::size_t C>
constexpr mat<T, C, R> transpose(const mat<T, R, C>& M)
{
    mat<T, C, R> res{};
    for (std::size_t i = 0; i < R; i++)
        for (std::size_t j = 0; j < C; j++)
            res(j, i) = M(i, j);
    return res;
}

// Determinant for 2x2
template <typename T>
constexpr T det(const mat<T, 2, 2>& M)
{
    return M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);
}

// Determinant for 3x3
template <typename T>
constexpr T det(const mat<T, 3, 3>& M)
{
    T a1 = M(0, 0) * (M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1));
    T a2 = M(0, 1) * (M(1, 0) * M(2, 2) - M(1, 2) * M(2, 0));
    T a3 = M(0, 2) * (M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0));
    return a1 - a2 + a3;
}

// Determinant for 4x4
template <typename T>
constexpr T det(const mat<T, 4, 4>& M)
{
    T subfactor0 = M(2, 2) * M(3, 3) - M(2, 3) * M(3, 2);
    T subfactor1 = M(2, 1) * M(3, 3) - M(2, 3) * M(3, 1);
    T subfactor2 = M(2, 1) * M(3, 2) - M(2, 2) * M(3, 1);
    T subfactor3 = M(2, 0) * M(3, 3) - M(2, 3) * M(3, 0);
    T subfactor4 = M(2, 0) * M(3, 2) - M(2, 2) * M(3, 0);
    T subfactor5 = M(2, 0) * M(3, 1) - M(2, 1) * M(3, 0);

    return M(0, 0) * (M(1, 1) * subfactor0 - M(1, 2) * subfactor1 + M(1, 3) * subfactor2) -
           M(0, 1) * (M(1, 0) * subfactor0 - M(1, 2) * subfactor3 + M(1, 3) * subfactor4) +
           M(0, 2) * (M(1, 0) * subfactor1 - M(1, 1) * subfactor3 + M(1, 3) * subfactor5) -
           M(0, 3) * (M(1, 0) * subfactor2 - M(1, 1) * subfactor4 + M(1, 2) * subfactor5);
}

template <typename T>
constexpr mat<T, 2, 2> c_inverse(const mat<T, 2, 2>& M)
{
    T d = det(M);
    return mat<T, 2, 2>{ M(1, 1) / d, -M(0, 1) / d, -M(1, 0) / d, M(0, 0) / d };
}

template <typename T>
constexpr mat<T, 3, 3> c_inverse(const mat<T, 3, 3>& M)
{
    T d = det(M);
    return mat<T, 3, 3>{ (M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1)) / d, (M(0, 2) * M(2, 1) - M(0, 1) * M(2, 2)) / d,
                         (M(0, 1) * M(1, 2) - M(0, 2) * M(1, 1)) / d,

                         (M(1, 2) * M(2, 0) - M(1, 0) * M(2, 2)) / d, (M(0, 0) * M(2, 2) - M(0, 2) * M(2, 0)) / d,
                         (M(0, 2) * M(1, 0) - M(0, 0) * M(1, 2)) / d,

                         (M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0)) / d, (M(0, 1) * M(2, 0) - M(0, 0) * M(2, 1)) / d,
                         (M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0)) / d };
}

template <typename T>
constexpr T det3x3(T a00, T a01, T a02, T a10, T a11, T a12, T a20, T a21, T a22)
{
    return a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);
}

template <typename T>
constexpr mat<T, 4, 4> c_inverse(const mat<T, 4, 4>& M)
{
    // Compute all cofactors for the adjugate matrix
    T c00 = det3x3(M(1, 1), M(1, 2), M(1, 3), M(2, 1), M(2, 2), M(2, 3), M(3, 1), M(3, 2), M(3, 3));
    T c01 = -det3x3(M(1, 0), M(1, 2), M(1, 3), M(2, 0), M(2, 2), M(2, 3), M(3, 0), M(3, 2), M(3, 3));
    T c02 = det3x3(M(1, 0), M(1, 1), M(1, 3), M(2, 0), M(2, 1), M(2, 3), M(3, 0), M(3, 1), M(3, 3));
    T c03 = -det3x3(M(1, 0), M(1, 1), M(1, 2), M(2, 0), M(2, 1), M(2, 2), M(3, 0), M(3, 1), M(3, 2));

    T c10 = -det3x3(M(0, 1), M(0, 2), M(0, 3), M(2, 1), M(2, 2), M(2, 3), M(3, 1), M(3, 2), M(3, 3));
    T c11 = det3x3(M(0, 0), M(0, 2), M(0, 3), M(2, 0), M(2, 2), M(2, 3), M(3, 0), M(3, 2), M(3, 3));
    T c12 = -det3x3(M(0, 0), M(0, 1), M(0, 3), M(2, 0), M(2, 1), M(2, 3), M(3, 0), M(3, 1), M(3, 3));
    T c13 = det3x3(M(0, 0), M(0, 1), M(0, 2), M(2, 0), M(2, 1), M(2, 2), M(3, 0), M(3, 1), M(3, 2));

    T c20 = det3x3(M(0, 1), M(0, 2), M(0, 3), M(1, 1), M(1, 2), M(1, 3), M(3, 1), M(3, 2), M(3, 3));
    T c21 = -det3x3(M(0, 0), M(0, 2), M(0, 3), M(1, 0), M(1, 2), M(1, 3), M(3, 0), M(3, 2), M(3, 3));
    T c22 = det3x3(M(0, 0), M(0, 1), M(0, 3), M(1, 0), M(1, 1), M(1, 3), M(3, 0), M(3, 1), M(3, 3));
    T c23 = -det3x3(M(0, 0), M(0, 1), M(0, 2), M(1, 0), M(1, 1), M(1, 2), M(3, 0), M(3, 1), M(3, 2));

    T c30 = -det3x3(M(0, 1), M(0, 2), M(0, 3), M(1, 1), M(1, 2), M(1, 3), M(2, 1), M(2, 2), M(2, 3));
    T c31 = det3x3(M(0, 0), M(0, 2), M(0, 3), M(1, 0), M(1, 2), M(1, 3), M(2, 0), M(2, 2), M(2, 3));
    T c32 = -det3x3(M(0, 0), M(0, 1), M(0, 3), M(1, 0), M(1, 1), M(1, 3), M(2, 0), M(2, 1), M(2, 3));
    T c33 = det3x3(M(0, 0), M(0, 1), M(0, 2), M(1, 0), M(1, 1), M(1, 2), M(2, 0), M(2, 1), M(2, 2));

    // Compute determinant using first row
    T determinant = M(0, 0) * c00 + M(0, 1) * c01 + M(0, 2) * c02 + M(0, 3) * c03;

    // Create adjugate matrix (transpose of cofactor matrix) divided by determinant
    // clang-format off
    return mat<T, 4, 4>{
        c00/determinant, c10/determinant, c20/determinant, c30/determinant,
        c01/determinant, c11/determinant, c21/determinant, c31/determinant,
        c02/determinant, c12/determinant, c22/determinant, c32/determinant,
        c03/determinant, c13/determinant, c23/determinant, c33/determinant
    };
    // clang-format on
}

template <typename T, std::size_t N>
mat<T, N, N> inverse(const mat<T, N, N>& M)
{
    static_assert(N <= 4 && N >= 2, "Inverse matrix calculation is implemented only for N=2,3,4.");
    if (T d = det(M); almost_zero(d))
        throw std::invalid_argument("inverse matrix require non-zero determinant!");

    return c_inverse(M);
}

// ========================= Vector operations =========================

template <typename T>
constexpr T dot(const vec3<T>& a, const vec3<T>& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename T>
constexpr vec3<T> cross(const vec3<T>& a, const vec3<T>& b)
{
    return vec3<T>{ a.y * b.z - a.z * b.y, //
                    a.z * b.x - a.x * b.z, //
                    a.x * b.y - a.y * b.x };
}

template <typename T>
constexpr T norm2(const vec3<T>& v)
{
    return dot(v, v);
}

template <typename T>
T norm(const vec3<T>& v)
{
    return std::sqrt(norm2(v));
}

template <typename T>
T length(const vec3<T>& v)
{
    return norm(v);
}

template <typename T>
vec3<T> normalize(const vec3<T>& v)
{
    return v / norm(v);
}

template <typename T>
T distance(const vec3<T> a, const vec3<T> b)
{
    return norm(a - b);
}

// ========================= Vector-Matrix Operations

// Homogeneous point/vec conversion
template <typename T>
constexpr mat<T, 1, 4> to_homogeneous(const vec3<T>& v, T w = T(1))
{
    return { v.x, v.y, v.z, w };
}

template <typename T>
constexpr vec3<T> from_homogeneous(const mat<T, 1, 4>& m)
{
    auto& w = m(0, 3);
    return { m(0, 0) / w, m(0, 1) / w, m(0, 2) / w };
}

template <typename T>
constexpr vec3<T> operator*(const mat3<T>& m, const vec3<T>& v)
{
    auto _v = static_cast<mat<T, 3, 1>>(v);
    return vec3<T>{ m * _v };
}

template <typename T>
constexpr vec3<T> operator*(const mat4<T>& m, const vec3<T>& v)
{
    mat<T, 4, 1> _v{ v.x, v.y, v.z, T{ 1 } };
    mat<T, 4, 1> result = m * _v;
    const T& w          = result(3, 0);
    if (almost_zero(w))
        return {};
    return vec3<T>{ result(0) / w, result(1) / w, result(2) / w };
}

template <typename T>
constexpr vec3<T> operator*(const vec3<T>& v, const mat3<T>& m)
{
    auto _v = static_cast<mat<T, 1, 3>>(v);
    return vec3<T>{ _v * m };
}

template <typename T>
constexpr vec3<T> operator*(const vec3<T>& v, const mat4<T>& m)
{
    mat<T, 1, 4> _v{ v.x, v.y, v.z, T{ 1 } };
    mat<T, 1, 4> result = _v * m;
    const T& w          = result(0, 3);
    if (almost_zero(w))
        return {};
    return vec3<T>{ result(0) / w, result(1) / w, result(2) / w };
}

// ========================= 3D Transforms (mat4) =========================

template <typename T>
constexpr mat4<T> translation(const vec3<T>& v)
{
    mat4<T> M = identity<T, 4>();
    M(0, 3)   = v.x;
    M(1, 3)   = v.y;
    M(2, 3)   = v.z;
    return M;
}

template <typename T>
constexpr mat4<T> rotation(const mat3<T>& m)
{
    mat4<T> M(m);
    M(3, 3) = T(1);
    return M;
}

template <typename T>
constexpr mat4<T> scale(const vec3<T>& v)
{
    mat4<T> M{};
    M(0, 0) = v.x;
    M(1, 1) = v.y;
    M(2, 2) = v.z;
    M(3, 3) = T(1);
    return M;
}

template <typename T>
constexpr mat4<T> transform(const vec3<T>& t, const mat3<T>& r, const vec3<T>& s)
{
    auto _t = translation(t);
    auto _r = rotation(r);
    auto _s = scale(s);
    return _t * _r * _s;
}

template <typename T>
mat4<T> rotation_x(T angle)
{
    mat4<T> M = identity<T, 4>();
    T c       = std::cos(angle);
    T s       = std::sin(angle);
    M(1, 1)   = c;
    M(1, 2)   = -s;
    M(2, 1)   = s;
    M(2, 2)   = c;
    return M;
}

// Rotation around Y axis
template <typename T>
mat4<T> rotation_y(T angle)
{
    mat4<T> M = identity<T, 4>();
    T c       = std::cos(angle);
    T s       = std::sin(angle);
    M(0, 0)   = c;
    M(0, 2)   = s;
    M(2, 0)   = -s;
    M(2, 2)   = c;
    return M;
}

// Rotation around Z axis
template <typename T>
mat4<T> rotation_z(T angle)
{
    mat4<T> M = identity<T, 4>();
    T c       = std::cos(angle);
    T s       = std::sin(angle);
    M(0, 0)   = c;
    M(0, 1)   = -s;
    M(1, 0)   = s;
    M(1, 1)   = c;
    return M;
}

// General rotation around arbitrary axis (normalized)
template <typename T>
mat4<T> rotation(const vec3<T>& axis, T angle)
{
    vec3<T> a = normalize(axis);
    T x = a.x, y = a.y, z = a.z;
    T c = std::cos(angle), s = std::sin(angle), ic = 1 - c;

    mat4<T> M = identity<T, 4>();

    M(0, 0) = c + x * x * ic;
    M(0, 1) = x * y * ic - z * s;
    M(0, 2) = x * z * ic + y * s;
    M(1, 0) = y * x * ic + z * s;
    M(1, 1) = c + y * y * ic;
    M(1, 2) = y * z * ic - x * s;
    M(2, 0) = z * x * ic - y * s;
    M(2, 1) = z * y * ic + x * s;
    M(2, 2) = c + z * z * ic;
    return M;
}

// Rodrigues rotation: rotate vector v around axis (normalized) by angle (radians)
template <typename T>
vec3<T> rotate(const vec3<T>& v, const vec3<T>& axis, T angle)
{
    vec3<T> k = normalize(axis); // ensure axis is normalized
    T cosA    = std::cos(angle);
    T sinA    = std::sin(angle);

    // v_rot = v*cosA + (k × v)*sinA + k*(k•v)*(1-cosA)
    return v * cosA + cross(k, v) * sinA + k * (dot(k, v) * (1 - cosA));
}

// Rotation with Euler angles
template <typename T>
mat4<T> rotation(T alpha, T beta, T gamma)
{
    return rotation_x(alpha) * rotation_y(beta) * rotation_z(gamma);
}

template <typename T>
bool is_rotation_valid(const mat3<T>& rot)
{
    if (almost_zero(rot))
        return false;

    T determinant = det(rot);
    if (almost_zero(determinant))
        return false;

    const bool is_orthogonal      = almost_equal(c_inverse(rot), transpose(rot));
    const bool determinant_is_one = almost_equal(std::abs(determinant), T(1.0));

    return is_orthogonal && determinant_is_one;
}

template <typename T>
constexpr bool is_scale_valid(const vec3<T>& s)
{
    return !almost_zero(s.x) && !almost_zero(s.y) && !almost_zero(s.z);
}

template <typename T>
constexpr vec3<T> get_translation(const mat4<T>& t)
{
    return { t(0, 3), t(1, 3), t(2, 3) };
}

template <typename T>
vec3<T> get_scale(const mat4<T>& t)
{
    return { length(vec3<T>{ t(0, 0), t(0, 1), t(0, 2) }),
             length(vec3<T>{ t(1, 0), t(1, 1), t(1, 2) }),
             length(vec3<T>{ t(2, 0), t(2, 1), t(2, 2) }) };
}

template <typename T>
mat3<T> get_rotation(const mat4<T>& t)
{
    auto s = get_scale(t);
    if (!is_scale_valid(s))
        throw std::invalid_argument("scale must be valid!");
    mat3<T> rot{};
    for (std::size_t r = 0; r < 3; r++)
        for (std::size_t c = 0; c < 3; c++)
            rot(r, c) = t(r, c) / s[c];
    if (!is_rotation_valid(rot))
        throw std::invalid_argument("the rotation matrix is invalid");
    return rot;
}

template <typename T>
void decompose(const mat4<T>& transform, vec3<T>& t, mat3<T>& r, vec3<T>& s)
{
    t = get_translation(transform);
    r = get_rotation(transform);
    s = get_scale(transform);
}

template <typename T>
mat4<T> inverse_transform(const mat4<T>& transform)
{
    auto t = get_translation(transform);
    auto r = get_rotation(transform);
    if (!is_rotation_valid(r))
        throw std::invalid_argument("the rotation matrix is invalid");
    auto s = get_scale(transform);
    if (!is_scale_valid(s))
        throw std::invalid_argument("scale must be valid!");

    auto inv_t = translation(-t);
    auto inv_r = rotation(transpose(r));
    auto inv_s = scale(vec3<T>{ 1 / s.x, 1 / s.y, 1 / s.z });
    return inv_s * inv_r * inv_t;
}

// ========================= Camera (mat4) =========================

template <typename T>
mat4<T> look_at(const vec3<T>& eye, const vec3<T>& center, const vec3<T>& up)
{
    vec3<T> f = normalize(center - eye); // Direction camera is looking
    vec3<T> r = normalize(cross(f, up)); // Camera's right
    vec3<T> u = cross(r, f);             // Camera's up (recomputed for orthogonality)

    // Create translation part (negative because we're moving the world, not the camera)
    vec3<T> t{ -dot(r, eye), -dot(u, eye), dot(f, eye) };

    return {
        r.x,    r.y,    r.z,    t.x,   //
        u.x,    u.y,    u.z,    t.y,   //
        -f.x,   -f.y,   -f.z,   t.z,   //
        T{ 0 }, T{ 0 }, T{ 0 }, T{ 1 } //
    };
}

template <typename T>
constexpr vec3<T> right_vec(const mat4<T>& view)
{
    return { view.template get<0, 0>(), view.template get<0, 1>(), view.template get<0, 2>() };
}

template <typename T>
constexpr vec3<T> up_vec(const mat4<T>& view)
{
    return { view.template get<1, 0>(), view.template get<1, 1>(), view.template get<1, 2>() };
}

template <typename T>
constexpr vec3<T> forward_vec(const mat4<T>& view)
{
    return { -view.template get<2, 0>(), -view.template get<2, 1>(), -view.template get<2, 2>() };
}

template <typename T>
constexpr vec3<T> translation_vec(const mat4<T>& view)
{
    return { view.template get<3, 0>(), view.template get<3, 1>(), view.template get<3, 2>() };
}

template <typename T>
constexpr mat4<T> ortho(T left, T right, T bottom, T top, T near, T far)
{
    mat4<T> result{};

    result(0, 0) = T{ 2 } / (right - left);
    result(1, 1) = T{ 2 } / (top - bottom);
    result(2, 2) = -T{ 2 } / (far - near);

    result(0, 3) = -(right + left) / (right - left);
    result(1, 3) = -(top + bottom) / (top - bottom);
    result(2, 3) = -(far + near) / (far - near);

    result(3, 3) = T{ 1 };

    return result;
}

/**
 * @tparam T
 * @param fovy field of view in Y direction (radians)
 * @param aspect aspect ratio (width/height)
 * @param near distance to near clipping plane
 * @param far distance to far clipping plane
 * @return
 */
template <typename T>
mat4<T> perspective(T fovy, T aspect, T near, T far)
{
    T tan_half_fovy = std::tan(fovy / T{ 2 });

    mat4<T> result{};

    result(0, 0) = T{ 1 } / (aspect * tan_half_fovy);
    result(1, 1) = T{ 1 } / tan_half_fovy;
    result(2, 2) = -(far + near) / (far - near);
    result(2, 3) = -(T{ 2 } * far * near) / (far - near);
    result(3, 2) = -T{ 1 };

    return result;
}

template <typename T>
constexpr T radians(T degrees)
{
    return degrees * pi<T> / T{ 180 };
}

} // namespace lina
