#include <vector>
#include <cassert>
#include "fun.h"

namespace gradlib {

struct Vec {
    std::vector<Fun> v;

    // Constructors
    Vec() = default;
    explicit Vec(size_t size) : v(size) {}
    Vec(size_t size, const Fun& val) : v(size, val) {}
    explicit Vec(std::vector<Fun> data) : v(std::move(data)) {}

    // Resize
    void resize(size_t new_size, const Fun& val = Fun{}) {
        v.resize(new_size, val);
    }

    // Element-wise addition
    Vec operator+(const Vec& other) const {
        assert(v.size() == other.v.size());
        Vec result(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            result.v[i] = v[i] + other.v[i];
        return result;
    }

    // Element-wise subtraction
    Vec operator-(const Vec& other) const {
        assert(v.size() == other.v.size());
        Vec result(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            result.v[i] = v[i] - other.v[i];
        return result;
    }

    // Element-wise multiplication
    Vec operator*(const Vec& other) const {
        assert(v.size() == other.v.size());
        Vec result(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            result.v[i] = v[i] * other.v[i];
        return result;
    }

    // Element-wise division
    Vec operator/(const Vec& other) const {
        assert(v.size() == other.v.size());
        Vec result(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            result.v[i] = v[i] / other.v[i];
        return result;
    }

    // Scalar operations
    Vec operator+(const Fun& scalar) const {
        Vec result(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            result.v[i] = v[i] + scalar;
        return result;
    }
    Vec operator+(double scalar) const {
        return *this + Fun(scalar);
    }

    Vec operator-(const Fun& scalar) const {
        Vec result(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            result.v[i] = v[i] - scalar;
        return result;
    }
    Vec operator-(double scalar) const {
        return *this - Fun(scalar);
    }

    Vec operator*(const Fun& scalar) const {
        Vec result(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            result.v[i] = v[i] * scalar;
        return result;
    }
    Vec operator*(double scalar) const {
        return *this * Fun(scalar);
    }

    Vec operator/(const Fun& scalar) const {
        Vec result(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            result.v[i] = v[i] / scalar;
        return result;
    }
    Vec operator/(double scalar) const {
        return *this / Fun(scalar);
    }

    Vec grad() {
        Vec res(size());
        for (size_t i = 0; i < v.size(); ++i)
            res[i] = v[i].grad();
        return res;
    }

    // Dot product
    Fun dot(const Vec& other) const {
        assert(v.size() == other.v.size());
        Fun result(0.0);
        for (size_t i = 0; i < v.size(); ++i)
            result = result + (v[i] * other.v[i]);
        return result;
    }

    // Access
    size_t size() const { return v.size(); }
    Fun& operator[](size_t idx) { return v[idx]; }
    const Fun& operator[](size_t idx) const { return v[idx]; }
};

}  // namespace gradlib
