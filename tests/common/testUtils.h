/*
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


/*!
  \file test/common/testUtils.h
  \brief Useful functions for tests
*/

#pragma once

#include "Eigen/Eigen"
#include "Ponca/src/Common/defines.h"
#include PONCA_MULTIARCH_INCLUDE_CU_STD(cmath)

#include <vector>

#define MIN_NOISE 0.99
#define MAX_NOISE 1.01

// Epsilon precision
template<typename T> inline T testEpsilon()
{
    return Eigen::NumTraits<T>::dummy_precision();
}

template<> inline float testEpsilon<float>()
{
    return 1e-2f;
}

template<> inline double testEpsilon<double>()
{
    return 1e-5;
}

template<> inline long double testEpsilon<long double>()
{
    return 1e-5;
}

//! [PointPositionNormal]
// Point with position and normal vector
template<typename _Scalar, int _Dim>
class PointPositionNormal
{
public:
    enum {Dim = _Dim};
    typedef _Scalar Scalar;
    typedef Eigen::Matrix<Scalar, Dim,   1>		VectorType;
    typedef Eigen::Matrix<Scalar, Dim, Dim>	MatrixType;

    PONCA_MULTIARCH inline PointPositionNormal(
            const VectorType &pos = VectorType::Zero(),
            const VectorType& normal = VectorType::Zero() )
        : m_pos(pos), m_normal(normal) {}

    PONCA_MULTIARCH inline const VectorType& pos()    const { return m_pos; }
    PONCA_MULTIARCH inline const VectorType& normal() const { return m_normal; }

    PONCA_MULTIARCH inline VectorType& pos()    { return m_pos; }
    PONCA_MULTIARCH inline VectorType& normal() { return m_normal; }

private:
    VectorType m_pos, m_normal;
};
//! [PointPositionNormal]

//! [PointPosition]
/// Point with position, without attribute
template<typename _Scalar, int _Dim>
class PointPosition
{
public:
    enum {Dim = _Dim};
    typedef _Scalar Scalar;
    typedef Eigen::Matrix<Scalar, Dim,   1>	VectorType;
    typedef Eigen::Matrix<Scalar, Dim, Dim>	MatrixType;

    PONCA_MULTIARCH inline PointPosition(  const VectorType &pos = VectorType::Zero() )
        : m_pos(pos) {}

    PONCA_MULTIARCH inline const VectorType& pos()    const { return m_pos; }

    PONCA_MULTIARCH inline VectorType& pos()    { return m_pos; }

private:
    VectorType m_pos;
};
//! [PointPosition]

template<typename DataPoint>
void reverseNormals(std::vector<DataPoint>& _dest, const std::vector<DataPoint>& _src, bool _bRandom = true)
{
    typedef typename DataPoint::VectorType VectorType;

    VectorType vNormal;

    for(unsigned int i = 0; i < _src.size(); ++i)
    {
        vNormal = _src[i].normal();

        if(_bRandom)
        {
            float reverse = Eigen::internal::random<float>(0.f, 1.f);
            if(reverse > 0.5f)
            {
                vNormal = -vNormal;
            }
        }
        else
        {
            vNormal = -vNormal;
        }

        _dest[i] = DataPoint(_src[i].pos(), vNormal);
    }
}

template<typename DataPoint>
DataPoint getPointOnSphere(typename DataPoint::Scalar _radius, typename DataPoint::VectorType _vCenter, bool _bAddPositionNoise = true,
                           bool _bAddNormalNoise = true, bool _bReverseNormals = false)
{
    typedef typename DataPoint::Scalar Scalar;
    typedef typename DataPoint::VectorType VectorType;

    VectorType vNormal = VectorType::Random().normalized();

    VectorType vPosition = _vCenter + vNormal * _radius; // * Eigen::internal::random<Scalar>(MIN_NOISE, MAX_NOISE);

    if(_bAddPositionNoise)
    {
        //vPosition = _vCenter + vNormal * _radius * Eigen::internal::random<Scalar>(MIN_NOISE, MAX_NOISE);
        vPosition = vPosition + VectorType::Random().normalized() *
                Eigen::internal::random<Scalar>(Scalar(0.), Scalar(1. - MIN_NOISE));
        vNormal = (vPosition - _vCenter).normalized();
    }

    if(_bAddNormalNoise)
    {
        VectorType vTempPos =  vPosition + VectorType::Random().normalized() *
                Eigen::internal::random<Scalar>(Scalar(0.), Scalar(1. - MIN_NOISE));
        vNormal = (vTempPos - _vCenter).normalized();
    }
    if(_bReverseNormals)
    {
        float reverse = Eigen::internal::random<float>(0.f, 1.f);
        if(reverse > 0.5f)
        {
            vNormal = -vNormal;
        }
    }

    return DataPoint(vPosition, vNormal);
}

/*! \brief Generate points on a plane without normals */
template<typename DataPoint>
DataPoint getPointOnRectangularPlane(
        const typename DataPoint::VectorType& _vPosition,
        const typename DataPoint::VectorType& /*_vNormal*/,
        const typename DataPoint::Scalar& _width,
        const typename DataPoint::Scalar& _height,
        const typename DataPoint::VectorType& _localxAxis,
        const typename DataPoint::VectorType& _localyAxis,
        bool _bAddPositionNoise = true)
{
    typedef typename DataPoint::Scalar Scalar;
    typedef typename DataPoint::VectorType VectorType;

    Scalar u = Eigen::internal::random<Scalar>(-_width/Scalar(2),
                                                _width/Scalar(2));
    Scalar v = Eigen::internal::random<Scalar>(-_height/Scalar(2),
                                                _height/Scalar(2));

    VectorType vRandomPosition = _vPosition + u*_localxAxis + v*_localyAxis;

    if(_bAddPositionNoise)
    {
        vRandomPosition = vRandomPosition +
                VectorType::Random().normalized() *
                Eigen::internal::random<Scalar>(0., 1. - MIN_NOISE);
    }

    return DataPoint(vRandomPosition);
}

template<typename DataPoint>
DataPoint getPointOnPlane(typename DataPoint::VectorType _vPosition,
                          typename DataPoint::VectorType _vNormal,
                          typename DataPoint::Scalar _radius,
                          bool _bAddPositionNoise = true,
                          bool _bAddNormalNoise = true,
                          bool _bReverseNormals = false	)
{
    typedef typename DataPoint::Scalar Scalar;
    typedef typename DataPoint::VectorType VectorType;
    typedef Eigen::Quaternion<Scalar> QuaternionType;


    VectorType vRandom;
    VectorType vRandomDirection = VectorType::Zero();
    VectorType vRandomPoint = VectorType::Zero();
    VectorType vLocalUp = _vNormal;

    do
    {
        vRandom = VectorType::Random().normalized(); // Direction in the unit sphere
        vRandomDirection = vRandom.cross(vLocalUp);
    }
    while(vRandomDirection == VectorType::Zero());

    vRandomDirection = vRandomDirection.normalized();
    vRandomPoint = vRandomDirection * _radius;
    vRandomPoint += _vPosition;

    if(_bAddPositionNoise)
    {
        vRandomPoint = vRandomPoint + VectorType::Random().normalized() *
                Eigen::internal::random<Scalar>(Scalar(0), Scalar(1. - MIN_NOISE));
    }

    if(_bAddNormalNoise)
    {
        VectorType vLocalLeft = vLocalUp.cross(vRandomDirection);
        VectorType vLocalFront = vLocalLeft.cross(vLocalUp);

        Scalar rotationAngle = Eigen::internal::random<Scalar>(Scalar(-M_PI / 16.), Scalar(M_PI / 16.));
        VectorType vRotationAxis = vLocalLeft;
        QuaternionType qRotation = QuaternionType(Eigen::AngleAxis<Scalar>(rotationAngle, vRotationAxis));
        qRotation = qRotation.normalized();
        vLocalUp = qRotation * vLocalUp;

        rotationAngle = Eigen::internal::random<Scalar>(Scalar(-M_PI / 16.), Scalar(M_PI / 16.));
        vRotationAxis = vLocalFront;
        qRotation = QuaternionType(Eigen::AngleAxis<Scalar>(rotationAngle, vRotationAxis));
        qRotation = qRotation.normalized();
        vLocalUp = qRotation * vLocalUp;
    }

    if(_bReverseNormals)
    {
        float reverse = Eigen::internal::random<float>(0.f, 1.f);
        if(reverse > 0.5f)
        {
            vLocalUp = -vLocalUp;
        }
    }

    return DataPoint(vRandomPoint, vLocalUp);
}

/// Generate z value using the equation z = ax^2 + by^2
template<typename Scalar>
inline Scalar
getParaboloidZ(Scalar _x, Scalar _y, Scalar _a, Scalar _b)
{
    return _a*_x*_x + _b*_y*_y;
}
/// Generate z value using the equation z = ax^2 + by^2
template<typename VectorType>
inline VectorType
getParaboloidNormal(const VectorType& in,
                    typename VectorType::Scalar _a,
                    typename VectorType::Scalar _b)
{
    return VectorType((_a * in.x()), (_b * in.y()), -1.).normalized();;
}

/// Generate point samples on the primitive z = ax^2 + by^2
/// Points (x,y) are generated in the interval [-_s, -s]^2
/// See getParaboloidZ and getParaboloidNormal
template<typename DataPoint>
[[nodiscard]] DataPoint
getPointOnParaboloid(typename DataPoint::Scalar _a,
                     typename DataPoint::Scalar _b,
                     typename DataPoint::Scalar _s,
                     bool _bAddNoise = true)
{
    typedef typename DataPoint::Scalar Scalar;
    typedef typename DataPoint::VectorType VectorType;

    VectorType vNormal;
    VectorType vPosition;

    Scalar x = Eigen::internal::random<Scalar>(-_s, _s),
           y = Eigen::internal::random<Scalar>(-_s, _s);

    vPosition = { x, y, getParaboloidZ(x, y, _a, _b)};
    vNormal = getParaboloidNormal(vPosition, _a, _b);

    if(_bAddNoise) //spherical noise
    {
        vPosition += VectorType::Random().normalized() * Eigen::internal::random<Scalar>(Scalar(0), Scalar(1. - MIN_NOISE));
    }

    return DataPoint(vPosition, vNormal);
}

template<typename DataPoint>
typename DataPoint::Scalar getPointKappaMean(typename DataPoint::VectorType _vPoint, typename DataPoint::Scalar _a, typename DataPoint::Scalar _b)
{
    typedef typename DataPoint::Scalar Scalar;

    Scalar x = _vPoint.x();
    Scalar y = _vPoint.y();

    Scalar ax2 = _a * x * _a * x;
    Scalar by2 = _b * y * _b * y;

    Scalar num = (1 + ax2) * _b + (1 + by2) * _a;
    Scalar den = (1 + ax2 + by2);
    den = std::pow(den, Scalar(3./2.));

    Scalar kappa = num / den * Scalar(0.5);

    return kappa;
}

template<typename DataPoint>
typename DataPoint::Scalar getKappaMean(const std::vector<DataPoint>& _vectorPoints, typename DataPoint::VectorType _vCenter,
                                        typename DataPoint::Scalar _a, typename DataPoint::Scalar _b, typename DataPoint::Scalar _analysisScale)
{
    typedef typename DataPoint::Scalar Scalar;
    typedef typename DataPoint::VectorType VectorType;

    int size = static_cast<int>(_vectorPoints.size());
    Scalar kappaMean = 0.;
    int nbNei = 0;

    for(int i = 0; i < size; ++i)
    {
        VectorType q = _vectorPoints[i].pos() - _vCenter;

        if(q.norm() <= _analysisScale)
        {
            kappaMean += getPointKappaMean<DataPoint>(_vectorPoints[i].pos(), _a, _b);
            ++nbNei;
        }
    }

    return kappaMean / nbNei;
}

template<typename Fit1, typename Fit2>
void isSamePlane(const Fit1& fit1, const Fit2& fit2) {
    const auto &plane1 = fit1.compactPlane();
    const auto &plane2 = fit2.compactPlane();

    // Test we fit the same plane
    VERIFY(plane1.isApprox(plane2));
}

template<typename Fit1, typename Fit2>
void isSameSphere(const Fit1& fit1, const Fit2& fit2) {
    const auto &sphere1 = fit1.algebraicSphere();
    const auto &sphere2 = fit2.algebraicSphere();

    // Test we fit the same plane
    VERIFY(sphere1 == sphere2);
}

template<typename Fit1, typename Fit2>
void hasSamePlaneDerivatives(const Fit1& fit1, const Fit2& fit2) {
    // Get covariance
    const auto& dpot1 = fit1.covariancePlaneDer().dPotential();
    const auto& dpot2 = fit2.covariancePlaneDer().dPotential();
    const auto& dnor1 = fit1.covariancePlaneDer().dNormal();
    const auto& dnor2 = fit2.covariancePlaneDer().dNormal();

    // Test we compute the same derivatives
    VERIFY(dpot1.isApprox( dpot2 ));
    VERIFY(dnor1.isApprox( dnor2 ));
}
