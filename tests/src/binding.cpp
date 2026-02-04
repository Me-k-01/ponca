/*
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


/*!
    \file test/basket.cpp
    \brief Test basket utility functions
 */

#include "../common/testing.h"
#include "../common/testUtils.h"

#include "../split_test_helper.h"

#include <Ponca/src/Fitting/basket.h>
#include <Ponca/src/Fitting/orientedSphereFit.h>
#include <Ponca/src/Fitting/weightFunc.h>
#include <Ponca/src/Fitting/weightKernel.h>
#include <Ponca/src/SpatialPartitioning/KdTree/kdTree.h>

#include <vector>

using namespace std;
using namespace Ponca;

template<typename DataPoint>
typename DataPoint::Scalar generateData(vector<DataPoint>& points)
{
    typedef typename DataPoint::Scalar Scalar;
    typedef typename DataPoint::VectorType VectorType;

    //generate sampled sphere
#ifdef NDEBUG
    int nbPoints = Eigen::internal::random<int>(500, 1000);
#else
    int nbPoints = Eigen::internal::random<int>(100, 200);
#endif

    Scalar radius = Eigen::internal::random<Scalar>(1., 10.);
    Scalar analysisScale = Scalar(10.) * std::sqrt( Scalar(4. * M_PI) * radius * radius / nbPoints);
    Scalar centerScale = Eigen::internal::random<Scalar>(1,10000);
    VectorType center = VectorType::Random() * centerScale;

    points.reserve(nbPoints);
#ifdef NDEBUG
#pragma omp parallel for
#endif
    for(int i = 0; i < nbPoints; ++i) {
        points.push_back(getPointOnSphere<DataPoint>(radius, center, false, false, false));
    }

    return analysisScale;
}

/*! \brief Variadic function to copy a vector of points to one or multiple vectors of points that are bound to a single interlaced array of positions and normals.
 *
 * \tparam DataPoint Regular point data type (e.g. \ref PointPositionNormal)
 * \tparam DataPointRef Point data types referencing the interlaced array (e.g. \ref PointPositionNormalBinding, \ref PointPositionNormalLateBinding)
 * \param points As an input, a vector of DataPoint.
 * \param pointsBinding As an output, empty vectors of DataPointRef types.
 * \return An interlaced array containing the position and normal values. The `pointsBinding` vectors are linked to this array.
 */
template<typename DataPoint, typename... DataPointRef>
typename DataPoint::Scalar * copyDataRef(std::vector<DataPoint>& points, std::vector<DataPointRef>&... pointsBinding)
{
    constexpr int DIMENSION     =  DataPoint::Dim;
    // static_assert(DIMENSION    == DataPointRef::Dim, "Both dimension should be the same");
    using VectorType            = typename DataPoint::VectorType;
    using Scalar                = typename DataPoint::Scalar;

    const int nPoints           = points.size();
    auto* const interlacedArray = new Scalar[2*DIMENSION*nPoints];
    (pointsBinding.reserve(nPoints), ...);

    for(int i=0; i<nPoints; ++i)
    {
        // We use Eigen Vectors to compute both coordinates and normals,
        // and then copy the raw values to an interlaced array.
        VectorType n = points[i].normal();
        VectorType p = points[i].pos();

        // Grab coordinates and store them as raw buffer
        memcpy(interlacedArray+2*DIMENSION*i          , p.data(), DIMENSION*sizeof(Scalar));
        memcpy(interlacedArray+2*DIMENSION*i+DIMENSION, n.data(), DIMENSION*sizeof(Scalar));

        (pointsBinding.emplace_back(interlacedArray, i), ...);
    }

    return interlacedArray;
}

/*! \brief Compares the fit results between two point data type.
 *
 * The fit is computed for each point of the point cloud.
 *
 * \param kdtree1 A kdtree containing the first set of data point
 * \param kdtree2 A kdtree containing the first set of data point that we are comparing to the first
 * \param analysisScale The radius of the neighborhood for the fitting process
 * \param compareFit A functor that does a comparison on the computed fits.
 */
template<template<typename> typename Fit, typename KdTree1, typename KdTree2, typename CompareFitFunctor>
void compareKdTreeWithFit( KdTree1& kdtree1, KdTree2& kdtree2, typename KdTree1::DataPoint::Scalar analysisScale, CompareFitFunctor compareFit)
{
    using DataPoint1          = typename KdTree1::DataPoint;
    using DataPoint2          = typename KdTree2::DataPoint;
    constexpr int  DIMENSION  = DataPoint1::Dim;
    static_assert( DIMENSION == DataPoint2::Dim, "Both dimension should be the same" );
    static_assert( std::is_same_v<typename DataPoint1::Scalar, typename DataPoint2::Scalar>, "Both scalar type should be the same" );

    const std::vector<DataPoint1>& points1 = kdtree1.points();
    const std::vector<DataPoint2>& points2 = kdtree2.points();

    VERIFY( points1.size() == points2.size() );

    // Quick testing is requested for coverage
    const int nPoint = QUICK_TESTS ? 1 : int(points1.size());
    // Test for each point if the fitted sphere correspond to the theoretical sphere
#ifdef NDEBUG
#pragma omp parallel for
#endif
    for(int i = 0; i < nPoint; ++i)
    {
        Fit<DataPoint1> f1;
        f1.setNeighborFilter({points1[i].pos(), analysisScale});
        auto neighborhoodRange1 = kdtree1.rangeNeighbors(points1[i].pos(), analysisScale);
        f1.computeWithIds( neighborhoodRange1, points1 );

        Fit<DataPoint2> f2;
        f2.setNeighborFilter({points2[i].pos(), analysisScale});
        auto neighborhoodRange2 = kdtree2.rangeNeighbors(points2[i].pos(), analysisScale);
        f2.computeWithIds( neighborhoodRange2, points2 );

        VERIFY((compareFit(f1, f2)));
    }
}

//! \brief Smooth weight neighbor filter class templated over the point data type.
template <typename DataPoint>
using NeighborFilter = DistWeightFunc<DataPoint, SmoothWeightKernel<typename DataPoint::Scalar> >;
//! \brief Fitting templated over the point data type.
template <typename DataPoint>
using TestSphereFit  = Basket<DataPoint, NeighborFilter<DataPoint>, OrientedSphereFit>;

///! \brief Verify that the Sphere Fit are equals, despite the DataPoint types being different.
template <typename DataPoint1, typename DataPoint2>
bool sphereFitAreEquals(const TestSphereFit<DataPoint1>& f1, const TestSphereFit<DataPoint2>& f2) {
    static_assert( std::is_same_v<typename DataPoint1::Scalar, typename DataPoint2::Scalar>, "Both scalar type should be the same" );
    using Scalar = typename DataPoint1::Scalar;
    // Both fitting result should be the same
    const Scalar eps        = Eigen::NumTraits<Scalar>::dummy_precision();
    const Scalar squaredEps = eps * eps;
    return pow(f1.m_uc - f2.m_uc, Scalar(2)) < squaredEps
        && pow(f1.m_uq - f2.m_uq, Scalar(2)) < squaredEps
        && f1.m_ul.isApprox(f2.m_ul);
}

template<typename Scalar, int Dim>
void callSubTests()
{
    typedef PointPositionNormal<Scalar, Dim> Point;
    typedef PointPositionNormalBinding<Scalar, Dim> PointRef;
    typedef PointPositionNormalLateBinding<Scalar, Dim> PointLateRef;

#ifdef NDEBUG
#pragma omp parallel for
#endif
    for(int i = 0; i < g_repeat; ++i)
    {
        // Points to compare
        vector<Point>        points;
        vector<PointRef>     pointsRef;
        vector<PointLateRef> pointsLateRef;

        const Scalar  analysisScale   = generateData(points);
        // Copy the point data to an external buffer, and bind some point vectors to it.
        const Scalar* interlacedArray = copyDataRef(points, pointsRef, pointsLateRef);

        KdTreeDense<Point>        kdtree(points);
        KdTreeDense<PointRef>     kdtreeRef(pointsRef);
        KdTreeDense<PointLateRef> kdtreeLateRef(pointsLateRef);

        // Compare fits made with the kdtree
        CALL_SUBTEST((compareKdTreeWithFit<TestSphereFit>(
            kdtree, kdtreeRef, analysisScale, sphereFitAreEquals<Point, PointRef>
        )));
        CALL_SUBTEST((compareKdTreeWithFit<TestSphereFit>(
            kdtree, kdtreeLateRef, analysisScale, sphereFitAreEquals<Point, PointLateRef>
        )));

        // Clear buffer before next pass
        delete[] interlacedArray;
    }
}

int main(const int argc, char** argv)
{
    if(!init_testing(argc, argv))
    {
        return EXIT_FAILURE;
    }

    cout << "Test Binding point type in 3 dimensions: float" << flush;
    CALL_SUBTEST_1((callSubTests<float, 3>()));
    cout << " (ok), double" << flush;
    CALL_SUBTEST_2((callSubTests<double, 3>()));
    cout << " (ok)" << flush;
    cout << ", long double" << flush;
    CALL_SUBTEST_3((callSubTests<long double, 3>()));
    cout << " (ok)" << flush;
}
