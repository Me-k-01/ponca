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
#include <Ponca/src/Fitting/covariancePlaneFit.h>
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
    for(int i = 0; i < int(points.size()); ++i) {
        points[i] = getPointOnSphere<DataPoint>(radius, center, false, false, false);
    }

    return analysisScale;
}

/*! \brief Copy the point from a vector of point to a vector of point bound to an interlaced array of position and normal
 *
 * @tparam DataPoint Regular point data type (e.g. \ref PointPositionNormal)
 * @tparam DataPointRef The point data type referencing the interlaced array (e.g. \ref PointPositionNormalBinding, \ref PointPositionNormalLateBinding)
 * @param points As an input, a vector of DataPoint.
 * @param pointsBinding As an output, an empty vector of DataPointRef.
 * @return An interlaced array containing the position and normal values. The pointsBinding points it's data to the array
 */
template<typename DataPoint, typename DataPointRef>
typename DataPoint::Scalar * copyDataRef(std::vector<DataPoint>& points, std::vector<DataPointRef>& pointsBinding)
{
    static_assert(DataPoint::Dim == DataPointRef::Dim, "Both dimension should be the same");
    using VectorType         = typename DataPoint::VectorType;
    using Scalar             = typename DataPoint::Scalar;
    constexpr auto DIMENSION = DataPoint::Dim;
    const int nPoints        = points.size();
    auto* interlacedArray    = new Scalar[2*DIMENSION*nPoints];
    pointsBinding.reserve(nPoints);

#ifdef NDEBUG
#pragma omp parallel for
#endif
    for(int i=0; i<nPoints; ++i)
    {
        // We use Eigen Vectors to compute both coordinates and normals,
        // and then copy the raw values to an interlaced array.
        VectorType n = points[i].normal();
        VectorType p = points[i].pos();

        // Grab coordinates and store them as raw buffer
        memcpy(interlacedArray+2*DIMENSION*i,           p.data(), DIMENSION*sizeof(Scalar));
        memcpy(interlacedArray+2*DIMENSION*i+DIMENSION, n.data(), DIMENSION*sizeof(Scalar));

        pointsBinding.emplace_back(interlacedArray, i);
    }

    return interlacedArray;
}

/*! \brief Compares the fit results between two point data type
 *
 * @tparam DataPoint1 A point data type
 * @tparam DataPoint2 Another point data type to compare to
 * @param points1
 * @param points2
 * @param analysisScale The radius of the neighborhood for the fitting process
 */
template<template<typename> typename Fit, typename DataPoint1, typename DataPoint2, typename CompareFitFunctor>
void compareDataPointOnFit( std::vector<DataPoint1>& points1, std::vector<DataPoint2>& points2, typename DataPoint1::Scalar analysisScale, CompareFitFunctor compareFit)
{
    static_assert( DataPoint1::Dim == DataPoint2::Dim, "Both dimension should be the same" );
    static_assert( std::is_same_v<typename DataPoint1::Scalar, typename DataPoint2::Scalar>, "Both scalar type should be the same" );
    PONCA_DEBUG_ASSERT_MSG( points1.size() == points2.size(), "Both size should be the same" );

    KdTreeDense<DataPoint1> tree1(points1);
    KdTreeDense<DataPoint2> tree2(points2);

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
        auto neighborhoodRange1 = tree1.rangeNeighbors(points1[i].pos(), analysisScale);
        f1.computeWithIds( neighborhoodRange1, points1 );

        Fit<DataPoint2> f2;
        f2.setNeighborFilter({points2[i].pos(), analysisScale});
        auto neighborhoodRange2 = tree2.rangeNeighbors(points2[i].pos(), analysisScale);
        f2.computeWithIds( neighborhoodRange2, points2 );

        // VERIFY((compareFit(f1, f2)));
        compareFit(f1, f2);
    }
}

//! \brief Smooth weight neighbor filter class templated over the point data type.
template <typename DataPoint>
using NeighborFilter = DistWeightFunc<DataPoint, SmoothWeightKernel<typename DataPoint::Scalar> >;
//! \brief Fitting templated over the point data type.
template <typename DataPoint>
using SphereFit      = Basket<DataPoint, NeighborFilter<DataPoint>, OrientedSphereFit>;

template<typename Scalar, int Dim>
void callSubTests()
{
    typedef PointPositionNormal<Scalar, Dim> Point;
    typedef PointPositionNormalLateBinding<Scalar, Dim> PointRef;

    // Prepare points
    std::vector<Point> points;
    std::vector<PointRef> pointsRef;
    for(int i = 0; i < g_repeat; ++i)
    {
        const Scalar analysisScale    = generateData(points);
        const Scalar* interlacedArray = copyDataRef(points, pointsRef);

        // Compare fits
        CALL_SUBTEST((compareDataPointOnFit<SphereFit>(points, pointsRef, analysisScale, [](auto& f1, auto& f2){
            // Both fitting result should be the same
            // const Scalar epsilon          = Eigen::NumTraits<Scalar>::dummy_precision();
            // const Scalar squaredEpsilon   = epsilon*epsilon;
            std::pow(f1.m_uc - f2.m_uc, Scalar(2)) < Eigen::NumTraits<Scalar>::dummy_precision()*Eigen::NumTraits<Scalar>::dummy_precision() &&
            std::pow(f1.m_uq - f2.m_uq, Scalar(2)) < Eigen::NumTraits<Scalar>::dummy_precision()*Eigen::NumTraits<Scalar>::dummy_precision() &&
            f1.m_ul.isApprox(f2.m_ul);
        })));

        points.clear();
        pointsRef.clear();
        delete[] interlacedArray;
    }
}

int main(int argc, char** argv)
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
