/**
Copyright (c) 2022
 Jacques-Olivier Lachaud (\c jacques-olivier.lachaud@univ-savoie.fr)
 Laboratory of Mathematics (CNRS, UMR 5807), University of Savoie, France,

All rights reserved.

*/

#pragma once

#include <iostream>
#include <random>
#include <vector>
#include "boundedRange.h"

namespace Ponca::internal {
    /*!
        \internal
        \brief Generates the triangles used by the CNC Fit depending on the method.
            As an output, pushes every generated triangle into the "triangles" vector and returns the number of triangle that was pushed into the List.
        \note Needs to be implemented for each triangle generation method by specializing the template over the TriangleGenerationMethod
    */
    template <TriangleGenerationMethod Method, typename P>
    struct TriangleGenerator {
        using VectorType = typename P::VectorType;
        template <typename IndexRange, typename PointContainer>
        static int generate(
            const IndexRange& /*ids*/,
            const PointContainer& /*points*/,
            const VectorType& /*_evalPointPos*/, const VectorType& /*_evalPointNormal*/,
            std::vector<Triangle<P>>& /*triangles*/
        ) {
            static_assert(true, "Triangle generation method not implemented!");
            return 0;
        }
    };

    /// Generates the triangles used by the CNC Fit using UniformGeneration
    template <typename P>
    struct TriangleGenerator<UniformGeneration, P> {
    private:
        static constexpr int maxTriangles {100};
    public:
        using VectorType = typename P::VectorType;

        template <typename IndexRange, typename PointContainer>
        static int generate(
            const IndexRange& ids,
            const PointContainer& points,
            const VectorType& /*_evalPointPos*/, const VectorType& /*_evalPointNormal*/,
            std::vector<Triangle<P>>& triangles
        ) {
            int nb_vt = 0; // Number of valid generated triangles

            for (int i = 0; i < maxTriangles; ++i) {
                // Randomly select triangles
                int i1 = ids[Eigen::internal::random<int>(0, ids.size()-1)];
                int i2 = ids[Eigen::internal::random<int>(0, ids.size()-1)];
                int i3 = ids[Eigen::internal::random<int>(0, ids.size()-1)];
                if (i1 == i2 || i1 == i3 || i2 == i3) continue;

                triangles.push_back(internal::Triangle<P>(points[i1], points[i2], points[i3]));
                nb_vt++;
            }
            return nb_vt;
        }
    };

    /// Generates the triangles used by the CNC Fit using IndependentGeneration
    template <typename P>
    struct TriangleGenerator<IndependentGeneration, P> {
    private:
        static constexpr int maxTriangles {100};
    public:
        using VectorType = typename P::VectorType;
        using Scalar = typename P::Scalar;

        template <typename IndexRange, typename PointContainer>
        static int generate(
            const IndexRange& ids,
            const PointContainer& points,
            const VectorType& /*_evalPointPos*/, const VectorType& /*_evalPointNormal*/,
            std::vector<Triangle<P>>& triangles
        ) {
            int nb_vt = 0; // Number of valid generated triangles

            // Makes a new array to shuffle
            std::vector<int> indices(ids.size());
            for (int i = 0; i < ids.size() ; ++i)
                indices[i] = ids[i];

            // Shuffles the neighbors
            std::random_device rd;
            std::mt19937 rg(rd());
            std::shuffle(indices.begin(), indices.end(), rg);

            // Compute the triangles
            triangles.clear();
            for (const int max_triangles = std::min(maxTriangles, static_cast<int>(ids.size()) / 3); nb_vt < max_triangles-2; nb_vt++) {
                int i1 = indices[nb_vt];
                int i2 = indices[nb_vt+1];
                int i3 = indices[nb_vt+2];
                triangles.push_back(internal::Triangle<P>(points[i1], points[i2], points[i3]));
            }
            return nb_vt;
        }
    };

    template<typename P>
    struct HexagramBase {
        using Scalar = typename P::Scalar;
    private:
#define FUNC_COMPUTE_COSIN(COSIN)                                         \
    template<typename Scalar>                                             \
    static constexpr std::array<Scalar, 6> compute_##COSIN##_values() {   \
        std::array<Scalar, 6> result = {};                                \
        for (int i = 0; i < 6; ++i) {                                     \
            result[i] = std::COSIN(i * M_PI / 3.0);                       \
        }                                                                 \
        return result;                                                    \
    }
        FUNC_COMPUTE_COSIN(cos)
        FUNC_COMPUTE_COSIN(sin)
#undef FUNC_COMPUTE_COSIN
    public:
        static constexpr Scalar avg_normal_coef {Scalar(0.5)};
        // Cos and sin values of a circle divided into six
        static constexpr std::array<Scalar, 6> cos_values = compute_cos_values<Scalar>();
        static constexpr std::array<Scalar, 6> sin_values = compute_sin_values<Scalar>();
    };
    /// Generates the triangles used by the CNC Fit using HexagramGeneration
    template <typename P>
    struct TriangleGenerator<HexagramGeneration, P> : protected HexagramBase<P> {
        using VectorType = typename P::VectorType;
        using Scalar = typename P::Scalar;

        template <typename IndexRange, typename PointContainer>
        static int generate(
            const IndexRange& ids,
            const PointContainer& points,
            const VectorType& _evalPointPos, const VectorType& _evalPointNormal,
            std::vector<Triangle<P>>& triangles
        )
        {
            // Compute normal and maximum distance.
            VectorType c = _evalPointPos;
            VectorType n = _evalPointNormal;
            VectorType a {VectorType::Zero()};
            Scalar avg_d = Scalar(0);

            for ( int index : ids ) {
                auto p = points[ index ];
                avg_d += ( p.pos() - c ).norm();
                a     += p.normal();
            }

            a     /= a.norm();
            n      = ( Scalar(1) - HexagramBase<P>::avg_normal_coef ) * n + HexagramBase<P>::avg_normal_coef * a;
            n     /= n.norm();
            avg_d /= ids.size();

            // Define basis for sector analysis.
            const int m = ( std::abs( n[0] ) > std::abs ( n[1] ))
                    ? ( ( std::abs( n[0] ) ) > std::abs( n[2] ) ? 0 : 2 )
                    : ( ( std::abs( n[1] ) ) > std::abs( n[2] ) ? 1 : 2 ) ;

            const VectorType e =
                ( m == 0 ) ? VectorType( 0, 1, 0 ) :
                ( m == 1 ) ? VectorType( 0, 0, 1 ) :
                             VectorType( 1, 0, 0 ) ;

            VectorType u = n.cross( e );
            VectorType v = n.cross( u );
            u /= u.norm();
            v /= v.norm();

            std::array<VectorType, 6> positions {c,c,c,c,c,c};
            std::array<VectorType, 6> normals   {n,n,n,n,n,n};

            std::array< Scalar    ,    6 > _distance2;
            std::array< VectorType,    6 > _targets;

            for ( int i = 0 ; i < 6 ; i++ ) {
                _distance2 [ i ] = avg_d * avg_d;
                _targets   [ i ] = avg_d * ( u * HexagramBase<P>::cos_values[i] + v * HexagramBase<P>::sin_values[i] );
            }

            // Compute closest points.
            for ( int index : ids ) {
                VectorType p = points[ index ].pos();
                const VectorType d = p - c;

                for ( int j = 0 ; j < 6 ; j++ ){
                    const Scalar d2 = ( d - _targets[ j ]).squaredNorm();
                    if ( d2 < _distance2[ j ] ){
                        _distance2 [ j ] = d2;
                        positions[ j ] = p;
                        normals  [ j ] = points[ index ].normal();
                    }
                }
            }
            triangles.push_back(internal::Triangle<P>({positions[0] , positions[2] , positions[4]}, {normals[0] , normals[2], normals[4]}));
            triangles.push_back(internal::Triangle<P>({positions[1] , positions[3] , positions[5]}, {normals[1] , normals[3], normals[5]}));

            return 2;
        }
    };

    /// Generates the triangles used by the CNC Fit using AvgHexagramGeneration
    template <typename P>
    struct TriangleGenerator<AvgHexagramGeneration, P> : protected HexagramBase<P> {
        using VectorType = typename P::VectorType;
        using Scalar = typename P::Scalar;

        template <typename IndexRange, typename PointContainer>
        static int generate(
            const IndexRange& ids,
            const PointContainer& points,
            const VectorType& evalPointPos, const VectorType& evalPointNormal,
            std::vector<Triangle<P>>& triangles
        ) {
            // Compute normal and maximum distance.
            VectorType c = evalPointPos;
            VectorType n = evalPointNormal;
            VectorType a = evalPointNormal;
            Scalar avg_d = Scalar(0);

            std::array< VectorType,    6 > _targets;
            Scalar avg_normal  = Scalar(0.5);

            for ( int index : ids ) {
                a     += points[ index ].normal();
                avg_d += ( points[ index ].pos() - c ).norm();
            }

            a     /= a.norm();
            n      = ( Scalar(1) - HexagramBase<P>::avg_normal_coef ) * n + HexagramBase<P>::avg_normal_coef * a;
            n     /= n.norm();
            avg_d /= ids.size();

            // Define basis for sector analysis.
            const int m = ( std::abs( n[0] ) > std::abs ( n[1] ))
                    ? ( ( std::abs( n[0] ) ) > std::abs( n[2] ) ? 0 : 2 )
                    : ( ( std::abs( n[1] ) ) > std::abs( n[2] ) ? 1 : 2 );

            const VectorType e = ( m == 0 ) ? VectorType( 0, 1, 0 ) :
                                 ( m == 1 ) ? VectorType( 0, 0, 1 ) :
                                              VectorType( 1, 0, 0 ) ;
            VectorType u = n.cross( e );
            VectorType v = n.cross( u );
            u /= u.norm();
            v /= v.norm();

            // Initialize the average values
            std::array< VectorType, 6 > array_avg_normals;
            std::array< VectorType, 6 > array_avg_pos;
            std::array< int, 6 >    array_nb {};
            for (int i = 0 ; i < 6 ; i++ ) {
                _targets[ i ]          = avg_d * ( u * HexagramBase<P>::cos_values[i] + v * HexagramBase<P>::sin_values[i] );
                array_avg_normals[ i ] = VectorType::Zero();
                array_avg_pos    [ i ] = VectorType::Zero();
            }

            // Compute closest points.
            for (int index : ids) {
                VectorType p = points[ index ].pos() - c;
                int best_k = 0;
                Scalar best_d2 = ( p - _targets[ 0 ] ).squaredNorm();
                for (int k = 1 ; k < 6 ; k++) {
                    const Scalar d2 = ( p - _targets[ k ] ).squaredNorm();
                    if ( d2 < best_d2 ) {
                        best_k = k;
                        best_d2 = d2;
                    }
                }
                array_avg_normals[ best_k ] += points[ index ].normal();
                array_avg_pos    [ best_k ] += points[ index ].pos();
                array_nb[ best_k ] += 1;
            }

            for (int i = 0 ; i < 6 ; i++) {
                if ( array_nb[ i ] == 0 ) {
                    array_avg_normals[ i ] = n;
                    array_avg_pos    [ i ] = c;
                } else {
                    array_avg_normals[ i ] /= array_avg_normals[ i ].norm();
                    array_avg_pos    [ i ] /= array_nb[ i ];
                }
            }

            triangles.push_back(internal::Triangle<P>(
                { array_avg_pos[0] , array_avg_pos[2] , array_avg_pos[4] },
                { array_avg_normals[0], array_avg_normals[2], array_avg_normals[4] }
            ));
            triangles.push_back(internal::Triangle<P>(
                { array_avg_pos[1] , array_avg_pos[3] , array_avg_pos[5] },
                { array_avg_normals[1], array_avg_normals[3], array_avg_normals[5] }
            ));
            return 2;
        }
    };
} // namespace Ponca::internal

namespace Ponca {
    template < class P, TriangleGenerationMethod M>
    template <typename PointContainer>
    FIT_RESULT CNC<P, M>::compute( const PointContainer& points ) {
        init();
        internal::BoundedIntRange indicesSample( points.size() );
        _nb_vt = internal::TriangleGenerator<M, P>::generate( indicesSample, points, _evalPointPos, _evalPointNormal, _triangles);

        return finalize();
    }

    template < class P, TriangleGenerationMethod M>
    template <typename IndexRange, typename PointContainer>
    FIT_RESULT CNC<P, M>::computeWithIds( const IndexRange& ids, const PointContainer& points ) {
        init();
        _nb_vt = internal::TriangleGenerator<M, P>::generate( ids, points, _evalPointPos, _evalPointNormal, _triangles);
        return finalize();
    }

    template < class P, TriangleGenerationMethod M>
    FIT_RESULT CNC<P, M>::finalize( ) {
        _A = Scalar(0);
        _H = Scalar(0);
        _G = Scalar(0);

        MatrixType localT = MatrixType::Zero();

        for (int t = 0; t < _nb_vt; ++t) {
            // Simple estimation.
            Scalar tA = _triangles[t].mu0InterpolatedU();
            if (tA < -internal::CNCEigen<P>::epsilon) {
                _A     -= tA;
                _H     += _triangles[t].template mu1InterpolatedU<true>();
                _G     += _triangles[t].template mu2InterpolatedU<true>();
                localT += _triangles[t].template muXYInterpolatedU<true>();
            } else if (tA > internal::CNCEigen<P>::epsilon) {
                _A     += tA;
                _H     += _triangles[t].mu1InterpolatedU();
                _G     += _triangles[t].mu2InterpolatedU();
                localT += _triangles[t].muXYInterpolatedU();
            }

        } // end for t

        _T11 = localT(0,0);
        _T12 = 0.5 * (localT(0,1) + localT(1,0));
        _T13 = 0.5 * (localT(0,2) + localT(2,0));
        _T22 = localT(1,1);
        _T23 = 0.5 * (localT(1,2) + localT(2,1));
        _T33 = localT(2,2);

        MatrixType T;
        if (_A != Scalar(0)) {
            T  << _T11, _T12, _T13,
                  _T12, _T22, _T23,
                  _T13, _T23, _T33;
            T  /= _A;
            _H /= _A;
            _G /= _A;
        } else {
            _H = Scalar(0);
            _G = Scalar(0);
        }

        std::tie (k2, k1, v2, v1) = internal::CNCEigen<P>::curvaturesFromTensor(T, 1.0, _evalPointNormal);

        return STABLE;
    }
} // namespace Ponca
