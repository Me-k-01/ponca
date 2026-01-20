#pragma once

/*!
 * \brief Extract a vector from a flattened array of vectors
 *
 * \tparam DataPoint The DataPoint type containing the VectorType and the number of dimensions.
 * \param idx The id of the vector that needs to be extracted (the index in the flattened array corresponds to : idx * number of dimensions).
 * \param flattenedVectorArray The flattened vector array, that is of size : total number of vector * Dimension.
 * \return The vector that was extracted from the flattened vector array.
 */
template <typename DataPoint>
PONCA_MULTIARCH typename DataPoint::VectorType extractVectorFromFlattenedArray(
    const int idx,
    const typename DataPoint::Scalar * const flattenedVectorArray
) {
    using VectorType = typename DataPoint::VectorType;
    const int singleDimIndex = idx * DataPoint::Dim;
    VectorType v;

    for (int d = 0; d < DataPoint::Dim; ++d) {
        v.row(d) << flattenedVectorArray[singleDimIndex + d];
    }

    return v;
}

/*!
 * \brief Variant of the MyPoint class allowing to work with external raw data.
 *
 * Using this approach, one can use the Ponca library with already existing
 * data-structures without any data-duplication.
 *
 * In this example, we use this class to map an interlaced raw array containing
 * both point normals and coordinates.
 */
template<typename _Scalar, int _Dim>
class PointRef
{
public:
    enum {Dim = _Dim};
    typedef _Scalar Scalar;
    typedef Eigen::Matrix<Scalar, Dim, 1>   VectorType;
    typedef Eigen::Matrix<Scalar, Dim, Dim> MatrixType;

    PONCA_MULTIARCH inline PointRef(Scalar* _interlacedArray, int _pId)
        : m_pos (Eigen::Map< const VectorType >(_interlacedArray + Dim*2*_pId  )),
        m_normal(Eigen::Map< const VectorType >(_interlacedArray + Dim*2*_pId+Dim))
    {}

    PONCA_MULTIARCH inline const Eigen::Map< const VectorType >& pos()    const { return m_pos; }
    PONCA_MULTIARCH inline const Eigen::Map< const VectorType >& normal() const { return m_normal; }

private:
    Eigen::Map< const VectorType > m_pos, m_normal;
};
