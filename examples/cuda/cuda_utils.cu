#pragma once

/*! \brief Converts a STL-like container of DataPoint to flattened arrays of positions and normals (only one dimension).
 *
 * \tparam DataPoint The DataPoint type.
 * \tparam PointContainer A STL-like container of DataPoint.
 * \param points As the input, an STL-like container that contains the point position and normal.
 * \param positionsOutput As an output, the flattened positions array.
 * \param normalsOutput As an output, the flattened normal array.
 */
template<typename DataPoint, typename PointContainer>
__host__ void pointsToFlattenedArray(
    PointContainer & points,
    typename DataPoint::Scalar * const positionsOutput,
    typename DataPoint::Scalar * const normalsOutput
) {
    for (int i = 0; i < points.size(); ++i) {
        const int singleDimIndex = i*DataPoint::Dim;
        for (int d = 0; d < DataPoint::Dim; ++d)
        {
            positionsOutput[singleDimIndex + d] = points[i].pos()[d];
            normalsOutput  [singleDimIndex + d] = points[i].normal()[d];
        }
    }
}

/*! \brief Extract a vector from a flattened array of vectors
 *
 * \tparam DataPoint The DataPoint type containing the VectorType and the number of dimensions.
 * \param idx The id of the vector that needs to be extracted (the index in the flattened array corresponds to : idx * number of dimensions).
 * \param flattenedVectorArray The flattened vector array, that is of size : total number of vector * Dimension.
 * \return The vector that was extracted from the flattened vector array.
 */
template <typename DataPoint>
__host__ __device__ typename DataPoint::VectorType extractVectorFromFlattenedArray(
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

/*! \brief Make a DataPoint from the positions and normals flattened array
 *
 * \tparam DataPoint The DataPoint type containing the VectorType and the number of Dimension.
 * \param idx The id of the vector that needs to be extracted (the index in the flattened array corresponds to : idx * number of dimensions).
 * \param positions As an input, the flattened positions array.
 * \param normals As an input, the flattened normal array.
 */
template<typename DataPoint>
__host__ __device__ DataPoint makeDataPoint(
    const int idx,
    const typename DataPoint::Scalar * const positions,
    const typename DataPoint::Scalar * const normals
) {
    using VectorType = typename DataPoint::VectorType;

    VectorType position = extractVectorFromFlattenedArray<DataPoint>(idx, positions);
    VectorType normal   = extractVectorFromFlattenedArray<DataPoint>(idx, normals);

    return DataPoint(position, normal);
}

/*! \brief Converts a flattened arrays of positions and normals (one dimension) to a STL-like container of DataPoint.
 *
 * \warning The point container must be of the correct sized, because `points.size()` is used to determine the number
 * of points in the arrays.
 *
 * \tparam DataPoint The DataPoint type containing the VectorType and the number of Dimension.
 * \tparam PointContainer A STL-like container of DataPoint.
 * \param positions As an input, the flattened positions array.
 * \param normals As an input, the flattened normal array.
 * \param pointsOutput As an output, an STL-like container that contains the point position and normal.
 */
template<typename DataPoint, typename PointContainer>
__host__ void flattenedArrayToPoints(
    const typename DataPoint::Scalar * const positions,
    const typename DataPoint::Scalar * const normals,
    PointContainer & pointsOutput
) {
    for (int i = 0; i < pointsOutput.size(); ++i) {
        pointsOutput[i] = makeDataPoint<DataPoint>(i, positions, normals);
    }
}
