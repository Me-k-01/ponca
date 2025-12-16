
#include <Ponca/src/Fitting/basket.h>
#include <Ponca/src/Fitting/covariancePlaneFit.h>
#include <Ponca/src/Fitting/meanPlaneFit.h>
#include <Ponca/src/Fitting/weightFunc.h>
#include <Ponca/src/Fitting/weightKernel.h>
#include <Ponca/SpatialPartitioning>
#include <Ponca/src/SpatialPartitioning/KdTree/kdTree.h>
#include <vector>


#include "../../tests/common/testing.h"
#include "../../tests/common/testUtils.h"


/*!
 * \brief Converts a STL-like container of DataPoint to flattened arrays of positions and normals (one dimension).
 *
 * \tparam DataPoint The DataPoint type.
 * \tparam PointContainer A STL-like container of DataPoint.
 * \param points As the input, an STL-like container that contains the point position and normal.
 * \param positionsOutput As an output, the flattened positions array.
 * \param normalsOutput As an output, the flattened normal array.
 */
template<typename DataPoint, typename PointContainer>
__host__ void pointsToFlattenedArray(PointContainer & points, typename DataPoint::Scalar * positionsOutput, typename DataPoint::Scalar * normalsOutput)
{
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
 * \tparam DataPoint Contains the Dimension of the vector and the VectorType.
 * \param idx The id of the vector that needs to be extracted (the index of the vector in the flattened array is idx * Dimension).
 * \param flattenedVectorBuffer The flattened vector array, that is of size : total number of vector * Dimension.
 * \return The vector that was extracted from the flattened vector buffer.
 */
template <typename DataPoint>
__host__ __device__ typename DataPoint::VectorType extractVectorFromFlattenedArray(int idx, typename DataPoint::Scalar * flattenedVectorBuffer) {
    using VectorType = typename DataPoint::VectorType;
    const int singleDimIndex = idx * DataPoint::Dim;
    VectorType v;

    for (int d = 0; d < DataPoint::Dim; ++d) {
        v.row(d) << flattenedVectorBuffer[singleDimIndex + d];
    }

    return v;
}

/*!
 * \brief Make a DataPoint from the positions and normals array
 *
 * \tparam DataPoint The DataPoint type.
 * \param index The index of the point.
 * \param positions As an input, the flattened positions array.
 * \param normals As an input, the flattened normal array.
 */
template<typename DataPoint>
__host__ __device__ DataPoint makeDataPoint(
    const int index,
    const typename DataPoint::Scalar * positions,
    const typename DataPoint::Scalar * normals
) {
    using VectorType = typename DataPoint::VectorType;

    VectorType position = VectorType::Zero();
    VectorType normal   = VectorType::Zero();

    const int singleDimIndex = index * DataPoint::Dim;
    for (int d = 0; d < DataPoint::Dim; ++d) {
        position.row(d) << positions[singleDimIndex + d];
        normal.row(d)   << normals  [singleDimIndex + d];
    }

    return DataPoint(position, normal);
}

/*!
 * \brief Converts a flattened arrays of positions and normals (one dimension) to a STL-like container of DataPoint.
 *
 * \warning The point container must be of the correct sized, because `points.size()` is used to determine the number
 * of points in the arrays.
 *
 * \tparam DataPoint The DataPoint type.
 * \tparam PointContainer A STL-like container of DataPoint.
 * \param positions As an input, the flattened positions array.
 * \param normals As an input, the flattened normal array.
 * \param pointsOutput As an output, an STL-like container that contains the point position and normal.
 */
template<typename DataPoint, typename PointContainer>
__host__ void flattenedArrayToPoints(
    const typename DataPoint::Scalar * positions,
    const typename DataPoint::Scalar * normals,
    PointContainer & pointsOutput
) {
    for (int i = 0; i < pointsOutput.size(); ++i) {
        pointsOutput[i] = makeDataPoint<DataPoint>(i, positions, normals);
    }
}

/*!
 * \brief Computes a fit for each point of the point cloud and returns the potential result.
 *
 * \tparam DataPoint The DataPoint type.
 * \tparam Fit The Fit that will be computed by the Kernel.
 * \param positions As an Input, the array of positions of the point cloud.
 * \param normals As an Input, the array of normals of the point cloud.
 * \param analysisScale The radius of the neighborhood.
 * \param nbPoints The total number of points in the point cloud.
 * \param potentialResults As an Output, the potential results of the fit for each point of the Point Cloud.
 * \param gradiantResults As an Output, the primitiveGradient results of the fit for each point of the Point Cloud.
 */
template<typename DataPoint, typename Fit, typename Scalar>
__global__ void fitPotentialKernel(
    const Scalar* positions,
    const Scalar* normals,
    const Scalar analysisScale,
    const int nbPoints,
    Scalar* potentialResults,
    Scalar* gradiantResults

) {
    using VectorType = typename DataPoint::VectorType;

    // Get global index
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Skip when not in the point cloud
    if (i >= nbPoints) return;

    // Make the evaluation point of the fit
    const auto evalPoint = makeDataPoint<DataPoint>(i, positions, normals);

    // Set up the fit
    Fit fit;
    fit.setNeighborFilter({ evalPoint.pos(), analysisScale });

    // Computes the fit
    fit.init();
    for (int j = 0; j < nbPoints; ++j) {
        fit.addNeighbor( makeDataPoint<DataPoint>(j, positions, normals) );
    }
    fit.finalize();

    // Returns NaN if not stable
    if (! fit.isStable()) {
        potentialResults[i] = NAN;
        for (int d = 0; d < DataPoint::Dim; ++d) {
            gradiantResults[i*DataPoint::Dim + d] = NAN;
        }
        return;
    }

    // Return the fit.potential result as an output
    potentialResults[i] = fit.potential(evalPoint.pos());
    VectorType grad = fit.primitiveGradient(evalPoint.pos());
    for (int d = 0; d < DataPoint::Dim; ++d) {
        gradiantResults[i*DataPoint::Dim + d] = grad(d);
    }
}

template<typename Scalar, int Dim>
__host__ void testPlaneCuda(bool _bUnoriented = false, bool _bAddPositionNoise = false, bool _bAddNormalNoise = false)
{
    typedef PointPositionNormal<Scalar, Dim> DataPoint;
    typedef Ponca::DistWeightFunc<DataPoint, Ponca::SmoothWeightKernel<Scalar> > WeightSmoothFunc;
    typedef Ponca::Basket<DataPoint, WeightSmoothFunc, Ponca::MeanPlaneFit> MeanFitSmooth;
    typedef typename DataPoint::VectorType VectorType;

    int nbPoints = Eigen::internal::random<int>(100, 1000);

    Scalar width  = Eigen::internal::random<Scalar>(1., 10.);
    Scalar height = width;

    Scalar analysisScale = Scalar(15.) * std::sqrt( width * height / nbPoints);
    Scalar centerScale   = Eigen::internal::random<Scalar>(1, 10000);
    VectorType center    = VectorType::Random() * centerScale;

    VectorType direction = VectorType::Random().normalized();

    Scalar epsilon = testEpsilon<Scalar>();
    std::vector<DataPoint> vectorPoints(nbPoints);

    for(unsigned int i = 0; i < vectorPoints.size(); ++i)
    {
        vectorPoints[i] = getPointOnPlane<DataPoint>(center,
                                                     direction,
                                                     width,
                                                     _bAddPositionNoise,
                                                     _bAddNormalNoise,
                                                     _bUnoriented);
    }

    // Ponca::KdTreeDense<DataPoint> tree(vectorPoints); // TODO : pass this to the device

    auto scalarBufferSize = nbPoints*sizeof(Scalar);
    auto vectorBufferSize = scalarBufferSize*Dim;

    // Convert point vector to flattened arrays
    Scalar positions[nbPoints*Dim];
    Scalar normals  [nbPoints*Dim];
    pointsToFlattenedArray<DataPoint>(vectorPoints, positions, normals);

    // Send inputs to the device
    Scalar* positionsDevice;
    Scalar* normalsDevice;
    cudaMalloc(&positionsDevice, vectorBufferSize);
    cudaMalloc(&normalsDevice  , vectorBufferSize);
    cudaMemcpy(positionsDevice , positions, vectorBufferSize, cudaMemcpyHostToDevice);
    cudaMemcpy(normalsDevice   , normals  , vectorBufferSize, cudaMemcpyHostToDevice);

    // Prepare output buffers
    auto *potentialResults = new Scalar[nbPoints];
    auto *gradientResults  = new Scalar[nbPoints*Dim];
    Scalar* potentialResultsDevice;
    Scalar* gradientResultsDevice;
    cudaMalloc(&potentialResultsDevice, scalarBufferSize);
    cudaMalloc(&gradientResultsDevice , vectorBufferSize);

    int blockSize = 128;
    // The grid size needed, based on input size
    int gridSize = (nbPoints + blockSize - 1) / blockSize;

    // Computes the kernel
    fitPotentialKernel<DataPoint, MeanFitSmooth><<<gridSize, blockSize>>>(positionsDevice, normalsDevice, analysisScale, nbPoints, potentialResultsDevice, gradientResultsDevice);
    cudaDeviceSynchronize(); // Wait for the results

    // Fetch results (Device to Host)
    cudaMemcpy(potentialResults, potentialResultsDevice, scalarBufferSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(gradientResults , gradientResultsDevice , vectorBufferSize, cudaMemcpyDeviceToHost);

    // Free CUDA memory
    cudaFree(positionsDevice);
    cudaFree(normalsDevice);
    cudaFree(potentialResultsDevice);
    cudaFree(gradientResultsDevice);

    for (int j = 0; j < nbPoints; ++j) {
        VectorType primGrad = extractVectorFromFlattenedArray<DataPoint>(j, gradientResults);

        if(!_bAddPositionNoise) {
            std::cout << "j:" << j << ", potential:"<< potentialResults[j] << ", ";
            std::cout << "primitiveGradient:" << primGrad(0) << ", " << primGrad(1) << ", " << primGrad(2) << " ; " << std::endl;
            VERIFY(std::abs(potentialResults[j]) <= epsilon);
            VERIFY(Scalar(1.) - std::abs(primGrad.dot(direction)) <= epsilon);
        }
    }

    delete[] potentialResults;
    delete[] gradientResults;
}


__host__ int main(int argc, char** argv) {
    if(!init_testing(argc, argv))
        return EXIT_FAILURE;

    std::cout << "Test plane fitting on cuda..." << std::endl;
    testPlaneCuda<float, 3>();
}
