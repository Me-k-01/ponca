
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
    for (int i = 0; i < points.size(); ++i)
    {
        const int singleDimIndex = i*DataPoint::Dim;
        for (int d = 0; d < DataPoint::Dim; ++d)
        {
            positionsOutput[singleDimIndex + d] = points[i].pos()[d];
            normalsOutput  [singleDimIndex + d] = points[i].normal()[d];
        }
    }
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
__device__ DataPoint makeDataPoint(const int index, const typename DataPoint::Scalar * positions, const typename DataPoint::Scalar * normals)
{
    using VectorType = typename DataPoint::VectorType;

    VectorType position = VectorType::Zero();
    VectorType normal   = VectorType::Zero();

    const int singleDimIndex = index * DataPoint::Dim;
    for (int d = 0; d < DataPoint::Dim; ++d)
    {
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
__host__ void flattenedArrayToPoints(const typename DataPoint::Scalar * positions, const typename DataPoint::Scalar * normals, PointContainer & pointsOutput)
{
    using VectorType = typename DataPoint::VectorType;

    for (int i = 0; i < pointsOutput.size(); ++i)
    {
        VectorType position = VectorType::Zero();
        VectorType normal   = VectorType::Zero();

        const int singleDimIndex = i*DataPoint::Dim;
        for (int d = 0; d < DataPoint::Dim; ++d)
        {
            position.row(d) << positions[singleDimIndex + d];
            normal.row(d)   << normals  [singleDimIndex + d];
        }
        pointsOutput[i] = DataPoint(position, normal);
    }
}

/*!
 * \brief Computes a fit for each point of the point cloud.
 *
 * \tparam DataPoint The DataPoint type.
 * \tparam Fit The Fit that will be computed by the Kernel.
 * \param positions As an Input, and array of positions.
 * \param normals As an Input, an array of normals.
 * \param analysisScale The radius of the neighborhood.
 * \param nbPoints The total number of points in the point cloud.
 */
template<typename DataPoint, typename Fit>
__global__ void fitKernel(const typename DataPoint::Scalar* positions, const typename DataPoint::Scalar* normals, const typename DataPoint::Scalar analysisScale, const int nbPoints)
{
    // Get global index
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    // Skip when not in the point cloud
    if (i >= nbPoints)
        return;

    // Make the evaluation point of the fit
    DataPoint evalPoint = makeDataPoint<DataPoint>(i, positions, normals);

    // Set up the fit
    Fit fit;
    fit.setNeighborFilter({ evalPoint.pos(), analysisScale });

    // Computes the fit
    fit.init();
    for (int j = 0; j < nbPoints; ++j) {
        fit.addNeighbor( makeDataPoint<DataPoint>(j, positions, normals) );
    }
    fit.finalize();

    // TODO : return the ouput values
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

    // Convert data to flattened arrays
    Scalar positions[nbPoints*Dim];
    Scalar normals  [nbPoints*Dim];
    pointsToFlattenedArray<DataPoint>(vectorPoints, positions, normals);

    // Send to device
    float* positionsDevice;
    float* normalsDevice;
    cudaMalloc(&positionsDevice, nbPoints*Dim);
    cudaMalloc(&normalsDevice  , nbPoints*Dim);
    cudaMemcpy(positionsDevice, positions, nbPoints*Dim, cudaMemcpyHostToDevice);
    cudaMemcpy(normalsDevice  , normals  , nbPoints*Dim, cudaMemcpyHostToDevice);

    // Start kernel
    fitKernel<DataPoint, MeanFitSmooth><<<1, 256>>>(positionsDevice, normalsDevice, analysisScale, nbPoints);

    cudaDeviceSynchronize();

    cudaFree(positionsDevice);
    cudaFree(normalsDevice);
}


__host__ int main(int argc, char** argv) {
    if(!init_testing(argc, argv))
        return EXIT_FAILURE;

    std::cout << "Test plane fitting on cuda..." << std::endl;
    testPlaneCuda<float, 3>();
}
