
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

template<typename Fit, typename Tree, typename Scalar>
__global__ void fitKernel(const Tree& tree, const Scalar analysisScale, const int nbPoints)
{
    // Global index
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nbPoints)
        return;

    // Fit fit;
    // fit.setWeightFunc({ vectorPoints[i].pos(), analysisScale });
    // fit.init();

    // TODO : fit compute and exit values
}

template<typename Scalar, int Dim>
__global__ int testPlaneCuda(bool _bUnoriented = false, bool _bAddPositionNoise = false, bool _bAddNormalNoise = false)
{
    typedef PointPositionNormal<Scalar, Dim> DataPoint;
    typedef Ponca::DistWeightFunc<DataPoint, Ponca::SmoothWeightKernel<Scalar> > WeightSmoothFunc;
    typedef Ponca::Basket<DataPoint, WeightSmoothFunc, Ponca::MeanPlaneFit> MeanFitSmooth;
    typedef typename DataPoint::VectorType VectorType;

    int nbPoints = Eigen::internal::random<int>(100, 1000);

    Scalar width  = Eigen::internal::random<Scalar>(1., 10.);
    Scalar height = width;

    Scalar analysisScale = Scalar(15.) * std::sqrt( width * height / nbPoints);
    Scalar centerScale   = Eigen::internal::random<Scalar>(1,10000);
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

    Ponca::KdTreeDense<DataPoint> tree(vectorPoints); // TODO : pass this to the device

    std::cout << "Plane fit" << std::endl;
    fitKernel<MeanFitSmooth><<<1, 256>>>(tree, analysisScale, nbPoints);
    cudaDeviceSynchronize();
}


__host__ int main(int argc, char** argv) {
    if(!init_testing(argc, argv))
        return EXIT_FAILURE;

    std::cout << "Test plane fitting on cuda..." << std::endl;
    testPlaneCuda<float, 3>();
}
