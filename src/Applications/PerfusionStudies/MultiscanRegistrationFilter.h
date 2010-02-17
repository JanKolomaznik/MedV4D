/**
 * @author Attila Ulman
 * @file MultiscanRegistrationFilter.h 
 * @{ 
 **/

#ifndef MULTISCAN_REGISTRATION_FILTER_H
#define MULTISCAN_REGISTRATION_FILTER_H

#include "Imaging/AImageFilterWholeAtOnce.h"
#include "Imaging/MultiHistogram.h"
#include "Imaging/criterion/NormalizedMutualInformation.h"
#include "Imaging/optimization/PowellOptimization.h"


namespace M4D {
namespace Imaging {

#define	HISTOGRAM_MIN_VALUE						0
#define HISTOGRAM_MAX_VALUE						200
#define HISTOGRAM_VALUE_DIVISOR       10

#define MIN_SAMPLING							    128
#define TRANSFORM_SAMPLING						256

enum InterpolationType {
	IT_NEAREST,
	IT_LINEAR
};

typedef Vector< float32, 2 > CoordType;
typedef float64 HistCellType;

template< typename ElementType >
struct SliceInfo;

struct TransformationInfo2D;

template< typename ElementType >
class Interpolator2D;

template< typename ImageType >
class MultiscanRegistrationFilter;

template< typename ElementType >
class MultiscanRegistrationFilter< Image< ElementType, 3 > >
	: public AImageFilterWholeAtOnceIExtents< Image< ElementType, 3 >, Image< ElementType, 3 > >
{
  public:	

	  typedef Image< ElementType, 3 > InputImageType;
	  typedef Image< ElementType, 3 > OutputImageType;
	  typedef AImageFilterWholeAtOnceIExtents< InputImageType, OutputImageType > PredecessorType;

	  struct Properties: public PredecessorType::Properties
	  {
		  Properties ()
        : examinedSliceNum( EXEMINED_SLICE_NUM ), boneDensityBottom( BONE_DENSITY_BOTTOM ), 
          boneDensityTop( BONE_DENSITY_TOP ), interpolationType( IT_NEAREST ) 
      {}

		  uint32 examinedSliceNum;
      ElementType	boneDensityBottom, boneDensityTop;
      InterpolationType	interpolationType;
	  };

	  MultiscanRegistrationFilter ( Properties * prop );
	  MultiscanRegistrationFilter ();
    ~MultiscanRegistrationFilter ();

    /**
	   * The optimization function that is to be optimized to align the images.
     *
	   * @param v the vector of input parameters of the optimization function
	   * @return the return value of the optimization function
	   */
	  double OptimizationFunction ( Vector< double, 3 > &v );

	  GET_SET_PROPERTY_METHOD_MACRO(uint32, ExaminedSliceNum, examinedSliceNum);
    GET_SET_PROPERTY_METHOD_MACRO(ElementType, BoneDensityBottom, boneDensityBottom);
    GET_SET_PROPERTY_METHOD_MACRO(ElementType, BoneDensityTop, boneDensityTop);
    GET_SET_PROPERTY_METHOD_MACRO(InterpolationType, InterpolationType, interpolationType);
  
  protected:

	  bool ProcessImage ( const InputImageType &in, OutputImageType &out );

	  bool ProcessImageHelper ( const InputImageType &in, OutputImageType &out );

    bool RegisterSlice ();

  private:

	  GET_PROPERTIES_DEFINITION_MACRO;

    Interpolator2D< ElementType > *interpolator;

    SliceInfo< ElementType > *inSlice, *outSlice, *refSlice;

    TransformationInfo2D transformationInfo;

    MultiHistogram< HistCellType, 2 > jointHistogram;

    CriterionBase< HistCellType > *criterionComponent;

    OptimizationBase< MultiscanRegistrationFilter< InputImageType >, double, 3 > *optimizationComponent;
};

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

template< typename ElementType >
struct SliceInfo
{
	SliceInfo () {}

	SliceInfo ( ElementType *pointer, Vector< uint32, 2 > &size, Vector< int32, 2 > &stride, Vector< float32, 2 > &extent ) 
    : pointer( pointer ), size( size ), stride( stride ), extent( extent )
	{}

	ElementType *pointer;
  Vector< uint32, 2 > size;
  Vector< int32, 2 > stride;
  Vector< float32, 2 > extent;
};

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

struct TransformationInfo2D
{
	TransformationInfo2D () 
  {
    Reset();
  }

	TransformationInfo2D ( CoordType &translation, float32 rotation, uint32 sampling )
    : translation( translation ), rotation( rotation ), sampling( sampling )
	{}

  void SetParams ( CoordType &trans, float32 rot )
  {
    for ( uint32 i = 0; i < 2; i++ ) {
      translation[i] = trans[i];
    }

    rotation = rot;
  }

  void Reset ()
  {
  	for ( uint32 i = 0; i < 2; i++ ) {
		  translation[i] = 0.0f;
    }

    rotation = 0.0f;

    sampling = TRANSFORM_SAMPLING;
  }

	CoordType translation;
  float32 rotation;
  uint32 sampling;
};

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

template< typename ElementType >
class Interpolator2D
{
  public:

    Interpolator2D ()
      : pointer( NULL )
    {}

	  Interpolator2D ( ElementType *p, Vector< int32, 2 > &s )
      : pointer( p ), stride( s )
	  {}

    void SetParams ( ElementType *p, Vector< int32, 2 > &s )
    {
      pointer = p;

      for ( uint32 i = 0; i < 2; i++ ) {
		    stride[i] = s[i];
      }
    }

	  virtual ElementType Get ( CoordType &coords ) = 0;

    ElementType *pointer;
	  Vector< int32, 2 > stride;
};


template< typename ElementType >
class NearestInterpolator2D: public Interpolator2D< ElementType >
{
  public:

    typedef Interpolator2D< ElementType > PredecessorType;

    NearestInterpolator2D () 
      : PredecessorType() 
    {}

	  NearestInterpolator2D ( ElementType *pointer, Vector< int32, 2 > &stride )
      : PredecessorType( pointer, stride ) 
	  {}

	  ElementType Get ( CoordType &coords )
	  { 
      return *(pointer + (ROUND(coords[0]) * stride[0] + ROUND(coords[1]) * stride[1]));
    }
};


template< typename ElementType >
class LinearInterpolator2D: public Interpolator2D< ElementType >
{
  public:

    typedef Interpolator2D< ElementType > PredecessorType;

    LinearInterpolator2D () 
      : PredecessorType() 
    {}

	  LinearInterpolator2D ( ElementType *pointer, Vector< int32, 2 > &stride )
      : PredecessorType( pointer, stride ) 
	  {}

	  ElementType Get ( CoordType &coords )
	  { 
	    double temp;

	    // calculate the racional part of the coordinate values
	    double cx = std::modf( (double)coords[0], &temp );
	    double cy = std::modf( (double)coords[1], &temp );

	    // calculate the proportions of each point
	    double w[ 4 ];

	    w[0] = (1 - cx) * (1 - cy);
	    w[1] = cx * (1 - cy);
	    w[2] = (1 - cx) * cy;
	    w[3] = cx * cy;

	    uint32 floor[ 2 ], ceil[ 3 ];

	    // calculate floor positions
	    floor[0] = (uint32)std::floor( coords[0] ) * stride[0];
	    floor[1] = (uint32)std::floor( coords[1] ) * stride[1];

	    // calculate ceil positions
	    ceil[0] = (uint32)std::ceil( coords[0] ) * stride[0];
	    ceil[1] = (uint32)std::ceil( coords[1] ) * stride[1];

	    // calculate the interpolated value and return
	    return *(pointer + (floor[0] + floor[1])) * w[0] + *(pointer + (ceil[0] + floor[1])) * w[1] +
		         *(pointer + (floor[0] +  ceil[1])) * w[2] + *(pointer + (ceil[0] +  ceil[1])) * w[3];
    }
};

} // namespace Imaging
} // namespace M4D


// include implementation
#include "MultiscanRegistrationFilter.tcc"

#endif // MULTISCAN_REGISTRATION_FILTER_H

/** @} */
