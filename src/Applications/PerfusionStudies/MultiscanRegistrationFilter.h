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

#define MIN_SAMPLING							    256
#define TRANSFORM_SAMPLING            256
  
#define DEGTORAD(a)                  (a * PI / 180.0) 

/// Type of the interpolator used during the transformation.
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

/**
 * Filter implementing multiscan, times series registration. 
 * 1st one is the reference slice, next slices in the time sequence are
 * transformed according to the first one.
 */
template< typename ElementType >
class MultiscanRegistrationFilter< Image< ElementType, 3 > >
	: public AImageFilterWholeAtOnceIExtents< Image< ElementType, 3 >, Image< ElementType, 3 > >
{
  public:	

	  typedef Image< ElementType, 3 > InputImageType;
	  typedef Image< ElementType, 3 > OutputImageType;
	  typedef AImageFilterWholeAtOnceIExtents< InputImageType, OutputImageType > PredecessorType;

    /**
     * Properties structure.
     */
	  struct Properties: public PredecessorType::Properties
	  {
      /**
       * Properties constructor - fills up the Properties with default values.
       */
		  Properties ()
        : examinedSliceNum( EXEMINED_SLICE_NUM ), registrationNeeded( false ),
          interpolationType( IT_NEAREST )
      {}

      /// Number of examined slices (number of time series).
		  uint32 examinedSliceNum;
      /// Flag indicating whether the registration is needed.
      bool registrationNeeded;
      /// Type of the selected interpolation.
      InterpolationType	interpolationType;
	  };

    /**
     * Registration filter constructor.
     *
     * @param prop pointer to the properties structure
     */
	  MultiscanRegistrationFilter ( Properties *prop );

    /**
     * Registration filter constructor.
     */
	  MultiscanRegistrationFilter ();

    /**
     * Registration filter destructor.
     */
    ~MultiscanRegistrationFilter ();

    /**
	   * The optimization function that is to be optimized to align the images.
     *
	   * @param v reference to the vector of input parameters of the optimization function
	   * @return the return value of the optimization function
	   */
	  double OptimizationFunction ( Vector< double, 3 > &v );

    /**
	   * Getter, setter macros for Properties attributes.
	   */
	  GET_SET_PROPERTY_METHOD_MACRO(uint32, ExaminedSliceNum, examinedSliceNum);
    GET_SET_PROPERTY_METHOD_MACRO(bool, RegistrationNeeded, registrationNeeded);
    GET_SET_PROPERTY_METHOD_MACRO(InterpolationType, InterpolationType, interpolationType);
  
  protected:

    /**
	   * The method executed by the pipeline's filter execution thread.
     *
	   *  @param in reference to the input image dataset
     *  @param out reference to the output image dataset
     *  @return true if finished successfully, false otherwise
     */
	  bool ProcessImage ( const InputImageType &in, OutputImageType &out );

    /**
	   * Method for managing registration functionality - called from ProcessImage.
     *
	   *  @param in reference to the input image dataset
     *  @param out reference to the output image dataset
     *  @return true if finished successfully, false otherwise
     */
	  bool ProcessImageHelper ( const InputImageType &in, OutputImageType &out );

    /**
	   * Registers (multiresolution) inSlice to outSlice according to refSlice.
     */
    bool RegisterSlice ();

  private:

	  GET_PROPERTIES_DEFINITION_MACRO;

    /// Pointer to the selected type of interpolation used for transformation.
    Interpolator2D< ElementType > *interpolator;

    /// Pointers to actual input, ouput and reference slices.
    SliceInfo< ElementType > *inSlice, *outSlice, *refSlice;

    /// Actual transformation parameters.
    TransformationInfo2D transformationInfo;

    /// Joint histogram of two images.   
    MultiHistogram< HistCellType, 2 > jointHistogram;

    /// Pointer to the criterion component.
    CriterionBase< HistCellType > *criterionComponent;

    /// Pointer to the optimization component.
    OptimizationBase< MultiscanRegistrationFilter< InputImageType >, double, 3 > *optimizationComponent;
};

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

/**
 * Class representing slice for easier manipulation.
 */
template< typename ElementType >
struct SliceInfo
{
  /**
   * SliceInfo constructor.
   */
	SliceInfo () {}

  /**
   * SliceInfo constructor.
   *
   * @param pointer pointer to the data of the slice
   * @param size reference to the dimensions of the slice
   * @param stride reference to the strides of the slice
   * @param extent reference to the extents of the slice
   */
	SliceInfo ( ElementType *pointer, Vector< uint32, 2 > &size, Vector< int32, 2 > &stride, Vector< float32, 2 > &extent ) 
    : pointer( pointer ), size( size ), stride( stride ), extent( extent )
	{}

	/// Pointer to the data of the slice.
  ElementType *pointer;
  /// Dimensions of the slice.
  Vector< uint32, 2 > size;
  /// Strides of the slice.
  Vector< int32, 2 > stride;
  /// Extents of the slice.
  Vector< float32, 2 > extent;
};

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

/**
 * Class for holding actual transformation parameters.
 */
struct TransformationInfo2D
{
  /**
   * TransformationInfo2D constructor.
   */
	TransformationInfo2D () 
  {
    Reset();
  }

  /**
   * TransformationInfo2D constructor.
   *
   * @param translation reference to the translation component of the transformation
   * @param rotation rotation component of the transformation
   * @param sampling sampling used during the iterative registration
   */
	TransformationInfo2D ( CoordType &translation, float32 rotation, uint32 sampling )
    : translation( translation ), rotation( rotation ), sampling( sampling )
	{}

  /**
   * Sets parameters of the structure.
   *
   * @param translation reference to the translation component of the transformation
   * @param rotation rotation component of the transformation
   */
  void SetParams ( CoordType &trans, float32 rot )
  {
    for ( uint32 i = 0; i < 2; i++ ) {
      translation[i] = trans[i];
    }

    rotation = rot;
  }

  /**
   * Resets the attributes of the structure.
   */
  void Reset ()
  {
  	for ( uint32 i = 0; i < 2; i++ ) {
		  translation[i] = 0.0f;
    }

    rotation = 0.0f;

    sampling = TRANSFORM_SAMPLING;
  }

  /// Translation component of the transformation.
	CoordType translation;
  /// Rotation component of the transformation.
  float32 rotation;
  /// Sampling used during the hierarchal/iterative registration (of transformation and histogram calculation) 
  uint32 sampling;
};

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

/**
 * Abstract base class for all kinds of interpolators.
 */
template< typename ElementType >
class Interpolator2D
{
  public:

    /**
     * Interpolator2D constructor.
     */
    Interpolator2D ()
      : pointer( NULL )
    {}

    /**
     * Interpolator2D constructor.
     *
     * @param p pointer to the data to be interpolated
     * @param s reference to the strides of the data to be interpolated
     */
	  Interpolator2D ( ElementType *p, Vector< int32, 2 > &s )
      : pointer( p ), stride( s )
	  {}

    /**
     * Sets parameters of the interpolator.
     *
     * @param p pointer to the data to be interpolated
     * @param s reference to the strides of the data to be interpolated
     */
    void SetParams ( ElementType *p, Vector< int32, 2 > &s )
    {
      pointer = p;

      for ( uint32 i = 0; i < 2; i++ ) {
		    stride[i] = s[i];
      }
    }

    /**
     * Pure virtual method for computing the interpolated value according to input coordinates.
     * Need to be implemented in subclasses, according to the method that the subclass wishes to realize.
     *
     * @param coords reference to the coordinates
     * @return the interpolated value
     */
	  virtual ElementType Get ( CoordType &coords ) = 0;

    /// Pointer to the data to be interpolated.   
    ElementType *pointer;
    /// Strides of the data to be interpolated. 
	  Vector< int32, 2 > stride;
};

/**
 * As a subclass of Interpolator2D, this class realizes the nearest neighbor interpolator method.
 */
template< typename ElementType >
class NearestInterpolator2D: public Interpolator2D< ElementType >
{
  public:

    typedef Interpolator2D< ElementType > PredecessorType;

    /**
     * NearestInterpolator2D constructor.
     */
    NearestInterpolator2D () 
      : PredecessorType() 
    {}

    /**
     * NearestInterpolator2D constructor.
     *
     * @param pointer pointer to the data to be interpolated
     * @param stride reference to the strides of the data to be interpolated
     */
    NearestInterpolator2D ( ElementType *pointer, Vector< int32, 2 > &stride )
      : PredecessorType( pointer, stride ) 
	  {}

    /**
     * Method for computing the interpolated value according to input coordinates.
     * Implementation of the base class’ Get method.
     *
     * @param coords reference to the coordinates
     * @return the interpolated value
     */
	  ElementType Get ( CoordType &coords )
	  { 
      return *(pointer + (ROUND(coords[0]) * stride[0] + ROUND(coords[1]) * stride[1]));
    }
};

/**
 * Subclass of Interpolator2D that realizes the bilinear interpolator method.
 */
template< typename ElementType >
class LinearInterpolator2D: public Interpolator2D< ElementType >
{
  public:

    typedef Interpolator2D< ElementType > PredecessorType;

    /**
     * LinearInterpolator2D constructor.
     */
    LinearInterpolator2D () 
      : PredecessorType() 
    {}

    /**
     * LinearInterpolator2D constructor.
     *
     * @param pointer pointer to the data to be interpolated
     * @param stride reference to the strides of the data to be interpolated
     */
	  LinearInterpolator2D ( ElementType *pointer, Vector< int32, 2 > &stride )
      : PredecessorType( pointer, stride ) 
	  {}

    /**
     * Method for computing the interpolated value according to input coordinates.
     * Implementation of the base class’ Get method.
     *
     * @param coords reference to the coordinates
     * @return the interpolated value
     */
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
