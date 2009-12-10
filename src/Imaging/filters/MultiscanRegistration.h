/**
 * @ingroup imaging 
 * @author Attila Ulman
 * @file MultiscanRegistration.h 
 * @{ 
 **/

#ifndef MULTISCAN_REGISTRATION_H
#define MULTISCAN_REGISTRATION_H

#include "common/Common.h"
#include "common/Vector.h"

#include "Imaging/AImageFilterWholeAtOnce.h"


/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D {
namespace Imaging {

enum InterpolationType {
	IT_NEAREST,
	IT_LINEAR
};

typedef Vector< float32, 2 > CoordType;

template< typename ElementType >
struct SliceInfo;

struct TransformationInfo2D;

template< typename ImageType >
class MultiscanRegistration;

template< typename ElementType >
class MultiscanRegistration< Image< ElementType, 3 > >
	: public AImageFilterWholeAtOnceIExtents< Image< ElementType, 3 >, Image< ElementType, 3 > >
{
  public:	

	  typedef Image< ElementType, 3 > InputImageType;
	  typedef Image< ElementType, 3 > OutputImageType;
	  typedef AImageFilterWholeAtOnceIExtents< InputImageType, OutputImageType > PredecessorType;

	  struct Properties: public PredecessorType::Properties
	  {
		  Properties ()
        : examinedSliceNum( 2 ), interpolationType( IT_NEAREST ) 
      {}

		  uint32 examinedSliceNum;
      InterpolationType	interpolationType;
	  };

	  MultiscanRegistration ( Properties * prop );
	  MultiscanRegistration ();

	  GET_SET_PROPERTY_METHOD_MACRO(uint32, ExaminedSliceNum, examinedSliceNum);
    GET_SET_PROPERTY_METHOD_MACRO(InterpolationType, InterpolationType, interpolationType);
  
  protected:

	  bool ProcessImage ( const InputImageType &in, OutputImageType &out );

    template< typename InterpolationType >
	  bool ProcessImageHelper ( const InputImageType &in, OutputImageType &out );

    template< typename InterpolationType >
	  bool TransformSlice ( SliceInfo< ElementType > &inSlice, SliceInfo< ElementType > &outSlice,
                          TransformationInfo2D &transInfo );

  private:

	  GET_PROPERTIES_DEFINITION_MACRO;
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
  	for ( uint32 i = 0; i < 2; i++ )
		  {
			  rotation[i]    = 0.0f;
			  translation[i] = 0.0f;
        sampling[i]    = 1.0f;
			}
  }

	TransformationInfo2D ( CoordType &rotation, CoordType &translation, CoordType &sampling )
    : rotation( rotation ), translation( translation ), sampling( sampling )
	{}

	CoordType rotation, translation, sampling;
};

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

template< typename ElementType >
class Interpolator2D
{
  public:

    Interpolator2D ()
      : pointer( NULL )
    {}

	  Interpolator2D ( ElementType *pointer, Vector< int32, 2 > &stride )
      : pointer( pointer ), stride( stride )
	  {}

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

/** @} */


//include implementation
#include "Imaging/filters/MultiscanRegistration.tcc"

#endif // MULTISCAN_REGISTRATION_H

/** @} */
