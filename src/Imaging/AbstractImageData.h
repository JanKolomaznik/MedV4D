#ifndef _ABSTRACT_IMAGE_DATA_H
#define _ABSTRACT_IMAGE_DATA_H

#include "Common.h"
#include "ExceptionBase.h"

#include <boost/shared_ptr.hpp>

namespace M4D
{

namespace Imaging
{
	
/**
 * Structure containing information about image in one dimension. Each 
 * dimension is supposed to have its own informations structure.
 **/
struct DimensionInfo
{
	/**
	 * Default constructor - initialize all members with default values.
	 **/
	DimensionInfo()
		:size(0), stride(0), elementExtent(1.0) {}

	/**
	 * Constructor used to create DimensionInfo with required member values.
	 * @param asize Number of elements in corresponding dimension.
	 * @param astride Number of elements needed to skip when increasing index 
	 * in corresponding dimension.
	 * @param aelementExtent Physical size of element in corresponding dimension.
	 **/
	DimensionInfo( uint32 asize, int32 astride, float32 aelementExtent = 1.0f )
		:size(asize), stride(astride), elementExtent(aelementExtent) {}
	/**
	 * Method for setting atributes.
	 * @param asize Number of elements in corresponding dimension.
	 * @param astride Number of elements needed to skip when increasing index 
	 * in corresponding dimension.
	 * @param aelementExtent Physical size of element in corresponding dimension.
	 **/
	void Set( uint32 asize, int32 astride, float32 aelementExtent = 1.0f )
		{ size = asize; stride = astride; elementExtent = aelementExtent; }

	/**
	 * Width of image in actual dimension.
	 **/
	uint32		size;
	/**
	 * Stride, which is used to increase coordinates in actual dimension.
	 **/
	int32		stride;
	/**
	 * Physical size of element in this dimension.
	 **/
	float32		elementExtent;
};

class AbstractImageData
{
public:
	/**
	 * Smart pointer type for accesing AbstractImageData instance (child).
	 **/
	typedef boost::shared_ptr< AbstractImageData > APtr;

	/**
	 * Constructor used for initialization by successors.
	 * @param parameters Pointer to array with informations about each dimension.
	 * @param dimension Number of dimensions - length of parameters array.
	 * @param elementCount Number of elements contained in image.
	 **/
	AbstractImageData( 
			DimensionInfo		*parameters,
			unsigned short		dimension,
			size_t			elementCount
			);

	virtual ~AbstractImageData()=0;

	/**
	 * @return ID of element type.
	 **/
	virtual int
	GetElementTypeID()const=0;

	/**
	 * @return Number of elements contained in image.
	 **/
	size_t
	GetSize() const
		{ return _elementCount; }

	/**
	 * @return Dimensionality of image.
	 **/
	size_t
	GetDimension()const 
		{ return _dimension; }

	/**
	 * Return info about desired dimension.
	 * @param dim Index of required dimension.
	 * @return Constant reference to informations about dimension with 
	 * passed index.
	 **/
	const DimensionInfo&
	GetDimensionInfo( unsigned short dim )const;
protected:
	/**
	 * Count of elements in image.
	 **/
	size_t			_elementCount;
	/**
	 * Dimensionality of image.
	 **/
	unsigned short		_dimension;
	/**
	 * Array of '_dimension' length with informations about each dimension.
	 **/
	DimensionInfo		*_parameters;
private:
	/**
	 * Not implemented - usage prohibited.
	 **/
	AbstractImageData();
	/**
	 * Not implemented - usage prohibited.
	 **/
	AbstractImageData( const AbstractImageData &);
	/**
	 * Not implemented - usage prohibited.
	 **/
	AbstractImageData &operator=( const AbstractImageData &);

public:
	class EWrongDimension: public ErrorHandling::ExceptionBase
	{
	public:
		/**
		 * @param wrong Wrong dimension number, which raised 
		 * this exception.
		 * @param actual Number of dimensions image, which raised 
		 * this exception.
		 **/
		EWrongDimension( unsigned short wrong, unsigned short actual )
			: ErrorHandling::ExceptionBase( "Accesing image data in wrong dimension." ), 
			_wrong( wrong ), _actual( actual ) {}
		
		/**
		 * @return Dimension index, which raised this exception.
		 **/
		unsigned short 
		GetWrong()const { return _wrong; }

		/**
		 * @return Dimension of image, which raised this exception.
		 **/
		unsigned short 
		GetActual()const { return _actual; }
	protected:
		/**
		 * Wrong dimension number, which caused throwing of this exception.
		 **/
		unsigned short	_wrong;	
		/**
		 * Real dimensionality of image.
		 **/
		unsigned short	_actual;	
	};
};

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_ABSTRACT_IMAGE_DATA_H*/
