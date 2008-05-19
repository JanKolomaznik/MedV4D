#ifndef _ABSTRACT_IMAGE_H
#define _ABSTRACT_IMAGE_H

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
	 * Method for setting atributes.
	 * @param asize Value used for size setting.
	 * @param astride Value used for stride setting.
	 **/
	void Set( size_t asize, uint32 astride )
		{ size = asize; stride = astride; }

	/**
	 * Width of image in actual dimension.
	 **/
	size_t		size;
	/**
	 * Stride, which is used to increase coordinates in actual dimension.
	 **/
	uint32		stride;
};

class AbstractImageData
{
public:
	/**
	 * Smart pointer type for accesing AbstractImageData instance (child).
	 **/
	typedef boost::shared_ptr< AbstractImageData > APtr;

	AbstractImageData( 
			DimensionInfo		*parameters,
			unsigned short		dimension,
			size_t			elementCount
			);

	virtual ~AbstractImageData()=0;

	virtual int
	GetElementTypeID()=0;

	size_t
	GetSize() const
		{ return _elementCount; }
	size_t
	GetDimension()const 
				{ return _dimension; }

	const DimensionInfo&
	GetDimensionInfo( unsigned short dim )const;
protected:
	size_t			_elementCount;
	unsigned short		_dimension;
	DimensionInfo		*_parameters;
private:
	AbstractImageData();
	AbstractImageData( const AbstractImageData &);
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
		unsigned short	_wrong;	
		unsigned short	_actual;	
	};
};


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_ABSTRACT_IMAGE_H*/
