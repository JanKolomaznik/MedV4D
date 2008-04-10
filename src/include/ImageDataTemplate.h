#ifndef _IMAGE_DATA_TEMPLATE_H
#define _IMAGE_DATA_TEMPLATE_H

#include "ExceptionBase.h"

#include "AbstractImage.h"

namespace M4D
{

namespace Images
{

//Forward declaration
class ImageFactory;


/**
 * Structure containing information about image in one dimension. Each 
 * dimension is supposed to have its own informations structure.
 **/
struct DimensionInformations
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


template < typename ElementType >
class ImageDataTemplate: public AbstractImage
{
public:
	friend class ImageFactory;
	typedef typename boost::shared_ptr< ImageDataTemplate< ElementType > > Ptr;

	~ImageDataTemplate();	

	ElementType
	Get( size_t index )const;

	ElementType&
	Get( size_t index );
	
	ElementType
	operator[]( size_t index )const
				{ return Get( index ); }
	ElementType&
	operator[]( size_t index )
				{ return Get( index ); }

	size_t
	GetDimension()const 
				{ return _dimension; }
protected:
	ImageDataTemplate( 
			ElementType 		*data, 
			DimensionInformations	*parameters,
			unsigned short		dimension,
			size_t			elementCount
			);	
private:
	size_t			_elementCount;
	unsigned short		_dimension;
	DimensionInformations	*_parameters;
	ElementType		*_data;

public:
	class EIndexOutOfBounds: public ErrorHandling::ExceptionBase
	{
	public:
		EIndexOutOfBounds( size_t wrongIndex )
			: ErrorHandling::ExceptionBase( "Wrong index to image element." ), 
			_wrongIndex( wrongIndex ) {}

		size_t GetIndex()const { return _wrongIndex; }
	protected:
		size_t		_wrongIndex;
	};
};



} /*namespace Images*/
} /*namespace M4D*/

/*Include template implementation.*/
#include "ImageDataTemplate.tcc"


#endif /*_IMAGE_DATA_TEMPLATE_H*/
