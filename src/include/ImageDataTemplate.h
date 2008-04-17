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


template < typename ElementType >
class ImageDataTemplate: public AbstractImage
{
public:
	friend class ImageFactory;
	/**
	 * Smart pointer to instance of this class.
	 **/
	typedef typename boost::shared_ptr< ImageDataTemplate< ElementType > > Ptr;

	typedef ElementType 	Type;

	~ImageDataTemplate();	

	ElementType
	Get( size_t index )const;
	ElementType&
	Get( size_t index );

	/**
	 * Access methods. 
	 * AREN'T CHECKING BOUNDS!!!
	 **/
	ElementType
	Get( size_t x, size_t y )const;
	ElementType&
	Get( size_t x, size_t y );

	inline ElementType
	Get( size_t x, size_t y, size_t z )const;
	inline ElementType&
	Get( size_t x, size_t y, size_t z );
	
	ElementType
	operator[]( size_t index )const
				{ return Get( index ); }
	ElementType&
	operator[]( size_t index )
				{ return Get( index ); }

	size_t
	GetDimension()const 
				{ return _dimension; }

	const DimensionInfo&
	GetDimensionInfo( unsigned short dim )const;

protected:
	ImageDataTemplate( 
			ElementType 		*data, 
			DimensionInfo		*parameters,
			unsigned short		dimension,
			size_t			elementCount
			);	
private:
	size_t			_elementCount;
	unsigned short		_dimension;
	DimensionInfo		*_parameters;
	ElementType		*_data;

public:
	class EIndexOutOfBounds: public ErrorHandling::ExceptionBase
	{
	public:
		EIndexOutOfBounds( size_t wrongIndex )
			: ErrorHandling::ExceptionBase( "Wrong index to image element." ), 
			_wrongIndex( wrongIndex ) {}

		/**
		 * @return Wrong index, which raised this exception.
		 **/
		size_t 
		GetIndex()const { return _wrongIndex; }
	protected:
		size_t		_wrongIndex;
	};

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



} /*namespace Images*/
} /*namespace M4D*/

/*Include template implementation.*/
#include "ImageDataTemplate.tcc"


#endif /*_IMAGE_DATA_TEMPLATE_H*/
