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

	int
	GetElementTypeID()
		{ return GetNumericTypeID<ElementType>(); }

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


protected:
	ImageDataTemplate( 
			ElementType 		*data, 
			DimensionInfo		*parameters,
			unsigned short		dimension,
			size_t			elementCount
			);	
private:
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

};



} /*namespace Images*/
} /*namespace M4D*/

/*Include template implementation.*/
#include "ImageDataTemplate.tcc"


#endif /*_IMAGE_DATA_TEMPLATE_H*/
