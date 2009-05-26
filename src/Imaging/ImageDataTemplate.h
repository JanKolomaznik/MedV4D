/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageDataTemplate.h 
 * @{ 
 **/

#ifndef _IMAGE_DATA_TEMPLATE_H
#define _IMAGE_DATA_TEMPLATE_H

#include "common/Common.h"

#include "Imaging/AbstractImageData.h"
#include "Imaging/ModificationManager.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

//Forward declaration
class ImageFactory;


template < typename ElementType >
class ImageDataTemplate: public AbstractImageData
{
public:
	/**
	 * ImageFactory needs to access private members, because its the
	 * only way to create instance of ImageDataTemplate.
	 **/
	friend class ImageFactory;

	/**
	 * Type of this class - can be used in other templates.
	 **/
	typedef	ImageDataTemplate< ElementType > ThisClass;

	/**
	 * Smart pointer to instance of this class.
	 **/
	typedef typename boost::shared_ptr< ThisClass > Ptr;

	/**
	 * Type of elements contained in this dataset.
	 **/
	typedef ElementType 	Type;

	~ImageDataTemplate();	

	/**
	 * Access method to data array of constant dataset. 
	 *
	 * We don't use idea of multidimensinal data (ignore strides).
	 * @param index Index used to access element in array.
	 * @return Copy of element on given position in array.
	 * @exception EIndexOutOfBounds If given index is outside of array.
	 **/
	ElementType
	Get( size_t index )const;
	/**
	 * Access method to data array. 
	 *
	 * We don't use idea of multidimensinal data (ignore strides).
	 * @param index Index used to access element in array.
	 * @return Reference to element on given position in array.
	 * @exception EIndexOutOfBounds If given index is outside of array.
	 **/
	ElementType&
	Get( size_t index );

	/**
	 * Method used for easy runtime type identification of 
	 * elements types - working only on predefined types.
	 * @return ID of numeric type defined in Common.h
	 **/
	int
	GetElementTypeID()const
		{ return GetNumericTypeID<ElementType>(); }

	/**
	 * Helper function to acces two dimensional data.
	 * Function doesn't check if dataset is two dimensional and 
	 * if given indices are in allowed interval. Only thing, that is 
	 * checked is whether final index to data array is valid. 
	 *
	 * As a result we always access valid memory (or catch exception), but 
	 * we can obtain wrong element, if we didn't check bounds and dimension before.
	 * @param x X coordinate.
 	 * @param y Y coordinate.
	 * @return Copy of element on desired location.
	 * @exception EIndexOutOfBounds If final index is outside of array.
	 **/
	inline ElementType
	Get( size_t x, size_t y )const;
	/**
	 * Helper function to acces two dimensional data.
	 * Function doesn't check if dataset is two dimensional and 
	 * if given indices are in allowed interval. Only thing, that is 
	 * checked is whether final index to data array is valid. 
	 *
	 * As a result we always access valid memory (or catch exception), but 
	 * we can obtain wrong element, if we didn't check bounds and dimension before.
	 * @param x X coordinate.
 	 * @param y Y coordinate.
	 * @return Reference to element on desired location.
	 * @exception EIndexOutOfBounds If final index is outside of array.
	 **/
	inline ElementType&
	Get( size_t x, size_t y );

	/**
	 * Helper function to acces three dimensional data.
	 * Function doesn't check if dataset is three dimensional and 
	 * if given indices are in allowed interval. Only thing, that is 
	 * checked is whether final index to data array is valid. 
	 *
	 * As a result we always access valid memory (or catch exception), but 
	 * we can obtain wrong element, if we didn't check bounds and dimension before.
	 * @param x X coordinate.
 	 * @param y Y coordinate.
 	 * @param z Z coordinate.
	 * @return Copy of element on desired location.
	 * @exception EIndexOutOfBounds If final index is outside of array.
	 **/
	inline ElementType
	Get( size_t x, size_t y, size_t z )const;
	/**
	 * Helper function to acces three dimensional data.
	 * Function doesn't check if dataset is three dimensional and 
	 * if given indices are in allowed interval. Only thing, that is 
	 * checked is whether final index to data array is valid. 
	 *
	 * As a result we always access valid memory (or catch exception), but 
	 * we can obtain wrong element, if we didn't check bounds and dimension before.
	 * @param x X coordinate.
 	 * @param y Y coordinate.
 	 * @param z Z coordinate.
	 * @return Reference to element on desired location.
	 * @exception EIndexOutOfBounds If final index is outside of array.
	 **/
	inline ElementType&
	Get( size_t x, size_t y, size_t z );

	/**
	 * Helper function to acces four dimensional data.
	 * Function doesn't check if dataset is four dimensional and 
	 * if given indices are in allowed interval. Only thing, that is 
	 * checked is whether final index to data array is valid. 
	 *
	 * As a result we always access valid memory (or catch exception), but 
	 * we can obtain wrong element, if we didn't check bounds and dimension before.
	 * @param x X coordinate.
 	 * @param y Y coordinate.
 	 * @param z Z coordinate.
 	 * @param t T coordinate.
	 * @return Copy of element on desired location.
	 * @exception EIndexOutOfBounds If final index is outside of array.
	 **/
	inline ElementType
	Get( size_t x, size_t y, size_t z, size_t t )const;
	/**
	 * Helper function to acces four dimensional data.
	 * Function doesn't check if dataset is four dimensional and 
	 * if given indices are in allowed interval. Only thing, that is 
	 * checked is whether final index to data array is valid. 
	 *
	 * As a result we always access valid memory (or catch exception), but 
	 * we can obtain wrong element, if we didn't check bounds and dimension before.
	 * @param x X coordinate.
 	 * @param y Y coordinate.
 	 * @param z Z coordinate.
 	 * @param t T coordinate.
	 * @return Reference to element on desired location.
	 * @exception EIndexOutOfBounds If final index is outside of array.
	 **/
	inline ElementType&
	Get( size_t x, size_t y, size_t z, size_t t );
	/**
	 * Another way to access directly data array.
	 * Same behavior as method Get.
	 * @param index Index used to access element in array.
	 * @return Copy of element on given position in array.
	 * @exception EIndexOutOfBounds If given index is outside of array.
	 **/
	ElementType
	operator[]( size_t index )const
				{ return Get( index ); }
	/**
	 * Another way to access directly data array.
	 * Same behavior as method Get.
	 * @param index Index used to access element in array.
	 * @return Reference to element on given position in array.
	 * @exception EIndexOutOfBounds If given index is outside of array.
	 **/
	ElementType&
	operator[]( size_t index )
				{ return Get( index ); }

	static Ptr
	CastAbstractPointer(  AbstractImageData::APtr aptr );

	ModificationManager &
	GetModificationManager()const
		{ return _modificationManager; }
protected:
	/**
	 * Protected constructor - used by ImageFactory.
	 **/
	ImageDataTemplate( 
			ElementType 		*data, 
			DimensionInfo		*parameters,
			unsigned short		dimension,
			size_t			elementCount
			);	

	ImageDataTemplate( 
			AlignedArrayPointer< ElementType >	data, 
			DimensionInfo				*parameters,
			unsigned short				dimension,
			size_t					elementCount
			);	

	mutable ModificationManager	_modificationManager;
private:
	ImageDataTemplate();
	ImageDataTemplate( const ImageDataTemplate &);
	ImageDataTemplate &operator=( const ImageDataTemplate &);


	ElementType		*_data;
	AlignedArrayPointer< ElementType > _arrayPointer;

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



} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

/*Include template implementation.*/
#include "Imaging/ImageDataTemplate.tcc"


#endif /*_IMAGE_DATA_TEMPLATE_H*/

/** @} */

