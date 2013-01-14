#ifndef GL_TEXTURE_IMAGE_H
#define GL_TEXTURE_IMAGE_H

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/cast.hpp>
#include "MedV4D/Common/Common.h"
#include "MedV4D/GUI/utils/OGLTools.h"
#include "MedV4D/Imaging/AImageRegion.h"
#include "MedV4D/Imaging/Image.h"
#include "MedV4D/GUI/utils/OGLDrawing.h"

namespace M4D
{


template < size_t Dim >
struct GLTextureImageTyped;

struct GLTextureImage
{
	typedef boost::shared_ptr< GLTextureImage > Ptr;
	typedef boost::weak_ptr< GLTextureImage > WPtr;

	GLTextureImage(): _linearInterpolation( false ), _gltextureID( 0 )
	{ }

	virtual
	~GLTextureImage()
	{
		DeleteTexture();
	}

	GLuint
	GetTextureGLID() const
	{ return _gltextureID; }


	/*virtual void
	SetImage( const M4D::Imaging::AImageRegion &image ) = 0;*/
	
	void
	Reset();

	/*bool
	IsPrepared()const;
	*/

	/*virtual void
	PrepareTexture() = 0;*/

	void
	DeleteTexture()
	{
		if( _gltextureID != 0 ) {
			D_PRINT( "Deleting texture id = " << _gltextureID );
			ASSERT( isGLContextActive() );
			GL_CHECKED_CALL( glDeleteTextures( 1, &_gltextureID ) );
			_gltextureID = 0;
		}
	}

	/*void
	SetLinearinterpolation( bool aLinearInterpolation )
	{ _linearInterpolation = aLinearInterpolation; }*/


	/**
	 * Check if texture is in sync with image.
	 **/
	/*virtual bool
	IsActual()const = 0;
*/

	virtual bool
	Is1D()const = 0;

	virtual bool
	Is2D()const = 0;

	virtual bool
	Is3D()const = 0;


	virtual uint32
	GetDimension()const = 0;

	template< size_t Dim >
	void
	CheckDimension()
	{
		if ( GetDimension() != Dim ) {
			_THROW_ ErrorHandling::EBadDimension();
		}
	}

	template< size_t Dim >
	GLTextureImageTyped< Dim > &
	GetDimensionedInterface();
	
	SIMPLE_GET_SET_METHODS( bool, LinearInterpolation, _linearInterpolation );
protected:
	GLTextureImage( GLuint aTexID, bool aLinearInterpolation ): _linearInterpolation( aLinearInterpolation ), _gltextureID( aTexID )
	{ ASSERT( aTexID ); }
	
	bool				_linearInterpolation;
	//M4D::Imaging::AImage::Ptr	_image;
	GLuint				_gltextureID;
};

template < size_t Dim >
struct GLTextureImageTyped: public GLTextureImage
{
	typedef boost::shared_ptr< GLTextureImageTyped > Ptr;
	typedef boost::weak_ptr< GLTextureImageTyped > WPtr;
	
	GLTextureImageTyped( GLuint aTexID, bool aLinearInterpolation, M4D::Imaging::ImageExtentsRecord< Dim > aExtents )
	: GLTextureImage( aTexID, aLinearInterpolation ), mExtents( aExtents )
	{ }
	
	bool
	Is1D()const
	{ return Dim == 1; }

	bool
	Is2D()const
	{ return Dim == 2; }

	
	bool
	Is3D()const
	{ return Dim == 3; }

	Vector2f
	GetMappedInterval()const
	{ return Vector2f( 0, 65535 );/*mMappedInterval;*/ }

	/*bool
	IsActual()const
	{ return IsPrepared() && (_datasetTimestamp == _image->GetStructureTimestamp()); } //TODO check  edit time stamp contained data
*/
	uint32
	GetDimension()const
	{ return Dim; }
	
	const M4D::Imaging::ImageExtentsRecord< Dim > &
	getExtents()const
	{ return mExtents; }

	/*Vector< float32, Dim >
	GetMinimum()const
	{ return _image->GetRealMinimum(); }

	Vector< float32, Dim > 
	GetMaximum()const
	{ return _image->GetRealMaximum(); }

	Vector< float32, Dim > 
	GetRealSize()const
	{ return _image->GetRealSize(); }

	Vector< float32, Dim > 
	GetElementExtents()const
	{ return mExtents.GetElementExtents(); }*/

	Vector< float32, Dim >
	GetElementExtentsInTextureSpace()const
	{
		return VectorMemberDivision( mExtents.elementExtents, (mExtents.maximum-mExtents.minimum) );
	}

	void
	updateTexture( GLuint aTexID, M4D::Imaging::ImageExtentsRecord< Dim > aExtents )
	{
		DeleteTexture(); //TODO check texture
		_gltextureID = aTexID;
		mExtents = aExtents;
	}
	/*Vector< uint32, Dim > 
	GetSize()const
	{ return _image->GetSize(); }*/


	/*void
	SetImage( const M4D::Imaging::AImageRegion &image )
	{ 
		SetImage( static_cast<const M4D::Imaging::AImageRegionDim<Dim> &>( image ) );
	}

	void
	SetImage( const M4D::Imaging::AImageRegionDim<Dim> &image )
	{	
		_image = M4D::Imaging::AImageRegionDim<Dim>::Cast( image.Clone() );
	}

	void
	PrepareTexture()
	{
		ASSERT( IsGLContextActive() );

		DeleteTexture();
		_gltextureID = GLPrepareTextureFromImageData( *_image, _linearInterpolation );
		D_PRINT( "Created texture id = " << _gltextureID );
	}*/

protected:
	//typename M4D::Imaging::AImageRegionDim<Dim>::ConstPtr	_image;
	M4D::Imaging::ImageExtentsRecord< Dim > mExtents;
};


template< size_t Dim >
GLTextureImageTyped< Dim > &
GLTextureImage::GetDimensionedInterface()
{
	return *boost::polymorphic_downcast< GLTextureImageTyped< Dim > *>( this ); //TODO
}

template< size_t Dim >
typename GLTextureImageTyped< Dim >::WPtr
GLTextureGetDimensionedInterfaceWPtr( GLTextureImage::WPtr aTexture )
{
	GLTextureImage::Ptr tex = aTexture.lock();
	//if( tex ) {
	return boost::dynamic_pointer_cast< GLTextureImageTyped< Dim > >( tex );
	//}
}



template < size_t Dim >
void
updateTextureSubImageTyped( GLTextureImageTyped< Dim > &aTexImage, const M4D::Imaging::AImageRegionDim<Dim> &aImage, Vector< int, Dim > aMinimum, Vector< int, Dim > aMaximum )
{
	//TODO some checks
	M4D::GLUpdateTextureFromSubImageData( aTexImage.GetTextureGLID(), aImage, aMinimum, aMaximum );
}

template < size_t Dim >
void
updateTextureSubImage( GLTextureImage &aTexImage, const M4D::Imaging::AImageRegion &aSubImage, Vector< int, Dim > aMinimum, Vector< int, Dim > aMaximum )
{
	if ( aTexImage.GetDimension() != aSubImage.GetDimension() || Dim != aTexImage.GetDimension() ) {
		_THROW_ M4D::ErrorHandling::EBadParameter( "Texture and subimage have different dimension" );
	}
	
	updateTextureSubImageTyped< Dim >( aTexImage.GetDimensionedInterface<Dim>(), static_cast< const M4D::Imaging::AImageRegionDim<Dim> & >( aSubImage ), aMinimum, aMaximum );
}


GLTextureImage::Ptr
createTextureFromImage( const M4D::Imaging::AImageRegion &image, bool aLinearInterpolation = true );

template < size_t Dim >
GLTextureImage::Ptr
createTextureFromImageTyped( const M4D::Imaging::AImageRegionDim<Dim> &image, bool aLinearInterpolation = true )
{
	typedef GLTextureImageTyped< Dim > TextureImage;
	M4D::Imaging::ImageExtentsRecord< Dim > extents = image.GetImageExtentsRecord();
	GLuint textureID = GLPrepareTextureFromImageData( image, aLinearInterpolation ); //TODO prevent loosing texture during exception

	return typename TextureImage::Ptr( new TextureImage( textureID, aLinearInterpolation, extents ) );
}

void
recreateTextureFromImage( GLTextureImage &aTexImage, const M4D::Imaging::AImageRegion &image );

template < size_t Dim >
void
recreateTextureFromImageTyped( GLTextureImageTyped< Dim > &aTexImage, const M4D::Imaging::AImageRegionDim<Dim> &image )
{
	//TODO test image properties if same
	aTexImage.DeleteTexture();
	
	GLuint textureID = GLPrepareTextureFromImageData( image, aTexImage.GetLinearInterpolation() ); //TODO prevent loosing texture during exception
	aTexImage.updateTexture( textureID, image.GetImageExtentsRecord() );
	/*aTexImage.SetImage( image );
	aTexImage.PrepareTexture();*/
}

typedef GLTextureImageTyped< 2 > GLTextureImage2D;
typedef GLTextureImageTyped< 3 > GLTextureImage3D;





} /*namespace M4D*/

#endif /*GL_TEXTURE_IMAGE_H*/

