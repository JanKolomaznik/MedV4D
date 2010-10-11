#ifndef GL_TEXTURE_IMAGE_H
#define GL_TEXTURE_IMAGE_H

#include <boost/shared_ptr.hpp>
#include <boost/cast.hpp>
#include "common/Common.h"
#include "common/OGLTools.h"
#include "Imaging/AImageRegion.h"
#include "Imaging/Image.h"
#include "GUI/utils/OGLDrawing.h"

namespace M4D
{


template < uint32 Dim >
struct GLTextureImageTyped;

struct GLTextureImage
{
	typedef boost::shared_ptr< GLTextureImage > Ptr;

	GLTextureImage(): _linearInterpolation( false ), _gltextureID( 0 )
	{ }

	virtual
	~GLTextureImage()
	{
		if( _gltextureID != 0 ) {
			glDeleteTextures( 1, &_gltextureID );
		}
	}

	GLuint
	GetTextureGLID() const
	{ return _gltextureID; }

	virtual void
	SetImage( const M4D::Imaging::AImageRegion &image ) = 0;
	
	void
	Reset();

	/*bool
	IsPrepared()const;
	*/

	virtual void
	PrepareTexture() = 0;

	void
	SetLinearinterpolation( bool aLinearInterpolation )
	{ _linearInterpolation = aLinearInterpolation; }


	/**
	 * Check if texture is in sync with image.
	 **/
	/*virtual bool
	IsActual()const = 0;
*/
	virtual bool
	Is2D()const = 0;

	virtual bool
	Is3D()const = 0;


	virtual uint32
	GetDimension()const = 0;

	template< uint32 Dim >
	void
	CheckDimension()
	{
		if ( GetDimension() != Dim ) {
			_THROW_ ErrorHandling::EBadDimension();
		}
	}

	template< uint32 Dim >
	GLTextureImageTyped< Dim > &
	GetDimensionedInterface();
	

	const Vector< float32, 3 > &
	GetMinimum3D()const
	{ return _min3D; }

	const Vector< float32, 3 > &
	GetMaximum3D()const
	{ return _max3D; }

	const Vector< float32, 3 > &
	GetElementExtents3D()const
	{ return _elementExtents3D; }

	const Vector< float32, 2 > &
	GetMinimum2D()const
	{ return _min2D; }

	const Vector< float32, 2 > &
	GetMaximum2D()const
	{ return _max2D; }

	const Vector< float32, 2 > &
	GetElementExtents2D()const
	{ return _elementExtents2D; }

	SIMPLE_GET_SET_METHODS( bool, LinearInterpolation, _linearInterpolation );
protected:
	bool				_linearInterpolation;
	//M4D::Imaging::AImage::Ptr	_image;
	GLuint				_gltextureID;

	Vector< float32, 3 >		_min3D, _max3D, _elementExtents3D;
	Vector< float32, 2 >		_min2D, _max2D, _elementExtents2D;
};

template < uint32 Dim >
struct GLTextureImageTyped: public GLTextureImage
{
	bool
	Is2D()const
	{ return Dim == 2; }

	
	bool
	Is3D()const
	{ return Dim == 3; }

	/*bool
	IsActual()const
	{ return IsPrepared() && (_datasetTimestamp == _image->GetStructureTimestamp()); } //TODO check  edit time stamp contained data
*/
	uint32
	GetDimension()const
	{ return Dim; }

	Vector< float32, Dim >
	GetMinimum()const
	{ return _image->GetRealMinimum(); }

	Vector< float32, Dim > 
	GetMaximum()const
	{ return _image->GetRealMaximum(); }

	Vector< float32, Dim > 
	GetElementExtents()const
	{ return _image->GetElementExtents(); }

	Vector< uint32, Dim > 
	GetSize()const
	{ return _image->GetSize(); }


	void
	SetImage( const M4D::Imaging::AImageRegion &image )
	{ 
		SetImage( static_cast<const M4D::Imaging::AImageRegionDim<Dim> &>( image ) );
	}

	void
	SetImage( const M4D::Imaging::AImageRegionDim<Dim> &image )
	{	
		_image = M4D::Imaging::AImageRegionDim<Dim>::Cast( image.Clone() );

		//Vector< float32, ImageType::Dimension > min( _image->GetRealMinimum() );
		//Vector< float32, ImageType::Dimension > max( _image->GetRealMaximum() );

		if ( Dim == 2 ) {
			_min2D = Vector< float32, 2 >( _image->GetRealMinimum().GetData() );
			_max2D = Vector< float32, 2 >( _image->GetRealMaximum().GetData() );
			_elementExtents2D = Vector< float32, 2 >( _image->GetElementExtents().GetData() );
			_min3D = Vector< float32, 3 >( _min2D, 0.0f );
			_max3D = Vector< float32, 3 >( _max2D, 0.0f );
			_elementExtents3D = Vector< float32, 3 >( _elementExtents2D, 0.0f );
		} else {
			_min3D = Vector< float32, 3 >( _image->GetRealMinimum().GetData() );
			_max3D = Vector< float32, 3 >( _image->GetRealMaximum().GetData() );
			_elementExtents3D = Vector< float32, 3 >( _image->GetElementExtents().GetData() );
			_min2D = Vector< float32, 2 >( _min3D.GetData() );
			_max2D = Vector< float32, 2 >( _max3D.GetData() );
			_elementExtents2D = Vector< float32, 2 >( _elementExtents3D.GetData() );
		}
	}

	void
	PrepareTexture()
	{
		_gltextureID = GLPrepareTextureFromImageData( *_image, _linearInterpolation );
	}

protected:
	typename M4D::Imaging::AImageRegionDim<Dim>::ConstPtr	_image;
};


template< uint32 Dim >
GLTextureImageTyped< Dim > &
GLTextureImage::GetDimensionedInterface()
{
	return *boost::polymorphic_downcast< GLTextureImageTyped< Dim > *>( this );
}






GLTextureImage::Ptr
CreateTextureFromImage( const M4D::Imaging::AImageRegion &image, bool aLinearInterpolation = true );

template < uint32 Dim >
GLTextureImage::Ptr
CreateTextureFromImageTyped( const M4D::Imaging::AImageRegionDim<Dim> &image, bool aLinearInterpolation = true )
{
	typedef GLTextureImageTyped< Dim > TextureImage;
	TextureImage * texture = new TextureImage;
	texture->SetLinearinterpolation( aLinearInterpolation );

	texture->SetImage( image );
	texture->PrepareTexture();

	return GLTextureImage::Ptr( texture );
}







} /*namespace M4D*/

#endif /*GL_TEXTURE_IMAGE_H*/

