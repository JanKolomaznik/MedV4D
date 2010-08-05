#ifndef GL_TEXTURE_IMAGE_H
#define GL_TEXTURE_IMAGE_H

#include <boost/shared_ptr.hpp>
#include "common/Common.h"
#include "common/OGLTools.h"
#include "Imaging/AImageRegion.h"
#include "Imaging/Image.h"
#include "GUI/utils/OGLDrawing.h"

namespace M4D
{

struct GLTextureImage
{
	typedef boost::shared_ptr< GLTextureImage > Ptr;

	GLTextureImage(): _gltextureID( 0 )
	{ }

	virtual
	~GLTextureImage()
	{
		if( _gltextureID != 0 ) {
			glDeleteTextures( 1, &_gltextureID );
		}
	}

	GLuint
	GetTextureGLID()
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

	const Vector< float32, 3 > &
	GetMinimum3D()const
	{ return _min3D; }

	const Vector< float32, 3 > &
	GetMaximum3D()const
	{ return _max3D; }

	const Vector< float32, 2 > &
	GetMinimum2D()const
	{ return _min2D; }

	const Vector< float32, 2 > &
	GetMaximum2D()const
	{ return _max2D; }

	SIMPLE_GET_SET_METHODS( bool, LinearInterpolation, _linearInterpolation );
protected:
	bool				_linearInterpolation;
	//M4D::Imaging::AImage::Ptr	_image;
	GLuint				_gltextureID;

	Vector< float32, 3 >		_min3D, _max3D;
	Vector< float32, 2 >		_min2D, _max2D;
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
			_min3D = Vector< float32, 3 >( _min2D, 0.0f );
			_max3D = Vector< float32, 3 >( _max2D, 0.0f );
		} else {
			_min3D = Vector< float32, 3 >( _image->GetRealMinimum().GetData() );
			_max3D = Vector< float32, 3 >( _image->GetRealMaximum().GetData() );
			_min2D = Vector< float32, 2 >( _min3D.GetData() );
			_max2D = Vector< float32, 2 >( _max3D.GetData() );
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

GLTextureImage::Ptr
CreateTextureFromImage( const M4D::Imaging::AImageRegion &image );

template < uint32 Dim >
GLTextureImage::Ptr
CreateTextureFromImageTyped( const M4D::Imaging::AImageRegionDim<Dim> &image )
{
	typedef GLTextureImageTyped< Dim > TextureImage;
	TextureImage * texture = new TextureImage;

	texture->SetImage( image );
	texture->PrepareTexture();

	return GLTextureImage::Ptr( texture );
}

} /*namespace M4D*/

#endif /*GL_TEXTURE_IMAGE_H*/

