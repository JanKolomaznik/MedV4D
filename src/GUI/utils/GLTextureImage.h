#ifndef GL_TEXTURE_IMAGE_H
#define GL_TEXTURE_IMAGE_H

#include <boost/shared_ptr.hpp>
#include "common/Common.h"
namespace M4D
{

struct GLTextureImage
{
	typedef boost::shared_ptr< GLTextureImage > Ptr;

	GLuint
	GetTextureGLID();

	virtual void
	SetImage( M4D::Imaging::AImage::Ptr image ) = 0;
	
	void
	Reset();

	bool
	IsPrepared()const;

	virtual bool
	Is2D()const = 0;

	virtual bool
	Is3D()const = 0;

	SIMPLE_GET_SET_METHODS( bool, LinearInterpolation, _linearInterpolation );
private:
	bool				_linearInterpolation;
	//M4D::Imaging::AImage::Ptr	_image;
	GLuint				_gltextureID;
};

template < typename ImageType >
struct GLTextureImageTyped: public GLTextureImage
{
	bool
	Is2D()const
	{ return ImageType::Dimension == 2; }

	
	bool
	Is3D()const
	{ return ImageType::Dimension == 3; }

	void
	SetImage( M4D::Imaging::AImage::Ptr image ) = 0;

	void
	SetImage( ImageType::Ptr image );

private:
	ImageType::Ptr	_image;
};

} /*namespace M4D*/

#endif /*GL_TEXTURE_IMAGE_H*/

