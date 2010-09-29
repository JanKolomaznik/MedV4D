#ifndef IMAGE_DATA_RENDERER_H
#define IMAGE_DATA_RENDERER_H

namespace M4D
{
namespace GUI
{


typedef uint32 RenderingMode;

class ImageDataRenderer
{
public:
	void
	Initialize();

	void
	Finalize();

	void
	SetImageData( GLTextureImage::Ptr aData );

	void
	SetMaskData( GLTextureImage::Ptr aData );

	void
	SetTransferFunction( GLTextureImage::Ptr aData );

	void
	SetMaskColorMap( GLTextureImage::Ptr aData );

	void
	SetViewConfiguration( ... );

	void
	SetRenderingMode( RenderingMode aMode );

	void
	Render();

protected:

private:

};

} /*namespace M4D*/
} /*namespace GUI*/

#endif /*IMAGE_DATA_RENDERER_H*/