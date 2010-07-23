#include "GUI/utils/SliceRenderer.h"


void
SliceRenderer::SetData( GLTextureImage::Ptr data )
{

}

bool
SliceRenderer::IsRenderingVolume()const
{

}

void
SliceRenderer::Initialize()
{

}

void
SliceRenderer::Finalize()
{

}

void
SliceRenderer::Render()
{
	switch ( _data->GetDimension() )
	{
	case 2:
		GLDraw2DImage(
			_data->GetMinimum2D(), 
			_data->GetMaximum2D()
			);
		break;
	case 3:
		GLDrawVolumeSlice(
			_data->GetMinimum3D(), 
			_data->GetMaximum3D(),
			_sliceCoord,
			_plane
			);
		break;
	default:
		ASSERT( false ); //TODO exception
	}
}

