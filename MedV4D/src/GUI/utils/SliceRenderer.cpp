#include "MedV4D/GUI/utils/SliceRenderer.h"

namespace M4D
{
	
void
SliceRenderer::SetData( GLTextureImage::Ptr data )
{

}

bool
SliceRenderer::IsRenderingVolume()const
{
	return false;
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
	/*switch ( _data->GetDimension() )
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
	}*/
}

} /*namespace M4D*/

