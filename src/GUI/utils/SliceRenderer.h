#ifndef SLICE_RENDERER_H
#define SLICE_RENDERER_H


#include "GUI/utils/GLTextureImage.h"

class SliceRenderer: public ARenderer
{
public:
	void
	SetData( GLTextureImage::Ptr data );

	bool
	IsRenderingVolume()const;

	void
	Render();
protected:

};

#endif /*SLICE_RENDERER_H*/

