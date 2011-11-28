#ifndef SLICE_RENDERER_H
#define SLICE_RENDERER_H


#include "MedV4D/GUI/utils/GLTextureImage.h"
#include "MedV4D/GUI/utils/ARenderer.h"

namespace M4D
{

class SliceRenderer: public ARenderer
{
public:
	void
	SetData( GLTextureImage::Ptr data );

	bool
	IsRenderingVolume()const;

	virtual void
	Initialize() = 0;

	virtual void
	Finalize() = 0;

	void
	Render();
protected:

};

} /*namespace M4D*/

#endif /*SLICE_RENDERER_H*/

