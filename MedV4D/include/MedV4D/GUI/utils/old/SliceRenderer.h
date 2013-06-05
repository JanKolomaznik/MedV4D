#ifndef SLICE_RENDERER_H
#define SLICE_RENDERER_H


#include "MedV4D/GUI/utils/GLTextureImage.h"
#include "MedV4D/GUI/utils/ARenderer.h"
#include <soglu/GLTextureImage.hpp>

namespace M4D
{

class SliceRenderer: public ARenderer
{
public:
	void
	SetData( soglu::GLTextureImage::Ptr data );

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

