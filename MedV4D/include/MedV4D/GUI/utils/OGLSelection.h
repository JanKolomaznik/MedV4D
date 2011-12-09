#ifndef OGL_SELECTION_H
#define OGL_SELECTION_H

#include "MedV4D/GUI/utils/OGLTools.h"
#include "MedV4D/GUI/utils/CgShaderTools.h"
#include "MedV4D/GUI/utils/FrameBufferObject.h"

namespace M4D
{

extern boost::filesystem::path gPickingShaderPath;
	
typedef size_t SelectID;

template < bool tIsInSelectionMode >
struct CurentObjectID
{
	void
	set( SelectID aId )
	{

	}
};

class PickManager
{
public:	
	void
	initialize( unsigned aPickingRadius );
	
	void
	finalize();
	
	template< typename TFunctor >
	void
	render( Vector2i aScreenCoordinates, TFunctor aFunctor );
protected:
	unsigned mPickingRadius;
	FrameBufferObject mFrameBuffer;
	M4D::GUI::CgEffect mCgEffect;
};


template< typename TFunctor >
void
PickManager::render( Vector2i aScreenCoordinates, TFunctor aFunctor )
{
	//mFrameBuffer.Bind();
	M4D::GLPushAtribs pushAttribs;
	glViewport(50, 50, 500, 500);
	
	mCgEffect.ExecuteTechniquePass( "PickingEffect", aFunctor );
	
	//mFrameBuffer.Unbind();
}


}//namespace M4D

#endif /*OGL_SELECTION_H*/
