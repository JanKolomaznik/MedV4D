#ifndef OGL_SELECTION_H
#define OGL_SELECTION_H

#include "MedV4D/GUI/utils/OGLTools.h"
#include "MedV4D/GUI/utils/CgShaderTools.h"
#include "MedV4D/GUI/utils/FrameBufferObject.h"
#include <glm/glm.hpp>
#include <glm/ext.hpp>

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
	render( Vector2i aScreenCoordinates, const GLViewSetup &aViewSetup, TFunctor aFunctor );
protected:
	unsigned mPickingRadius;
	FrameBufferObject mFrameBuffer;
	M4D::GUI::CgEffect mCgEffect;
};


template< typename TFunctor >
void
PickManager::render( Vector2i aScreenCoordinates, const GLViewSetup &aViewSetup, TFunctor aFunctor )
{
	mFrameBuffer.Bind();
	M4D::GLPushAtribs pushAttribs;
	glm::dmat4x4 pick = glm::pickMatrix(
			glm::dvec2(aScreenCoordinates[0],aScreenCoordinates[1]),
			glm::dvec2(mPickingRadius,mPickingRadius),
			aViewSetup.viewport
		);
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadMatrixd( glm::value_ptr(pick*aViewSetup.projection) );
	//glViewport(aScreenCoordinates[0]-mPickingRadius, aScreenCoordinates[1]-mPickingRadius, aScreenCoordinates[0]+mPickingRadius, aScreenCoordinates[1]+mPickingRadius);
	
	//mCgEffect.ExecuteTechniquePass( "PickingEffect", aFunctor );
	
	glPopMatrix();
	mFrameBuffer.Unbind();
	
	glViewport(aScreenCoordinates[0]-mPickingRadius, aScreenCoordinates[1]-mPickingRadius, aScreenCoordinates[0]+mPickingRadius, aScreenCoordinates[1]+mPickingRadius);
	mFrameBuffer.Render();
}


}//namespace M4D

#endif /*OGL_SELECTION_H*/
