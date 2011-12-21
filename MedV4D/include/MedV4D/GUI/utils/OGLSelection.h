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
	//try {
	M4D::GLPushAtribs pushAttribs;
	//mFrameBuffer.Bind();
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glColor4f( 0.0f, 0.0f, 0.3f, 0.5f );
	DrawRectangleOverViewPort(aScreenCoordinates[0]-mPickingRadius, aScreenCoordinates[1]-mPickingRadius, aScreenCoordinates[0]+mPickingRadius, aScreenCoordinates[1]+mPickingRadius);
	
	glm::dmat4x4 pick = glm::pickMatrix(
			glm::dvec2(aScreenCoordinates[0],aScreenCoordinates[1]),
			glm::dvec2(mPickingRadius,mPickingRadius),
			aViewSetup.viewport
		);
	GL_CHECKED_CALL( glMatrixMode( GL_PROJECTION ) );
	GL_CHECKED_CALL( glPushMatrix() );
	GL_CHECKED_CALL( glLoadMatrixd( glm::value_ptr(pick*aViewSetup.projection) ) );
	glViewport(aScreenCoordinates[0]-mPickingRadius, aScreenCoordinates[1]-mPickingRadius, aScreenCoordinates[0]+mPickingRadius, aScreenCoordinates[1]+mPickingRadius);
	//glViewport(0, 0, 2*mPickingRadius, 2*mPickingRadius);
	
	mCgEffect.ExecuteTechniquePass( "PickingEffect", aFunctor );
	
	//glClearColor(0.2f,0.2f, 0.3f, 1.0f);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	GL_CHECKED_CALL( glPopMatrix() );
	//mFrameBuffer.Unbind();
	
	//GL_CHECKED_CALL( glViewport(aScreenCoordinates[0]-mPickingRadius, aScreenCoordinates[1]-mPickingRadius, aScreenCoordinates[0]+mPickingRadius, aScreenCoordinates[1]+mPickingRadius) );
	//mFrameBuffer.Render();
	CheckForGLError( "Selection rendering" );
	/*} catch (std::exception &e) {
		LOG( e.what() );
		throw;
	} catch(...) {
		LOG( "Picking error" );
		throw;
	}*/
}


}//namespace M4D

#endif /*OGL_SELECTION_H*/
