#ifndef OGL_SELECTION_H
#define OGL_SELECTION_H

#include "MedV4D/GUI/utils/OGLTools.h"
#include "MedV4D/GUI/utils/CgShaderTools.h"
#include "MedV4D/GUI/utils/FrameBufferObject.h"
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <boost/shared_array.hpp>
#include <set>
namespace M4D
{

extern boost::filesystem::path gPickingShaderPath;
	
typedef size_t SelectID;

template < bool tIsInSelectionMode >
struct CurentObjectID;

template <>
struct CurentObjectID< true >
{
	void
	set( SelectID aId )
	{
		GL_CHECKED_CALL( glColor4us(static_cast<GLushort>(aId), 0, 0, MAX_UINT16) );
	}
};

template <>
struct CurentObjectID< false >
{
	void
	set( SelectID aId )
	{ /*empty*/ }
};

class PickManager
{
public:	
	struct ID_Depth_Pair
	{
		SelectID id;
		float depth;
	};
	typedef std::set< SelectID > SelectedIDsSet;
	void
	initialize( unsigned aPickingRadius );
	
	void
	finalize();
	
	template< typename TFunctor >
	void
	render( Vector2i aScreenCoordinates, const GLViewSetup &aViewSetup, TFunctor aFunctor );
	
	void
	getIDs( SelectedIDsSet &aIDs );

	~PickManager()
	{ finalize(); }
protected:
	typedef boost::shared_array< uint16 > BufferArray;
	void
	getBufferFromGPU();
	
	unsigned mPickingRadius;
	FrameBufferObject mFrameBuffer;
	M4D::GUI::CgEffect mCgEffect;
	BufferArray mBuffer;
};


template< typename TFunctor >
void
PickManager::render( Vector2i aScreenCoordinates, const GLViewSetup &aViewSetup, TFunctor aFunctor )
{
	ASSERT(isGLContextActive());
	try {
	M4D::GLPushAtribs pushAttribs;
	mFrameBuffer.Bind();
	glClearColor(0.0f,0.0f,0.0f,0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	//glColor4f( 0.0f, 0.0f, 0.3f, 0.5f );
	//DrawRectangleOverViewPort(int(aScreenCoordinates[0]-mPickingRadius), int(aScreenCoordinates[1]-mPickingRadius), int(aScreenCoordinates[0]+mPickingRadius), int(aScreenCoordinates[1]+mPickingRadius));
	
	glm::dmat4x4 pick = glm::pickMatrix(
			glm::dvec2(aScreenCoordinates[0],aScreenCoordinates[1]),
			glm::dvec2(2*mPickingRadius,2*mPickingRadius),
			aViewSetup.viewport
		);
	GL_CHECKED_CALL( glMatrixMode( GL_PROJECTION ) );
	GL_CHECKED_CALL( glPushMatrix() );
	GL_CHECKED_CALL( glLoadMatrixd( glm::value_ptr(pick*aViewSetup.projection) ) );
	glViewport(aScreenCoordinates[0]-mPickingRadius, aScreenCoordinates[1]-mPickingRadius, 2*mPickingRadius, 2*mPickingRadius);
	//glViewport(0, 0, 2*mPickingRadius, 2*mPickingRadius);
	//glColor4f( 0.0f, 0.3f, 0.0f, 0.5f );
	//GL_CHECKED_CALL(DrawRectangleOverViewPort(-1000, -1000, 1000, 1000));
	
	mCgEffect.ExecuteTechniquePass( "PickingEffect", aFunctor );
	
	//glClearColor(0.2f,0.2f, 0.3f, 1.0f);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	GL_CHECKED_CALL( glPopMatrix() );
	mFrameBuffer.Unbind();
	
	//GL_CHECKED_CALL( glViewport(aScreenCoordinates[0]-mPickingRadius, aScreenCoordinates[1]-mPickingRadius, aScreenCoordinates[0]+mPickingRadius, aScreenCoordinates[1]+mPickingRadius) );
	//mFrameBuffer.Render();
	CheckForGLError( "Selection rendering" );
	} catch (std::exception &e) {
		LOG( e.what() );
		throw;
	} catch(...) {
		LOG( "Picking error" );
		throw;
	}
}


}//namespace M4D

#endif /*OGL_SELECTION_H*/
