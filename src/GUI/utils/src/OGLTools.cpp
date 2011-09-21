#include "GUI/utils/OGLTools.h"
#include "GUI/utils/OGLDrawing.h"

#ifdef USE_DEVIL
#include <IL/il.h>
#include <IL/ilu.h>
#endif /*USE_DEVIL*/	


namespace M4D
{

void 
getCurrentGLSetup( GLViewSetup &aSetup )
{
	glGetDoublev( GL_PROJECTION_MATRIX, aSetup.model );
	glGetDoublev( GL_MODELVIEW_MATRIX, aSetup.proj );
	glGetIntegerv( GL_VIEWPORT, aSetup.view );
	CheckForGLError( "getCurrentGLSetup()" );
};

void
getPointAndDirectionFromScreenCoordinates( Vector2f aScreenCoords, const GLViewSetup &aViewSetup, Point3Df &point, Vector3f &direction )
{
	Vector3d objCoords1;
	Vector3d objCoords2;
	GLint res = gluUnProject(
			aScreenCoords[0],  
			aScreenCoords[1],  
			0.2,  
			aViewSetup.model,  
			aViewSetup.proj,  
			aViewSetup.view,  
			&(objCoords1[0]),  
			&(objCoords1[1]),  
			&(objCoords1[2])
			);
	if( res == GLU_FALSE ) {
		_THROW_ GLException( "Cannot unproject screen coordinates" );
	}
	res = gluUnProject(
			aScreenCoords[0],  
			aScreenCoords[1],  
			0.95,  
			aViewSetup.model,  
			aViewSetup.proj,  
			aViewSetup.view,  
			&(objCoords2[0]),  
			&(objCoords2[1]),  
			&(objCoords2[2])
			);
	if( res == GLU_FALSE ) {
		_THROW_ GLException( "Cannot unproject screen coordinates" );
	}
	
	LOG( "screen : " << aScreenCoords );
	LOG( "coords1 : " << objCoords1 );
	LOG( "coords2 : " << objCoords2 );
	point = objCoords1;
	direction = objCoords2-objCoords1;
}


void 
CheckForGLError( const std::string &situation  )
{
	GLenum errorCode = glGetError();
	if (errorCode != GL_NO_ERROR) {
		const char *string = (const char *)gluErrorString(errorCode);
		_THROW_ GLException( TO_STRING( situation << " : " << string ) );
	}
}


#ifdef USE_DEVIL

void 
CheckForDevILError( const std::string &situation  )
{
	ILenum errorCode = glGetError();
	if( errorCode != IL_NO_ERROR ) {
		std::stringstream finalMessage; situation;
		finalMessage << situation;
		while (errorCode != IL_NO_ERROR) {
			const char *string = (const char *)gluErrorString(errorCode);
			finalMessage << " : " << string << std::endl;
			errorCode = glGetError();
		}
		_THROW_ DevILException( finalMessage.str() );
	}
}

void
SaveTextureToImageFile( uint32 aWidth, uint32 aHeight, GLuint aTexture, std::string aPath, bool aOverwrite )
{
	//TODO auto_ptr on textures
	ILuint imageID; // The image name to return.
	DEVIL_CHECKED_CALL( ilGenImages( 1, &imageID ) );
	DEVIL_CHECKED_CALL( ilBindImage( imageID ) );
	if ( aOverwrite ) {
		DEVIL_CHECKED_CALL( ilEnable(IL_FILE_OVERWRITE) );
	} else {
		DEVIL_CHECKED_CALL( ilDisable(IL_FILE_OVERWRITE) );
	}

	DEVIL_CHECKED_CALL( ilTexImage( aWidth, aHeight, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, NULL ) );
	ILubyte* data = new ILubyte[ 3 * aWidth * aHeight ];

	GL_CHECKED_CALL( glBindTexture( GL_TEXTURE_2D, aTexture ) );
	GL_CHECKED_CALL( glGetTexImage(	
				GL_TEXTURE_2D, 
				0, 
				GL_RGB, 
				GL_UNSIGNED_BYTE, 
				(void*)data
				) );
	GL_CHECKED_CALL( glBindTexture( GL_TEXTURE_2D, 0 ) );

	DEVIL_CHECKED_CALL( ilSetPixels( 0, 0, 0, aWidth, aHeight, 1, IL_RGB, IL_UNSIGNED_BYTE, data ) );
	delete [] data;

	DEVIL_CHECKED_CALL( ilSaveImage( aPath.data() ) );
	DEVIL_CHECKED_CALL( ilDeleteImages( 1, &imageID) );
}
#endif /*USE_DEVIL*/

void
InitOpenGL()
{
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		_THROW_ M4D::ErrorHandling::EInitError( "GLEW" );
	}
	LOG( "Status: Using GLEW " << glewGetString(GLEW_VERSION) );
	LOG( "\tGLEW_VERSION_1_1 " << ((GLEW_VERSION_1_1) ? std::string("OK") : std::string("FAIL")) );
	LOG( "\tGLEW_VERSION_1_2 " << ((GLEW_VERSION_1_2) ? std::string("OK") : std::string("FAIL")) );
	LOG( "\tGLEW_VERSION_1_3 " << ((GLEW_VERSION_1_3) ? std::string("OK") : std::string("FAIL")) );
	LOG( "\tGLEW_VERSION_1_4 " << ((GLEW_VERSION_1_4) ? std::string("OK") : std::string("FAIL")) );
	LOG( "\tGLEW_VERSION_1_5 " << ((GLEW_VERSION_1_5) ? std::string("OK") : std::string("FAIL")) );
	LOG( "\tGLEW_VERSION_2_0 " << ((GLEW_VERSION_2_0) ? std::string("OK") : std::string("FAIL")) );
	LOG( "\tGLEW_VERSION_2_1 " << ((GLEW_VERSION_2_1) ? std::string("OK") : std::string("FAIL")) );
	LOG( "\tGLEW_VERSION_3_0 " << ((GLEW_VERSION_3_0) ? std::string("OK") : std::string("FAIL")) );
	LOG( "\tGLEW_VERSION_3_1 " << ((GLEW_VERSION_3_1) ? std::string("OK") : std::string("FAIL")) );
	LOG( "\tGLEW_VERSION_3_2 " << ((GLEW_VERSION_3_2) ? std::string("OK") : std::string("FAIL")) );
}


} /*namespace M4D*/

