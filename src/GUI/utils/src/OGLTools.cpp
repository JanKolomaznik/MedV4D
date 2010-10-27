#include "GUI/utils/OGLTools.h"

namespace M4D
{

#ifdef USE_DEVIL
void
SaveTextureToImageFile( uint32 aWidth, uint32 aHeight, GLuint aTexture, std::string aPath, bool aOverwrite )
{
	ILuint imageID; // The image name to return.
	ilGenImages( 1, &imageID );
	ilBindImage( imageID );
	if ( aOverwrite ) {
		ilEnable(IL_FILE_OVERWRITE);
	}

	ILboolean result = ilTexImage( aWidth, aHeight, 0, 3, IL_RGB, IL_UNSIGNED_BYTE, NULL );
	ILubyte* data = ilGetData(ILvoid);

	glBind( GL_TEXTURE_2D, aTexture );
	glGetTexImage(	GL_TEXTURE_2D, 
			0, 
			GL_RGB, 
			GL_UNSIGNED_BYTE, 
			(void*)data
			);
	glBind( GL_TEXTURE_2D, 0 );

	ilSaveImage( aPath.data() );
	ilDeleteImages( 1, &imageID);
}
#endif /*USE_DEVIL*/


} /*namespace M4D*/

