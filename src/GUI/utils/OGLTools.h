#ifndef OGL_TOOLS_H
#define OGL_TOOLS_H


namespace M4D
{

#ifdef USE_DEVIL
void
SaveTextureToImageFile( uint32 aWidth, uint32 aHeight, GLuint aTexture, std::string aPath, bool aOverwrite = false );
#endif /*USE_DEVIL*/

} /*namespace M4D*/

#endif /*OGL_TOOLS_H*/


