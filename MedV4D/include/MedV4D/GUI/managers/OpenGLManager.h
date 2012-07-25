#ifndef OPENGL_MANAGER_H
#define OPENGL_MANAGER_H

#include "MedV4D/GUI/utils/OGLTools.h"
#include "MedV4D/GUI/utils/GLTextureImage.h"
#include "MedV4D/Common/Common.h"

//#include <QtOpenGL>
#include <QGLWidget>
#include <QGLContext>
#include <QColor>

#include <boost/thread.hpp> 
#include <boost/thread/recursive_mutex.hpp>
//#include <boost/thread/unique_lock.hpp>

#include <map>

struct OpenGLManagerPimpl;


class OpenGLManager
{
public:
	struct CurrentContext
	{
		CurrentContext( OpenGLManager &aManager ): manager( aManager )
		{
			manager.makeCurrent();
		}

		~CurrentContext()
		{
			manager.doneCurrent();
		}

		OpenGLManager &manager;
	};	

	static OpenGLManager *
	getInstance();

	virtual void
	initialize();

	virtual void
	finalize();

	virtual 
	~OpenGLManager();

	QGLWidget *
	getSharedGLWidget();

	virtual M4D::GLTextureImage::Ptr
	getTextureFromImage( const M4D::Imaging::AImage &aImage );

	template< typename TFtor >
	TFtor
	doGL( TFtor ftor )
	{
		CurrentContext curr( *this );
		//makeCurrent();
		ftor();
		M4D::CheckForGLError( "OpenGL manager - doGL() call" );
		//doneCurrent();
		return ftor;
	}
	
	void
	deleteTextures( GLuint &aTexture, size_t aCount = 1 )
	{
		CurrentContext curr( *this );
		GL_CHECKED_CALL( glDeleteTextures( aCount, &aTexture ) );
	}
protected:
	virtual M4D::GLTextureImage::Ptr
	getActualizedTextureFromImage( const M4D::Imaging::AImage &aImage );
	
	virtual M4D::GLTextureImage::Ptr
	createNewTextureFromImage( const M4D::Imaging::AImage &aImage );
	
	
	void
	makeCurrent();
	void
	doneCurrent();

	OpenGLManager( OpenGLManager *aInstance );

	OpenGLManagerPimpl *mPimpl;

};


#endif /*OPENGL_MANAGER_H*/
