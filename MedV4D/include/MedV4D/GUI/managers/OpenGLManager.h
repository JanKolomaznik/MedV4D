#ifndef OPENGL_MANAGER_H
#define OPENGL_MANAGER_H

//Temporary workaround
#ifndef Q_MOC_RUN
//#include "soglu/OGLTools.hpp"
#include <soglu/ErrorHandling.hpp>
#include <soglu/GLTextureImage.hpp>
#include "MedV4D/Imaging/AImage.h"
#include <boost/thread.hpp>
#include <boost/thread/recursive_mutex.hpp>
//#include <boost/thread/unique_lock.hpp>
#endif

//#include <QtOpenGL>
#include <QtOpenGL/QGLWidget>
#include <QtOpenGL/QGLContext>
#include <QColor>

//#include <soglu/OGLTools.hpp>

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

	virtual soglu::GLTextureImage::Ptr
	getTextureFromImage( const M4D::Imaging::AImage &aImage );

	template< typename TFtor >
	TFtor
	doGL( TFtor ftor )
	{
		CurrentContext curr( *this );
		//makeCurrent();
		ftor();
		soglu::checkForGLError( "OpenGL manager - doGL() call" );
		//doneCurrent();
		return ftor;
	}

	void
	deleteTextures( GLuint &aTexture, size_t aCount = 1 )
	{
		CurrentContext curr( *this );
		GL_CHECKED_CALL( glDeleteTextures( static_cast<GLsizei>(aCount), &aTexture ) );
	}
protected:
	virtual soglu::GLTextureImage::Ptr
	getActualizedTextureFromImage( const M4D::Imaging::AImage &aImage );

	virtual soglu::GLTextureImage::Ptr
	createNewTextureFromImage( const M4D::Imaging::AImage &aImage );


	void
	makeCurrent();
	void
	doneCurrent();

	OpenGLManager( OpenGLManager *aInstance );

	OpenGLManagerPimpl *mPimpl;

};


#endif /*OPENGL_MANAGER_H*/
