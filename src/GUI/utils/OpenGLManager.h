#ifndef OPENGL_MANAGER_H
#define OPENGL_MANAGER_H

#include "GUI/utils/OGLTools.h"
#include "GUI/utils/GLTextureImage.h"
#include "common/Common.h"
#include <QtOpenGL>
#include <boost/thread.hpp> 
#include <boost/thread/recursive_mutex.hpp>
//#include <boost/thread/unique_lock.hpp>

#include <map>

struct OpenGLManagerPimpl;

class OpenGLManager
{
public:
	

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

protected:
	void
	makeCurrent();
	void
	doneCurrent();

	OpenGLManager( OpenGLManager *aInstance );

	OpenGLManagerPimpl *mPimpl;

};


#endif /*OPENGL_MANAGER_H*/
