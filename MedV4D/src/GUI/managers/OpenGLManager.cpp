#include <GL/glew.h>
#include "MedV4D/GUI/managers/OpenGLManager.h"
#include "MedV4D/Common/RAII.h"
#include <soglu/OGLTools.hpp>
#include <MedV4D/GUI/utils/TextureUtils.h>
#include <QShowEvent>
#include <memory>

#include <boost/scope_exit.hpp>

class DummyOGLWidget: public QGLWidget
{
public:
	DummyOGLWidget(const QGLFormat & format, QWidget *parent = nullptr):QGLWidget( format, parent, nullptr, Qt::Tool )
	{
		makeCurrent();
		soglu::initOpenGL();
		doneCurrent();
	}
protected:
	void
	initializeGL()
	{
	}
};


struct TextureRecord
{
	soglu::GLTextureImage::Ptr texture;
	M4D::Common::TimeStamp editTimeStamp; //<Contents timestamp
	M4D::Common::TimeStamp structTimeStamp;
};
typedef std::map< M4D::Common::TimeStamp::IDType, TextureRecord > TextureStorage;

struct OpenGLManagerPimpl
{
	QGLWidget *widget;
	QGLContext *context;

	TextureStorage textureStorage; //<Indexed by ID from structural timestamp
	mutable boost::recursive_mutex mTextureMutex;
};

OpenGLManager *oglManagerInstance = NULL;
//*******************************************************************************************
OpenGLManager *
OpenGLManager::getInstance()
{
	ASSERT( oglManagerInstance );
	return oglManagerInstance;
}

OpenGLManager::OpenGLManager( OpenGLManager *aInstance )
{
	ASSERT( aInstance );
	oglManagerInstance = aInstance;
}

OpenGLManager::~OpenGLManager()
{

}

void
OpenGLManager::initialize()
{
	mPimpl = new OpenGLManagerPimpl;

	QGLFormat glformat = QGLFormat::defaultFormat();
	glformat.setVersion( 3, 3 );
	glformat.setProfile(QGLFormat::CoreProfile);

	mPimpl->widget = new DummyOGLWidget(glformat);
	glformat = mPimpl->widget->format();
	LOG( "OpenGL version : " << glformat.majorVersion() << "." << glformat.minorVersion() );

	mPimpl->context = const_cast< QGLContext * >( mPimpl->widget->context() );
}

void
OpenGLManager::finalize()
{
	makeCurrent();
	mPimpl->textureStorage.clear();
	doneCurrent();

	delete mPimpl->widget;

	delete mPimpl;
}

QGLWidget *
OpenGLManager::getSharedGLWidget()
{
	ASSERT( mPimpl );
	return mPimpl->widget;
}

soglu::GLTextureImage::Ptr
OpenGLManager::getTextureFromImage( const M4D::Imaging::AImage &aImage )
{
	ASSERT( mPimpl );
	D_FUNCTION_COMMENT;
	boost::unique_lock< boost::recursive_mutex > lock( mPimpl->mTextureMutex );

	soglu::GLTextureImage::Ptr result;
	result = getActualizedTextureFromImage( aImage );

	if ( result ) {
		return result;
	}
	// texture not available yet
	return createNewTextureFromImage( aImage );
}

soglu::GLTextureImage::Ptr
OpenGLManager::getActualizedTextureFromImage( const M4D::Imaging::AImage &aImage )
{
	D_FUNCTION_COMMENT;
	//Image should be locked!!!
	makeCurrent();
	BOOST_SCOPE_EXIT_ALL(this) {
		doneCurrent();
	};

	M4D::Common::TimeStamp structTimestamp( aImage.GetStructureTimestamp() );
	M4D::Common::TimeStamp::IDType id = structTimestamp.getID();

	TextureStorage::iterator it = mPimpl->textureStorage.find( id );
	if ( it != mPimpl->textureStorage.end() ) { //We already created one instance
		M4D::RAII makeCurrentContext( boost::bind( &OpenGLManager::makeCurrent, this ), boost::bind( &OpenGLManager::doneCurrent, this ) );
		M4D::Common::TimeStamp timestamp( aImage.GetEditTimestamp() );
		if ( structTimestamp != it->second.structTimeStamp ) { //Dataset structure changed
			return soglu::GLTextureImage::Ptr();
		}

		if ( timestamp == it->second.editTimeStamp ) { //Dataset contents didn't changed - texture is actual
			D_PRINT( "Returning valid texture instance" );
			return it->second.texture;
		} else {
			Vector3i c1, c2;
			M4D::Imaging::AImageDim<3>::Cast( aImage ).getChangedRegionSinceTimestamp( c1, c2, it->second.editTimeStamp );
			LOG( "Edited : " << c1 << " => " << c2 );
			M4D::updateTextureSubImage( *(it->second.texture), *(aImage.GetAImageRegion()), c1, c2 );
			//M4D::recreateTextureFromImage( *(it->second.texture), *(aImage.GetAImageRegion()) );
			it->second.editTimeStamp = timestamp;
			return it->second.texture;
		}
	}
	return soglu::GLTextureImage::Ptr();
}


soglu::GLTextureImage::Ptr
OpenGLManager::createNewTextureFromImage( const M4D::Imaging::AImage &aImage )
{
	D_FUNCTION_COMMENT;
	//Image should be locked!!!
	M4D::Common::TimeStamp structTimestamp( aImage.GetStructureTimestamp() );
	M4D::Common::TimeStamp::IDType id = structTimestamp.getID();
	M4D::Common::TimeStamp timestamp( aImage.GetEditTimestamp() );
	//M4D::RAII makeCurrentContext( boost::bind( &OpenGLManager::makeCurrent, this ), boost::bind( &OpenGLManager::doneCurrent, this ) );
	TextureRecord rec;
	makeCurrent();
	BOOST_SCOPE_EXIT_ALL(this) {
		doneCurrent();
	};
	soglu::checkForGLError("Before make current");
	{
		soglu::checkForGLError("After make current");
		try {
			rec.texture = M4D::createTextureFromImage( *(aImage.GetAImageRegion()), true ) ;
		} catch ( std::exception &e) {
			LOG_ERR( "Create texture " << e.what() );
			throw;
		}
		rec.structTimeStamp = structTimestamp;
		rec.editTimeStamp = timestamp;
		mPimpl->textureStorage[ id ] = rec;
	}
	doneCurrent();
	D_PRINT( "Returning newly created texture instance" );
	return rec.texture;
}

void
OpenGLManager::makeCurrent()
{
	//TODO handle in better way
	GL_ERROR_CLEAR_AFTER_CALL(mPimpl->context->makeCurrent());
}

void
OpenGLManager::doneCurrent()
{
	//TODO handle in better way
	mPimpl->context->doneCurrent();
}




