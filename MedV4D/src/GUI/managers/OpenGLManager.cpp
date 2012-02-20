#include "MedV4D/GUI/managers/OpenGLManager.h"

class DummyOGLWidget: public QGLWidget
{
public:
	DummyOGLWidget(const QGLFormat & format ):QGLWidget( format )
	{
		makeCurrent();
		M4D::InitOpenGL();
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
	M4D::GLTextureImage::Ptr texture;
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
	glformat.setVersion( 3, 1 );

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

M4D::GLTextureImage::Ptr
OpenGLManager::getTextureFromImage( const M4D::Imaging::AImage &aImage )
{
	ASSERT( mPimpl );
	boost::unique_lock< boost::recursive_mutex > lock( mPimpl->mTextureMutex );

	M4D::GLTextureImage::Ptr result;
	result = getActualizedTextureFromImage( aImage );
	
	if ( result ) {
		return result;
	}
	// texture not available yet
	return createNewTextureFromImage( aImage );
}

M4D::GLTextureImage::Ptr
OpenGLManager::getActualizedTextureFromImage( const M4D::Imaging::AImage &aImage )
{
	//Image should be locked!!!
	
	M4D::Common::TimeStamp structTimestamp( aImage.GetStructureTimestamp() );
	M4D::Common::TimeStamp::IDType id = structTimestamp.getID();

	TextureStorage::iterator it = mPimpl->textureStorage.find( id );
	if ( it != mPimpl->textureStorage.end() ) { //We already created one instance

		M4D::Common::TimeStamp timestamp( aImage.GetEditTimestamp() );
		if ( structTimestamp != it->second.structTimeStamp ) { //Dataset structure changed
			return M4D::GLTextureImage::Ptr();
		}
		
		if ( timestamp == it->second.editTimeStamp ) { //Dataset contents didn't changed - texture is actual
			D_PRINT( "Returning valid texture instance" );
			return it->second.texture;
		} else {
			it->second.texture->DeleteTexture();
		}
	}
	return M4D::GLTextureImage::Ptr();
}


M4D::GLTextureImage::Ptr
OpenGLManager::createNewTextureFromImage( const M4D::Imaging::AImage &aImage )
{
	//Image should be locked!!!
	M4D::Common::TimeStamp structTimestamp( aImage.GetStructureTimestamp() );
	M4D::Common::TimeStamp::IDType id = structTimestamp.getID();
	M4D::Common::TimeStamp timestamp( aImage.GetEditTimestamp() );
	
	makeCurrent();
	TextureRecord rec;
	try {
		rec.texture = M4D::CreateTextureFromImage( *(aImage.GetAImageRegion()), true ) ;
	} catch ( M4D::GLException &e) {
		LOG_ERR( e.what() );
		throw;
	} catch ( ... ) {
		LOG_ERR( "Problem with texture creation" );
		throw;
	}
	rec.structTimeStamp = structTimestamp;
	rec.editTimeStamp = timestamp;
	mPimpl->textureStorage[ id ] = rec;
	doneCurrent();
	D_PRINT( "Returning newly created texture instance" );
	return rec.texture;
}

void
OpenGLManager::makeCurrent()
{
	//TODO handle in better way
	mPimpl->context->makeCurrent();
}

void
OpenGLManager::doneCurrent()
{
	//TODO handle in better way
	mPimpl->context->doneCurrent();
}


