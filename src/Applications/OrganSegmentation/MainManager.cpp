
#include "MainManager.h"

M4D::Dicom::DcmProvider::DicomObjSetPtr	MainManager::_inputDcmSet;
InputImagePtr				MainManager::_inputImage;
ImageConnectionPtr			MainManager::_inConnection;

void
MainManager::Initialize()
{
	_inConnection = ImageConnectionPtr( new M4D::Imaging::AbstractImageConnection() );
}

void
MainManager::Finalize()
{

}

void
MainManager::InitInput( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjSet )
{
	_inputDcmSet = dicomObjSet;
	_inputImage = M4D::Dicom::DcmProvider::CreateImageFromDICOM( dicomObjSet );

	try {
		_inConnection->PutImage( _inputImage );

		/*_convertor->Execute();

		mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
		_inConnection->ConnectConsumer( mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

		_settings->SetEnabledExecButton( true );*/

	} catch ( ... ) {
		throw;
	}

}
