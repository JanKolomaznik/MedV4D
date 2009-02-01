
#include "MainManager.h"

M4D::Dicom::DcmProvider::DicomObjSetPtr	MainManager::_inputDcmSet;
InputImagePtr				MainManager::_inputImage;
InImageConnection *			MainManager::_inConnection;
ImageConnectionType *			MainManager::_inConvConnection;
M4D::Imaging::PipelineContainer		MainManager::_conversionPipeline;

void
MainManager::Initialize()
{
	M4D::Imaging::AbstractPipeFilter *filter = new M4D::Imaging::ImageConvertor< InputImageType >();
	_conversionPipeline.AddFilter( filter );
	_inConnection =  static_cast< InImageConnection * >(&(_conversionPipeline.MakeInputConnection( *filter, 0, false )));
	_inConvConnection =  static_cast< ImageConnectionType * >(&(_conversionPipeline.MakeOutputConnection( *filter, 0, true ) ));
}

void
MainManager::Finalize()
{

}

void
MainManager::InitInput( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjSet )
{
	_inputDcmSet = dicomObjSet;

	AbstractImagePtr image = M4D::Dicom::DcmProvider::CreateImageFromDICOM( dicomObjSet );

	try {
		_inConnection->PutImage( image );

		_conversionPipeline.ExecuteFirstFilter();
		/*_convertor->Execute();

		mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
		_inConnection->ConnectConsumer( mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

		_settings->SetEnabledExecButton( true );*/

	} catch ( ... ) {
		throw;
	}

}
