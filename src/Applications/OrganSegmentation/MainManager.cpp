
#include "MainManager.h"

MainManager 	* MainManager::_instance;

MainManager &
MainManager::Instance()
{
	if( _instance == NULL ) {
		_instance = new MainManager();
		//_THROW_ EInstanceUnavailable();
	}
	return *_instance;
}
void
MainManager::Initialize()
{
	M4D::Imaging::AbstractPipeFilter *filter = new M4D::Imaging::ImageConvertor< InputImageType >();
	_conversionPipeline.AddFilter( filter );
	_inConnection =  static_cast< InImageConnection * >(&(_conversionPipeline.MakeInputConnection( *filter, 0, false )));
	_inConvConnection =  static_cast< ImageConnectionType * >(&(_conversionPipeline.MakeOutputConnection( *filter, 0, true ) ));


	CreateResultProcessPipeline();
}

void
MainManager::Finalize()
{

}

void
MainManager::InitInput( M4D::Dicom::DicomObjSetPtr dicomObjSet )
{
	_inputDcmSet = dicomObjSet;

	AbstractImagePtr image = M4D::Dicom::DcmProvider::CreateImageFromDICOM( dicomObjSet );

	try {
		_inConnection->PutDataset( image );

		_conversionPipeline.ExecuteFirstFilter();
		/*_convertor->Execute();

		mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
		_inConnection->ConnectConsumer( mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

		_settings->SetEnabledExecButton( true );*/

	} catch ( ... ) {
		throw;
	}

}

void
MainManager::ProcessResultDatasets( InputImagePtr image, GDataSet::Ptr splines )
{
	_inResultGDatasetConnection->PutDataset( splines );
	
	_splineFillFilter->SetMinimum( image->GetMinimum() );
	_splineFillFilter->SetMaximum( image->GetMaximum() );
	_splineFillFilter->SetElementExtents( image->GetElementExtents() );

	D_PRINT( "Processing results - min = " << image->GetMinimum() << " max = " << image->GetMaximum() << " elemExtents = " << image->GetElementExtents() );

	_splineFillFilter->ExecuteOnWhole();

	while( _splineFillFilter->IsRunning() ) { }

	M4D::Imaging::ImageFactory::DumpImage( "BLABLA.dump", _resultProcessMaskConnection->GetDatasetTyped() );
}

void
MainManager::CreateResultProcessPipeline()
{
	_splineFillFilter = new M4D::Imaging::SliceSplineFill< float32 >();
	_resultProcessingPipeline.AddFilter( _splineFillFilter );

	_inResultGDatasetConnection = (GDatasetConnectionType *)&(_resultProcessingPipeline.MakeInputConnection( *_splineFillFilter, 0, false ));
	_resultProcessMaskConnection = (Mask3DConnectionType *)&(_resultProcessingPipeline.MakeOutputConnection( *_splineFillFilter, 0, true ));

}

