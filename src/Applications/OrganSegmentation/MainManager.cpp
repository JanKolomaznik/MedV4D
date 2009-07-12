
#include "MainManager.h"
#include "ResultsPage.h"
#include <QtGui>

#include <boost/filesystem.hpp>

typedef boost::filesystem::path	Path;

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
	qRegisterMetaType<AnalysisRecord>();

	M4D::Imaging::AbstractPipeFilter *filter = new M4D::Imaging::ImageConvertor< InputImageType >();
	_conversionPipeline.AddFilter( filter );
	_inConnection =  static_cast< InImageConnection * >(&(_conversionPipeline.MakeInputConnection( *filter, 0, false )));
	_inConvConnection =  static_cast< ImageConnectionType * >(&(_conversionPipeline.MakeOutputConnection( *filter, 0, true ) ));


	CreateResultProcessPipeline();

	_resultsPage = new ResultsPage( *this );
	
	QObject::connect( this, SIGNAL(ResultProcessingStarted()), _resultsPage, SLOT( WaitForData() ), Qt::QueuedConnection );
	QObject::connect( this, SIGNAL(ShowResultsSignal( AnalysisRecord )), _resultsPage, SLOT( ShowResults( AnalysisRecord ) ), Qt::QueuedConnection );

	//TODO
}

void
MainManager::Finalize()
{

}

QWidget *
MainManager::GetResultsPage()
{ 
	return _resultsPage; 
}

void
MainManager::InitInput( M4D::Imaging::AbstractDataSet::Ptr inputDataSet )
{
//	_inputDcmSet = dicomObjSet;


	try {
		AbstractImagePtr image = boost::static_pointer_cast< M4D::Imaging::AbstractImage >( inputDataSet );

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

struct ResultProcessingThread
{
	ResultProcessingThread( 
		MainManager	*manager 
		)
		: _manager( manager ) 
	{ /*empty*/ }

	/**
	 * Method executed by thread, which has copy of this object.
	 **/
	void
	operator()()
	{
		D_PRINT( "Entering ProcessResultDatasetsThreadMethod()" );
		_manager->ProcessResultDatasetsThreadMethod();
	}
private:
	MainManager	*_manager;

};

void
MainManager::ProcessResultDatasets( InputImageType::Ptr image, GDataSet::Ptr splines )
{
	emit ResultProcessingStarted();

	_tmpImage = image;
	_tmpSplines = splines;

	M4D::Multithreading::Thread thread( ResultProcessingThread( this ) );

}

void
MainManager::CreateResultProcessPipeline()
{
	_splineFillFilter = new M4D::Imaging::SliceSplineFill< float32 >();
	_resultProcessingPipeline.AddFilter( _splineFillFilter );

	_inResultGDatasetConnection = (GDatasetConnectionType *)&(_resultProcessingPipeline.MakeInputConnection( *_splineFillFilter, 0, false ));
	_resultProcessMaskConnection = (Mask3DConnectionType *)&(_resultProcessingPipeline.MakeOutputConnection( *_splineFillFilter, 0, true ));

}

void
MainManager::ProcessResultDatasetsThreadMethod()
{
	_inResultGDatasetConnection->PutDataset( _tmpSplines );
	
	_splineFillFilter->SetMinimum( _tmpImage->GetMinimum() );
	_splineFillFilter->SetMaximum( _tmpImage->GetMaximum() );
	_splineFillFilter->SetElementExtents( _tmpImage->GetElementExtents() );

	D_PRINT( "Processing results - min = " << _tmpImage->GetMinimum() << " max = " << _tmpImage->GetMaximum() << " elemExtents = " << _tmpImage->GetElementExtents() );

	_splineFillFilter->ExecuteOnWhole();


	while( _splineFillFilter->IsRunning() ) { }

	_tmpMask = _resultProcessMaskConnection->GetDatasetPtrTyped();

	AnalysisRecord record;
	AnalyseResults( *_tmpImage, *_tmpMask, record );

	emit ShowResultsSignal( record );

}

void
MainManager::SaveResultDatasets()
{
	QString name = QFileDialog::getSaveFileName();

	Path maskName = TO_STRING( name.toStdString() << "Mask.dump" );
	Path dataName = TO_STRING( name.toStdString() << "Data.dump" );
	Path indexName = TO_STRING( name.toStdString() << ".idx" );

	M4D::Imaging::ImageFactory::DumpImage( maskName.file_string(), *_tmpMask );
	M4D::Imaging::ImageFactory::DumpImage( dataName.file_string(), *_tmpImage );

	std::ofstream indexFile( indexName.file_string().data() );

	indexFile << dataName.filename() << std::endl;
	indexFile << maskName.filename() << std::endl;

	indexFile.close();
	
	QMessageBox::information( NULL, "Saving finished", "Results saved" );

}
