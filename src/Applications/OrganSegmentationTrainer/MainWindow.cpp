#include "MainWindow.h"
#include "Imaging.h"

using namespace M4D::Imaging;


MainWindow::MainWindow()
{
	Q_INIT_RESOURCE( MainWindow ); 

	resize( 800, 600 );

	CreateWidgets();
}

void
MainWindow::CreateWidgets()
{
	QWidget *centralWidget = new QWidget();
	QHBoxLayout *layout = new QHBoxLayout();

	QVBoxLayout *listLayout = new QVBoxLayout();
	layout->addLayout( listLayout, 5 );

	_fileList = new QListWidget();
	listLayout->addWidget( _fileList );

	QPushButton *loadButton = new QPushButton( "Load training sets..." );
	listLayout->addWidget( loadButton );
	QObject::connect( loadButton, SIGNAL( released() ), this, SLOT( ReloadTrainingSetInfos() ) );

	QVBoxLayout *rightLayout = new QVBoxLayout();
	layout->addStretch( 1 );
	layout->addLayout( rightLayout, 6 );
	layout->addStretch( 1 );
	rightLayout->addStretch( 1 );
	QGridLayout *gridLayout = new QGridLayout();
	rightLayout->addLayout( gridLayout, 6 );
	rightLayout->addStretch( 2 );	

	QLabel *label;
	QSpinBox *spinBox;
	QDoubleSpinBox *doubleSpinBox;

	int sampleRow = 0;
	int sizeRow = 2;
	int intervalRow = 4;
	label = new QLabel( "X Samples" );
	_xSamplesSB = spinBox = new QSpinBox();
	spinBox->setMinimum( 10 );
	spinBox->setMaximum( 512 );
	spinBox->setValue( 40 );
	gridLayout->addWidget( label, sampleRow, 0 );
	gridLayout->addWidget( spinBox, sampleRow, 1 );

	label = new QLabel( "Y Samples" );
	_ySamplesSB = spinBox = new QSpinBox();
	spinBox->setMinimum( 10 );
	spinBox->setMaximum( 512 );
	spinBox->setValue( 40 );
	gridLayout->addWidget( label, sampleRow, 2 );
	gridLayout->addWidget( spinBox, sampleRow, 3 );

	label = new QLabel( "Z Samples" );
	_zSamplesSB = spinBox = new QSpinBox();
	spinBox->setMinimum( 5 );
	spinBox->setMaximum( 200 );
	spinBox->setValue( 20 );
	gridLayout->addWidget( label, sampleRow, 4 );
	gridLayout->addWidget( spinBox, sampleRow, 5 );

	label = new QLabel( "X Size" );
	_xSizeSB = doubleSpinBox = new QDoubleSpinBox();
	doubleSpinBox->setMinimum( 10 );
	doubleSpinBox->setMaximum( 1500 );
	doubleSpinBox->setValue( 150 );
	gridLayout->addWidget( label, sizeRow, 0 );
	gridLayout->addWidget( doubleSpinBox, sizeRow, 1 );

	label = new QLabel( "Y Size" );
	_ySizeSB = doubleSpinBox = new QDoubleSpinBox();
	doubleSpinBox->setMinimum( 10 );
	doubleSpinBox->setMaximum( 1500 );
	doubleSpinBox->setValue( 150 );
	gridLayout->addWidget( label, sizeRow, 2 );
	gridLayout->addWidget( doubleSpinBox, sizeRow, 3 );

	label = new QLabel( "Histogram min" );
	_minHistogramSB = spinBox = new QSpinBox();
	spinBox->setMinimum( 0 );
	spinBox->setMaximum( 4095 );
	spinBox->setValue( 900 );
	gridLayout->addWidget( label, intervalRow, 0 );
	gridLayout->addWidget( spinBox, intervalRow, 1 );

	label = new QLabel( "Histogram max" );
	_maxHistogramSB = spinBox = new QSpinBox();
	spinBox->setMinimum( 0 );
	spinBox->setMaximum( 4095 );
	spinBox->setValue( 1100 );
	gridLayout->addWidget( label, intervalRow, 2 );
	gridLayout->addWidget( spinBox, intervalRow, 3 );
	
	rightLayout->addStretch( 4 );

	_trainButton = new QPushButton( "Train model" );
	rightLayout->addWidget( _trainButton );
	QObject::connect( _trainButton, SIGNAL( released() ), this, SLOT( ExecuteTraining() ) );

	_cancelTraining = new QPushButton( "Cancel training" );
	rightLayout->addWidget( _cancelTraining );

	QHBoxLayout *saveLayout = new QHBoxLayout();
	rightLayout->addLayout( saveLayout );
	_saveModel = new QPushButton( "Save model..." );
	QObject::connect( _saveModel, SIGNAL( released() ), this, SLOT( SaveTrainedModel() ) );
	saveLayout->addWidget( _saveModel );

	_saveVisualization = new QPushButton( "Save visualization..." );
	QObject::connect( _saveVisualization, SIGNAL( released() ), this, SLOT( SaveModelVisualization() ) );
	saveLayout->addWidget( _saveVisualization );
	
	rightLayout->addStretch( 2 );	


	_progressBar = new QProgressBar();
	rightLayout->addWidget( _progressBar, Qt::AlignCenter );
	rightLayout->addStretch( 1 );	

	centralWidget->setLayout( layout );
	this->setCentralWidget( centralWidget );
}

void
MainWindow::ReloadTrainingSetInfos()
{
	std::string dirName = QFileDialog::getExistingDirectory( this ).toStdString();

	if( dirName == "" ) {
		return;
	}
	_trainingsetInfos.clear();
	GetTrainingSetInfos( dirName, ".idx", _trainingsetInfos, true );

	UpdateList();
}

void
MainWindow::ExecuteTraining()
{
	Vector< uint32, 3 > size( 
			static_cast<uint32>(_xSamplesSB->value()), 
			static_cast<uint32>(_ySamplesSB->value()), 
			static_cast<uint32>(_zSamplesSB->value()) 
			); 
	Vector< float32, 3 > step( 
			_xSizeSB->value() / (size[0]-1), 
			_ySizeSB->value() / (size[1]-1), 
			1.0 / (size[2]-1) 
			);
	Vector< float32, 3 > origin( _xSizeSB->value() * 0.5f, _ySizeSB->value() * 0.5f, 0.0f );

	if( _trainingsetInfos.empty() ) {
		return;
	}

	
	_model = Train( _trainingsetInfos, size, step, origin, _minHistogramSB->value(), _maxHistogramSB->value() );

	TrainingFinished();
}

void
MainWindow::SaveTrainedModel()
{
	if( ! _model ) {
		return;
	}

	std::string fileName = QFileDialog::getSaveFileName( this ).toStdString();

	if( fileName == "" ) {
		return;
	}

	_model->SaveToFile( fileName );
	
	/*M4D::Imaging::CanonicalProbModel *test;
	test = CanonicalProbModel::LoadFromFile( fileName );
	ImageType::Ptr tmp;
	tmp = MakeImageFromProbabilityGrid<InProbabilityAccessor>( test->GetGrid(), InProbabilityAccessor() );
	ImageFactory::DumpImage( "pom.dump", *tmp );*/
}

void
MainWindow::SaveModelVisualization()
{
	if( ! _model ) {
		return;
	}

	std::string fileName = QFileDialog::getSaveFileName( this ).toStdString();

	if( fileName == "" ) {
		return;
	}

	ImageType::Ptr tmp;
	tmp = MakeImageFromProbabilityGrid<InProbabilityAccessor>( _model->GetGrid(), InProbabilityAccessor() );
	ImageFactory::DumpImage( fileName, *tmp );

	/*M4D::Imaging::CanonicalProbModel *test;
	test = CanonicalProbModel::LoadFromFile( fileName );
	ImageType::Ptr tmp;
	tmp = MakeImageFromProbabilityGrid<InProbabilityAccessor>( test->GetGrid(), InProbabilityAccessor() );
	ImageFactory::DumpImage( "pom.dump", *tmp );*/
}

void
MainWindow::TrainingFinished()
{
	QMessageBox::information( this, "Training finished", "Training finished" );
}

void
MainWindow::UpdateList()
{
	_fileList->clear();

	for( unsigned i = 0; i < _trainingsetInfos.size(); ++i ) {
		_fileList->addItem( QString( _trainingsetInfos[i].first.string().data() ) );
		_fileList->addItem( QString( _trainingsetInfos[i].second.string().data() ) );
	}
}


