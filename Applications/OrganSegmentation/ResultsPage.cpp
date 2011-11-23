#include "ResultsPage.h"


ResultsPage::ResultsPage( MainManager &manager )
	: _manager( manager )
{
	QGridLayout *layout = new QGridLayout;
	QLabel *label;

	layout->setRowStretch( 2, 1 );
	label = new QLabel( "Organ voxel count = ");
	layout->addWidget( label, 3, 1 );

	_voxelCountLabel = new QLabel( "0" );
	_voxelCountLabel->setAlignment( Qt::AlignRight );
	layout->addWidget( _voxelCountLabel, 3, 3 );

	layout->setRowStretch( 4, 1 );
	label = new QLabel( "Organ volume = ");
	layout->addWidget( label, 5, 1 );

	_volumeValueLabel = new QLabel( "0 cm^3" );
	_volumeValueLabel->setAlignment( Qt::AlignRight );
	layout->addWidget( _volumeValueLabel, 5, 3 );

	layout->setRowStretch( 6, 3 );
	layout->setColumnStretch( 4, 3 );
	layout->setColumnStretch( 0, 1 );
	layout->setColumnStretch( 2, 1 );
	setLayout( layout );
}

void
ResultsPage::WaitForData()
{

}

void
ResultsPage::ShowResults( AnalysisRecord record )
{
	_voxelCountLabel->setText( QString::number( record.voxelCount ) );
	_volumeValueLabel->setText( QString::number( record.organVolume / 1000.f ) + QString( " cm^3" ) );
}
