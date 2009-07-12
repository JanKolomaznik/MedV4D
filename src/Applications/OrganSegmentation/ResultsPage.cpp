#include "ResultsPage.h"


ResultsPage::ResultsPage( MainManager &manager )
	: _manager( manager )
{
	QVBoxLayout *layout = new QVBoxLayout;

	_volumeValueLabel = new QLabel( "0 mm^3" );
	layout->addWidget( _volumeValueLabel );

	setLayout( layout );
}

void
ResultsPage::WaitForData()
{

}

void
ResultsPage::ShowResults( AnalysisRecord record )
{
	_volumeValueLabel->setText( QString::number( record.organVolume ) + QString( " mm^3" ) );
}
