#ifndef RESULTS_PAGE_H
#define RESULTS_PAGE_H

#include <QtGui>
#include "MainManager.h"
#include "AnalyseResults.h"

class ResultsPage : public QWidget
{
	Q_OBJECT;
public:
	ResultsPage( MainManager &manager );

public slots:
	void
	WaitForData();

	void
	ShowResults( AnalysisRecord record );
protected:
	MainManager &_manager;

	QLabel	*_volumeValueLabel;
};

#endif /*RESULTS_PAGE_H*/
