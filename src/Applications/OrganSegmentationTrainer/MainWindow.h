#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QtGui>
#include "ImageTools.h"

class MainWindow: public QMainWindow
{
	Q_OBJECT

public:

	MainWindow ();

public slots:
	void
	ReloadTrainingSetInfos();

	void
	UpdateList();

	void
	ExecuteTraining();

	void
	TrainingFinished();
protected:
	void
	CreateWidgets();
private:
	QListWidget	*_fileList;
	QProgressBar	*_progressBar;
	QSpinBox	*_xSamplesSB; 
	QSpinBox	*_ySamplesSB; 
	QSpinBox	*_zSamplesSB; 
	QSpinBox	*_maxHistogramSB; 
	QSpinBox	*_minHistogramSB; 
	QDoubleSpinBox	*_xSizeSB;
	QDoubleSpinBox	*_ySizeSB;
	QPushButton	*_trainButton;
	QPushButton	*_cancelTraining;
	QPushButton	*_saveModel;
	QPushButton	*_saveVisualization;

	TrainingDataInfos _trainingsetInfos;
};


#endif // MAIN_WINDOW_H


