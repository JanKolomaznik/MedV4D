#ifndef PROGRESS_INFO_DIALOG_H
#define PROGRESS_INFO_DIALOG_H

#include "MedV4D/generated/ui_ProgressInfoDialog.h"
//Temporary workaround
#ifndef Q_MOC_RUN 
#include "MedV4D/Common/ProgressNotifier.h"
#include "MedV4D/Common/Common.h"
#endif

class ProgressInfoDialog: public QDialog, public Ui::ProgressInfoDialog, public ProgressNotifier
{
	Q_OBJECT;
public:
	typedef std::shared_ptr< ProgressInfoDialog > Ptr;

	ProgressInfoDialog( QWidget * parent = 0 ): QDialog( parent )
	{
		setupUi( this );
		//setModal( true );
		QObject::connect( this, SIGNAL( maximumChangedSignal() ), this, SLOT( changeMaximumSlot() ), Qt::QueuedConnection );
		QObject::connect( this, SIGNAL( progressUpdatedSignal() ), this, SLOT( updateProgressSlot() ), Qt::QueuedConnection );
		QObject::connect( this, SIGNAL( finishedSignal() ), this, SLOT( close() ), Qt::QueuedConnection );
	}

	void
	init( size_t aPhaseCount )
	{
		ProgressNotifier::init( aPhaseCount );
	}

	void
	initNextPhase( size_t aStepCount )
	{
		ProgressNotifier::initNextPhase( aStepCount );
		
		emit maximumChangedSignal();
	}

	bool
	updateProgress( int aStepProgress = 1 )
	{
		bool res = ProgressNotifier::updateProgress( aStepProgress );
		emit progressUpdatedSignal();

		//boost::this_thread::sleep(boost::posix_time::milliseconds(250));

		//LOG( "Progress" );
		return res;		
	}

	void
	finished()
	{
		emit finishedSignal();
		//LOG( "finishedSignal emited" );
	}

public slots:
	void
	cancelAction()
	{

	}
	
	void
	pauseAction( bool aPause )
	{

	}
signals:
	void
	maximumChangedSignal();

	void
	progressUpdatedSignal();

	void
	finishedSignal();
protected slots:
	void
	changeMaximumSlot()
	{
		phaseProgressBar->setMaximum( static_cast<int>(mStepCount) );
		//LOG( "Current StepCOunt " << mStepCount );
	}
	void
	updateProgressSlot()
	{
		phaseProgressBar->setValue( static_cast<int>(mCurrentStep) );
		//LOG( "CurrentStep " << mCurrentStep );
	}
};


#endif /*PROGRESS_INFO_DIALOG_H*/
