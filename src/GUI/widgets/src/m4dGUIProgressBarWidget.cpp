/**
 *  @ingroup gui
 *  @file m4dGUIProgressBarWidget.cpp
 *  @brief some brief
 */
#include "GUI/widgets/m4dGUIProgressBarWidget.h"

#include <QtGui>


namespace M4D {
namespace GUI {

/// Duration of the timeline in milliseconds
#define TIME_LINE_DURATION      1000
/// Frame range of the timeline 
#define FRAME_RANGE   100

m4dGUIProgressBarWidget::m4dGUIProgressBarWidget ( const QString &label, QWidget *parent )
  : QWidget( parent )
{
  QLabel *descriptionLabel = new QLabel( label );

  progressBar = new QProgressBar;
  progressBar->setRange( 0, FRAME_RANGE );
  progressBar->setTextVisible( false );

  timeLine = new QTimeLine( TIME_LINE_DURATION, this );
  timeLine->setFrameRange( 0, FRAME_RANGE );
  timeLine->setLoopCount( 0 );

  connect( timeLine, SIGNAL(frameChanged( int )), progressBar, SLOT(setValue( int )) );

  // building layout
  QVBoxLayout *mainLayout = new QVBoxLayout;
  mainLayout->addWidget( descriptionLabel );
  mainLayout->addWidget( progressBar );
  setLayout( mainLayout );
}


void m4dGUIProgressBarWidget::start ()
{ 
  timeLine->start();
}


void m4dGUIProgressBarWidget::stop ()
{ 
  timeLine->setPaused( true );

  emit ready();
}


void m4dGUIProgressBarWidget::pause ()
{ 
  timeLine->setPaused( true );
}

} // namespace GUI
} // namespace M4D

