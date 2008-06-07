#include "GUI/m4dGUIScreenLayoutWidget.h"

#include <QtGui>


/// Number of various layouts - in one groupBox
#define LAYOUT_NUMBER   9
/// Number of layouts in one row (in one groupBox)
#define LAYOUTS_IN_ROW  3

const char *m4dGUIScreenLayoutWidget::layoutIconNames[] = { "layout-1x1.png", "layout-2x1.png", 
                                                            "layout-1x2.png", "layout-2x2.png",
                                                            "layout-4x2.png", "layout-2x4.png",
                                                            "layout-4x4.png", "layout-6x4.png", 
                                                            "layout-8x4.png" };

m4dGUIScreenLayoutWidget::m4dGUIScreenLayoutWidget ( m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget,
                                                     QDialog *screenLayoutDialog, QWidget *parent )
  : QWidget( parent ),
    screenLayoutDialog( screenLayoutDialog )
{
  Q_INIT_RESOURCE( m4dGUIScreenLayoutWidget );

  // creating Series groupBox
  QGroupBox *seriesGroupBox = createSeriesGroupBox();
  // creating Image groupBox
  QGroupBox *imageGroupBox = createImageGroupBox();
  // creating OK Cancel dialogButtonBox
  QDialogButtonBox *dialogButtonBox = new QDialogButtonBox( QDialogButtonBox::Ok | QDialogButtonBox::Cancel );
  connect( dialogButtonBox, SIGNAL(accepted()), this, SLOT(accept()) );
  connect( dialogButtonBox, SIGNAL(rejected()), this, SLOT(reject()) );

  // building layout
  QGridLayout *mainLayout = new QGridLayout;
  mainLayout->addWidget( seriesGroupBox, 0, 0 );
  mainLayout->addWidget( imageGroupBox, 0, 1 );
  mainLayout->addItem( new QSpacerItem( 2, 15, QSizePolicy::Minimum, QSizePolicy::Expanding ), 1, 0, 1, 2 );
  mainLayout->addWidget( dialogButtonBox, 2, 0, 1, 2 );
  setLayout( mainLayout ); 
}


void m4dGUIScreenLayoutWidget::accept ()
{ 
  screenLayoutDialog->close();
}


void m4dGUIScreenLayoutWidget::reject ()
{ 
  screenLayoutDialog->close();
}


/**
 * Creates Series groupBox - with buttons for various layouts and custom settings.
 */
QGroupBox *m4dGUIScreenLayoutWidget::createSeriesGroupBox ()
{
  QGroupBox *seriesGroupBox = new QGroupBox( tr( "Series" ) );

  QGridLayout *seriesGroupBoxLayout = new QGridLayout;

  seriesLayoutToolButtons = new QToolButton *[LAYOUT_NUMBER];

  unsigned layoutIdx = 0;
  for ( ; layoutIdx < LAYOUT_NUMBER; layoutIdx++ ) 
  {
    QString fileName = QString( ":/icons/" ).append( layoutIconNames[layoutIdx] );
    seriesLayoutToolButtons[layoutIdx] = createToolButton( QIcon( fileName ) );
    seriesGroupBoxLayout->addWidget( seriesLayoutToolButtons[layoutIdx], 
                                     layoutIdx / LAYOUTS_IN_ROW, layoutIdx % LAYOUTS_IN_ROW );
  }

  if ( LAYOUT_NUMBER > 0 ) {
    seriesLayoutToolButtons[0]->setChecked( true );  
  }

  // creating Custom groupBox - within Series groupBox
  // --------------------------------------------------------------------------
  QGroupBox *customSeriesGroupBox = new QGroupBox( tr( "Custom" ) );

  QGridLayout *customSeriesGroupBoxLayout = new QGridLayout;
  customSeriesGroupBoxLayout->setContentsMargins( 30, 10, 30, 10 );

  customSeriesGroupBoxLayout->addWidget( new QLabel( tr( "Rows:" ) ), 0, 0 );
  customSeriesGroupBoxLayout->addWidget( new QLabel( tr( "Columns:" ) ), 0, 1 );
  seriesRowSpinBox = createSpinBox( 1 );
  customSeriesGroupBoxLayout->addWidget( seriesRowSpinBox, 1, 0 );
  seriesColumnSpinBox = createSpinBox( 2 );
  customSeriesGroupBoxLayout->addWidget( seriesColumnSpinBox, 1, 1 );

  customSeriesGroupBox->setLayout( customSeriesGroupBoxLayout );
  // --------------------------------------------------------------------------

  seriesGroupBoxLayout->addWidget( customSeriesGroupBox, layoutIdx / LAYOUTS_IN_ROW, 
                                   0, 1, LAYOUTS_IN_ROW );

  seriesGroupBox->setLayout( seriesGroupBoxLayout );

  return seriesGroupBox;
}


/**
 * Creates Image groupBox - with buttons for various layouts and custom settings.
 */
QGroupBox *m4dGUIScreenLayoutWidget::createImageGroupBox ()
{
  QGroupBox *imageGroupBox  = new QGroupBox( tr( "Image" ) );

  QGridLayout *imageGroupBoxLayout = new QGridLayout;

  imageLayoutToolButtons = new QToolButton *[LAYOUT_NUMBER];

  unsigned layoutIdx = 0;
  for ( ; layoutIdx < LAYOUT_NUMBER; layoutIdx++ ) 
  {
    QString fileName = QString( ":/icons/" ).append( layoutIconNames[layoutIdx] );
    imageLayoutToolButtons[layoutIdx] = createToolButton( QIcon( fileName ) );
    imageGroupBoxLayout->addWidget( imageLayoutToolButtons[layoutIdx], 
                                    layoutIdx / LAYOUTS_IN_ROW, layoutIdx % LAYOUTS_IN_ROW );
  }

  if ( LAYOUT_NUMBER > 0 ) {
    imageLayoutToolButtons[0]->setChecked( true );  
  }

  // creating Custom groupBox - within Image groupBox
  // --------------------------------------------------------------------------
  QGroupBox *customImageGroupBox = new QGroupBox( tr( "Custom" ) );

  QGridLayout *customImageGroupBoxLayout = new QGridLayout;
  customImageGroupBoxLayout->setContentsMargins( 30, 10, 30, 10 );

  customImageGroupBoxLayout->addWidget( new QLabel( tr( "Rows:" ) ), 0, 0 );
  customImageGroupBoxLayout->addWidget( new QLabel( tr( "Columns:" ) ), 0, 1 );
  imageRowSpinBox = createSpinBox( 1 );
  customImageGroupBoxLayout->addWidget( imageRowSpinBox, 1, 0 );
  imageColumnSpinBox = createSpinBox( 1 );
  customImageGroupBoxLayout->addWidget( imageColumnSpinBox, 1, 1 );

  customImageGroupBox->setLayout( customImageGroupBoxLayout );
  // --------------------------------------------------------------------------

  imageGroupBoxLayout->addWidget( customImageGroupBox, layoutIdx / LAYOUTS_IN_ROW, 
                                  0, 1, LAYOUTS_IN_ROW );

  imageGroupBox->setLayout( imageGroupBoxLayout );

  return imageGroupBox;
}


QToolButton *m4dGUIScreenLayoutWidget::createToolButton ( const QIcon &icon )
{
  QToolButton *toolButton = new QToolButton();
  toolButton->setCheckable( true );
  toolButton->setAutoExclusive( true );

  toolButton->setFixedWidth( 52 );
  toolButton->setFixedHeight( 42 );
  toolButton->setIconSize( QSize( 34, 34 ) );
  toolButton->setIcon( icon );

  return toolButton;
}


QSpinBox *m4dGUIScreenLayoutWidget::createSpinBox ( const int value )
{
  QSpinBox *spinBox = new QSpinBox();
  spinBox->setMinimum( 1 );
  spinBox->setValue( value );

  return spinBox;
}

