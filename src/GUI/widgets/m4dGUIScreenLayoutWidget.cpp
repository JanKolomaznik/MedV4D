#include "GUI/m4dGUIScreenLayoutWidget.h"

#include <QtGui>


/// Number of various layouts (in one groupBox)
#define LAYOUT_NUMBER   9
/// Number of layouts in one row (in one groupBox)
#define LAYOUTS_IN_ROW  3

const char *m4dGUIScreenLayoutWidget::layoutIconNames[] = { "layout-1x1.png", "layout-1x2.png", 
                                                            "layout-2x1.png", "layout-2x2.png",
                                                            "layout-2x4.png", "layout-4x2.png",
                                                            "layout-4x4.png", "layout-4x6.png", 
                                                            "layout-4x8.png" };
const int m4dGUIScreenLayoutWidget::layoutDimensions[][2] = { {1, 1}, {1, 2}, {2, 1}, {2, 2},
                                                              {2, 4}, {4, 2}, {4, 4}, {4, 6},   
                                                              {4, 8} };

m4dGUIScreenLayoutWidget::m4dGUIScreenLayoutWidget ( m4dGUIMainViewerDesktopWidget *mainViewerDesktop,
                                                     QDialog *screenLayoutDialog, QWidget *parent )
  : QWidget( parent ),
    mainViewerDesktop( mainViewerDesktop ), screenLayoutDialog( screenLayoutDialog )
{
  Q_INIT_RESOURCE( m4dGUIScreenLayoutWidget );

  // creating Series groupBox
  QGroupBox *seriesGroupBox = createLayoutGroupBox( tr( "Series" ), &seriesLayoutToolButtons, &seriesRowSpinBox,
                                                    &seriesColumnSpinBox, 1, SLOT(seriesLayoutChanged()),
                                                    SLOT(seriesApply()) );
  // creating Image groupBox
  QGroupBox *imageGroupBox = createLayoutGroupBox( tr( "Image" ), &imageLayoutToolButtons, &imageRowSpinBox,
                                                   &imageColumnSpinBox, 0, SLOT(imageLayoutChanged()),
                                                   SLOT(imageApply()) );
  // creating OK Cancel dialogButtonBox
  QDialogButtonBox *dialogButtonBox = new QDialogButtonBox( QDialogButtonBox::Ok );
  connect( dialogButtonBox, SIGNAL(accepted()), this, SLOT(accept()) );

  // building layout
  QGridLayout *mainLayout = new QGridLayout;
  mainLayout->addWidget( seriesGroupBox, 0, 0 );
  mainLayout->addWidget( imageGroupBox, 0, 1 );
  mainLayout->addItem( new QSpacerItem( 2, 15, QSizePolicy::Minimum, QSizePolicy::Expanding ), 1, 0, 1, 2 );
  mainLayout->addWidget( dialogButtonBox, 2, 0, 1, 2 );
  setLayout( mainLayout ); 
}


void m4dGUIScreenLayoutWidget::seriesLayoutChanged ()
{ 
  unsigned layoutIdx = 0;
  for ( ; layoutIdx < LAYOUT_NUMBER; layoutIdx++ ) 
  {   
    if ( seriesLayoutToolButtons[layoutIdx]->isChecked() ) {
      break;
    }
  }

  seriesRowSpinBox->setValue( layoutDimensions[layoutIdx][0] );
  seriesColumnSpinBox->setValue( layoutDimensions[layoutIdx][1] );

  mainViewerDesktop->setDesktopLayout( layoutDimensions[layoutIdx][0],
                                       layoutDimensions[layoutIdx][1] );
}


void m4dGUIScreenLayoutWidget::imageLayoutChanged ()
{ 
  unsigned layoutIdx = 0;
  for ( ; layoutIdx < LAYOUT_NUMBER; layoutIdx++ ) 
  {   
    if ( imageLayoutToolButtons[layoutIdx]->isChecked() ) {
      break;
    }
  }

  imageRowSpinBox->setValue( layoutDimensions[layoutIdx][0] );
  imageColumnSpinBox->setValue( layoutDimensions[layoutIdx][1] );

  mainViewerDesktop->setDesktopLayout( layoutDimensions[layoutIdx][0],
                                       layoutDimensions[layoutIdx][1] );
}


void m4dGUIScreenLayoutWidget::seriesApply ()
{
  mainViewerDesktop->setDesktopLayout( seriesRowSpinBox->value(), seriesColumnSpinBox->value() );
}


void m4dGUIScreenLayoutWidget::imageApply ()
{ 

}


void m4dGUIScreenLayoutWidget::accept ()
{ 
  screenLayoutDialog->close();
}


QGroupBox *m4dGUIScreenLayoutWidget::createLayoutGroupBox ( const QString &title, QToolButton ***toolButtons,
                                                            QSpinBox **rowSpinBox, QSpinBox **columnSpinBox,
                                                            const unsigned dimensionsIdx, const char *layoutChangedMember,
                                                            const char *applyMember )
{
  QGroupBox *layoutGroupBox = new QGroupBox( title );

  QGridLayout *layoutGroupBoxLayout = new QGridLayout;

  *toolButtons = new QToolButton *[LAYOUT_NUMBER];

  unsigned layoutIdx = 0;
  for ( ; layoutIdx < LAYOUT_NUMBER; layoutIdx++ ) 
  {
    QString fileName = QString( ":/icons/" ).append( layoutIconNames[layoutIdx] );
    (*toolButtons)[layoutIdx] = createToolButton( QIcon( fileName ), layoutChangedMember );
    layoutGroupBoxLayout->addWidget( (*toolButtons)[layoutIdx], 
                                     layoutIdx / LAYOUTS_IN_ROW, layoutIdx % LAYOUTS_IN_ROW );
  }

  if ( LAYOUT_NUMBER > 0 ) {
    (*toolButtons)[dimensionsIdx]->setChecked( true );  
  }

  // creating Custom GroupBox - within given GroupBox
  // --------------------------------------------------------------------------
  QGroupBox *customLayoutGroupBox = new QGroupBox( tr( "Custom" ) );

  QGridLayout *customLayoutGroupBoxLayout = new QGridLayout;
  customLayoutGroupBoxLayout->setContentsMargins( 30, 10, 30, 10 );

  customLayoutGroupBoxLayout->addWidget( new QLabel( tr( "Rows:" ) ), 0, 0 );
  customLayoutGroupBoxLayout->addWidget( new QLabel( tr( "Columns:" ) ), 0, 1 );

  *rowSpinBox = createSpinBox( layoutDimensions[dimensionsIdx][0] );
  customLayoutGroupBoxLayout->addWidget( *rowSpinBox, 1, 0 );
  *columnSpinBox = createSpinBox( layoutDimensions[dimensionsIdx][1] );
  customLayoutGroupBoxLayout->addWidget( *columnSpinBox, 1, 1 );

  QPushButton *applyButton = new QPushButton( tr( "Apply Custom" ) );
  connect( applyButton, SIGNAL(clicked()), this, applyMember );
  customLayoutGroupBoxLayout->addWidget( applyButton, 2, 0, 1, 2 );

  customLayoutGroupBox->setLayout( customLayoutGroupBoxLayout );
  // --------------------------------------------------------------------------

  layoutGroupBoxLayout->addWidget( customLayoutGroupBox, layoutIdx / LAYOUTS_IN_ROW, 
                                   0, 1, LAYOUTS_IN_ROW );

  layoutGroupBox->setLayout( layoutGroupBoxLayout );

  return layoutGroupBox;
}


QToolButton *m4dGUIScreenLayoutWidget::createToolButton ( const QIcon &icon, const char *member )
{
  QToolButton *toolButton = new QToolButton();
  toolButton->setCheckable( true );
  toolButton->setAutoExclusive( true );

  toolButton->setFixedWidth( 52 );
  toolButton->setFixedHeight( 42 );
  toolButton->setIconSize( QSize( 34, 34 ) );
  toolButton->setIcon( icon );

  connect( toolButton, SIGNAL(clicked()), this, member );

  return toolButton;
}


QSpinBox *m4dGUIScreenLayoutWidget::createSpinBox ( const int value )
{
  QSpinBox *spinBox = new QSpinBox();
  spinBox->setMinimum( 1 );
  spinBox->setValue( value );

  return spinBox;
}

