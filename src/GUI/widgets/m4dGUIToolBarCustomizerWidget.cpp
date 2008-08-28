/**
 *  @ingroup gui
 *  @file m4dGUIToolBarCustomizerWidget.cpp
 *  @brief some brief
 */
#include "GUI/m4dGUIToolBarCustomizerWidget.h"

#include <QtGui>


// Q_INIT_RESOURCE macro cannot be used in a namespace
inline void initToolBarCustomizerWidgetResource () { Q_INIT_RESOURCE( m4dGUIToolBarCustomizerWidget ); }

namespace M4D {
namespace GUI {

/// Prefixes for actions visibility (also in the toolTip for visibility icon)
#define SHOWN_ACTION_PREFIX          "Shown"
#define HIDDEN_ACTION_PREFIX         "Hidden"
/// Name of the group of actions - in QSettings - saving/loading settings (all actions - their groups - are in this group)
#define ACTIONS_GROUP_NAME           "Actions"
/// Names of the keys in QSettings - to identify settings values for each action; action ~ group
#define VISIBILITY_KEY_NAME          "Visibility"
#define SHORTCUT_KEY_NAME            "Shortcut"

m4dGUIToolBarCustomizerWidget::m4dGUIToolBarCustomizerWidget ( QAction **actions, unsigned actionsNum, 
                                                               QWidget *parent )
  : QWidget( parent ), actions( actions ), actionsNum( actionsNum ) 
{
  initToolBarCustomizerWidgetResource();

  // loads previously saved actions settings (from QSettings)
  loadActions();

  toolBarButtonsTable = createToolBarButtonsTable();

  connect( toolBarButtonsTable, SIGNAL(cellClicked( int, int )), 
           this, SLOT(changeVisibility( int, int )) );
  connect( toolBarButtonsTable, SIGNAL(currentItemChanged( QTableWidgetItem *, QTableWidgetItem * ) ), 
           this, SLOT(recordActionShortcut( QTableWidgetItem * )) );
  connect( toolBarButtonsTable, SIGNAL(itemChanged( QTableWidgetItem * )), 
           this, SLOT(validateActionShortcut( QTableWidgetItem * )) );
  
  QPushButton *okButton     = new QPushButton( tr( "&OK" ), this );
  QPushButton *cancelButton = new QPushButton( tr( "&Cancel" ), this );

  connect( okButton,     SIGNAL(clicked()), this, SLOT(accept()) );
  connect( cancelButton, SIGNAL(clicked()), this, SLOT(reject()) );

  QHBoxLayout *buttonLayout = new QHBoxLayout;
  buttonLayout->setSpacing( 8 );
  buttonLayout->addStretch( 1 );
  buttonLayout->addWidget( okButton );
  buttonLayout->addWidget( cancelButton );

  QVBoxLayout *mainLayout = new QVBoxLayout( this );
  mainLayout->setMargin( 8 );
  mainLayout->setSpacing( 8 );
  mainLayout->addWidget( toolBarButtonsTable );
  mainLayout->addLayout( buttonLayout );
}


void m4dGUIToolBarCustomizerWidget::changeVisibility ( int row, int column )
{
  if ( column != 1 ) {
    return;
  }

  QTableWidgetItem *visibilityItem = toolBarButtonsTable->item( row, 1 );

  if  ( visibilityItem->toolTip() == SHOWN_ACTION_PREFIX )
  {
    visibilityItem->setIcon( QIcon( ":/icons/hidden.png" ) );
    visibilityItem->setToolTip( HIDDEN_ACTION_PREFIX );
  }
  else
  {
    visibilityItem->setIcon( QIcon( ":/icons/shown.png" ) );
    visibilityItem->setToolTip( SHOWN_ACTION_PREFIX );
  }
}


void m4dGUIToolBarCustomizerWidget::recordActionShortcut ( QTableWidgetItem *item )
{
  oldAccelText = item->text();
}


void m4dGUIToolBarCustomizerWidget::validateActionShortcut ( QTableWidgetItem *item )
{
  QString accelText = QString( QKeySequence( item->text() ) );

  if ( accelText.isEmpty() && !item->text().isEmpty() ) {
    item->setText( oldAccelText );
  }
  else {
    item->setText( accelText );
  }
}


void m4dGUIToolBarCustomizerWidget::accept ()
{
  for ( unsigned row = 1; row < actionsNum; row++ ) 
  {
    if ( toolBarButtonsTable->item( row - 1, 1 )->toolTip() == SHOWN_ACTION_PREFIX ) {
      actions[row]->setVisible( true );
    }
    else {
      actions[row]->setVisible( false );
    }

    actions[row]->setShortcut( QKeySequence( toolBarButtonsTable->item( row - 1, 2 )->text() ) );
  }

  saveActions();

  emit ready();
}


void m4dGUIToolBarCustomizerWidget::reject ()
{
  for ( unsigned row = 1; row < actionsNum; row++ )
  {
    QTableWidgetItem *visibilityItem = toolBarButtonsTable->item( row - 1, 1 );
    if ( actions[row]->isVisible() ) 
    {
      visibilityItem->setIcon( QIcon( ":/icons/shown.png" ) );
      visibilityItem->setToolTip( SHOWN_ACTION_PREFIX );
    }
    else
    {
      visibilityItem->setIcon( QIcon( ":/icons/hidden.png" ) );   
      visibilityItem->setToolTip( HIDDEN_ACTION_PREFIX );
    }

    QTableWidgetItem *shortcutItem = toolBarButtonsTable->item( row - 1, 2 );
    shortcutItem->setText( QString( actions[row]->shortcut() ) );
  }

  emit cancel();
}


QTableWidget *m4dGUIToolBarCustomizerWidget::createToolBarButtonsTable ()
{
  if ( !actions ) {
    return new QTableWidget;
  }

  QTableWidget *actionsTable = new QTableWidget( actionsNum - 1, 3, this );

  QStringList labels;
  labels << tr( "Description" ) << tr( "" ) << tr( "Shortcut" );
  actionsTable->setHorizontalHeaderLabels( labels );
  actionsTable->horizontalHeader()->setResizeMode( 0, QHeaderView::ResizeToContents );
  actionsTable->horizontalHeader()->setResizeMode( 1, QHeaderView::ResizeToContents );
  actionsTable->horizontalHeader()->setStretchLastSection( true );
  actionsTable->verticalHeader()->hide();

  // from second - first is the empty action for all unplugged slots (it's not in toolBar)
  for ( unsigned row = 1; row < actionsNum; row++ )
  {
    QTableWidgetItem *actionItem = new QTableWidgetItem( actions[row]->text() );
    actionItem->setFlags( actionItem->flags() & ~Qt::ItemIsEditable );
    actionItem->setIcon( actions[row]->icon() );
    actionsTable->setItem( row - 1, 0, actionItem );

    QTableWidgetItem *visibilityItem = new QTableWidgetItem;
    if ( actions[row]->isVisible() ) 
    {
      visibilityItem->setIcon( QIcon( ":/icons/shown.png" ) );
      visibilityItem->setToolTip( SHOWN_ACTION_PREFIX );
    }
    else
    {
      visibilityItem->setIcon( QIcon( ":/icons/hidden.png" ) );   
      visibilityItem->setToolTip( HIDDEN_ACTION_PREFIX );
    }
    visibilityItem->setFlags( visibilityItem->flags() & ~Qt::ItemIsEditable );
    actionsTable->setItem( row - 1, 1, visibilityItem );
    
    QTableWidgetItem *shortcutItem = new QTableWidgetItem( QString( actions[row]->shortcut() ) );
    actionsTable->setItem( row - 1, 2, shortcutItem );
  }

  return actionsTable;
}


void m4dGUIToolBarCustomizerWidget::loadActions ()
{
  QSettings settings;
  settings.beginGroup( ACTIONS_GROUP_NAME );
    
  for ( unsigned row = 1; row < actionsNum; row++ )
  {
    // each action is a different group - within the actions group
    settings.beginGroup( actions[row]->text() );

    QVariant visibility = settings.value( VISIBILITY_KEY_NAME, "" );
    if ( visibility != "" ) {
      actions[row]->setVisible( visibility.toBool() );
    }

    QString accelText = settings.value( SHORTCUT_KEY_NAME ).toString();
    if ( !accelText.isEmpty() ) {
      actions[row]->setShortcut( QKeySequence( accelText ) );
    }

    settings.endGroup();
  }

  settings.endGroup();
}


void m4dGUIToolBarCustomizerWidget::saveActions ()
{
  QSettings settings;
  settings.beginGroup( ACTIONS_GROUP_NAME );
     
  for ( unsigned row = 1; row < actionsNum; row++ )
  {
    // each action is a different group - within the actions group
    settings.beginGroup( actions[row]->text() );

    settings.setValue( VISIBILITY_KEY_NAME, actions[row]->isVisible() );

    settings.setValue( SHORTCUT_KEY_NAME, QString( actions[row]->shortcut() ) );

    settings.endGroup();
  }

  settings.endGroup();
}

} // namespace GUI
} // namespace M4D

