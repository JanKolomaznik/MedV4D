#ifndef SETTINGS_BOX_H
#define SETTINGS_BOX_H

#include <QtGui>

#include "Imaging/filters/MultiscanRegistration.h"


class SettingsBox: public QWidget
{
	Q_OBJECT

  public:

	  static const unsigned MINIMUM_WIDTH = 200;
	  static const unsigned EXECUTE_BUTTON_SPACING = 40;
  
	  SettingsBox ( M4D::Imaging::APipeFilter *filter, QWidget *parent );
  	
	  void SetEnabledExecButton ( bool val ) { execButton->setEnabled( val ); }

  protected slots:

    void InterpolationTypeChanged ( int val );

	  void ExecuteFilter ();

	  void EndOfExecution ();

  protected:

	  void CreateWidgets ();

	  M4D::Imaging::APipeFilter *filter;

    QWidget *parent;

	  QPushButton *execButton;
};


#endif // SETTINGS_BOX_H


