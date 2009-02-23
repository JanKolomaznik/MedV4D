#ifndef _SETTINGS_BOX_H
#define _SETTINGS_BOX_H

#include <QtGui>
#include "remoteComp/remoteFilterProperties/levelSetRemoteProperties.h"
#include "remoteComp/remoteFilter.h"

typedef int16	ElementType;
typedef M4D::RemoteComputing::LevelSetRemoteProperties< ElementType, ElementType > 
	LevelSetFilterProperties;

typedef M4D::Imaging::Image< ElementType, 3 > ImageType;
typedef M4D::RemoteComputing::RemoteFilter< ImageType, ImageType > 
	LevelSetFilterType;

class SettingsBox : public QWidget
{
	Q_OBJECT
public:
	static const unsigned MINIMUM_WIDTH = 200;
	static const unsigned EXECUTE_BUTTON_SPACING = 80;
	static const unsigned ROW_SPACING = 15;
	

	SettingsBox( LevelSetFilterType *filter, QWidget * parent );
	
	void
	SetEnabledExecButton( bool val )
		{ execButton->setEnabled( val ); }
protected slots:

	void 
	ChangeProjPlane( int val );

	void
	ChangeProjectionType( int val );

	void
	ExecuteFilter();

	void
	EndOfExecution();
protected:
	void
	CreateWidgets();

	LevelSetFilterType *filter_;

	QWidget *_parent;
	
	QPushButton *execButton;
};

#endif /*_SETTINGS_BOX_H*/


