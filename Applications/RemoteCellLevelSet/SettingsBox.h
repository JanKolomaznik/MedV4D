#ifndef _SETTINGS_BOX_H
#define _SETTINGS_BOX_H

#include <QtGui>
#include "remoteComp/remoteFilter.h"
#include "remoteComp/remoteServerFilters/levelsetSegmentation/medevedWrapperFilter.h"

typedef float32	ElementType;
typedef M4D::RemoteComputing::ThreshLSSegMedvedWrapper< ElementType, ElementType >::Properties
	LevelSetFilterProperties;

typedef M4D::Imaging::Image< ElementType, 3 > ImageType;
typedef M4D::RemoteComputing::RemoteFilter< ImageType, ImageType > 
	RemoteFilterType;

class SettingsBox : public QWidget
{
	Q_OBJECT
public:
	static const unsigned MINIMUM_WIDTH = 200;
	static const unsigned EXECUTE_BUTTON_SPACING = 80;
	static const unsigned ROW_SPACING = 15;
	

	SettingsBox( RemoteFilterType *filter, LevelSetFilterProperties *props, QWidget * parent,
			const M4D::Imaging::AbstractDataSet &decimDS);
	
	void SetEnabledExecButton( bool val ) { execButton->setEnabled( val ); }
	
	void SetSeed(const Vector<uint32, 3> &size);
	
protected slots:

	// events callbacks
	void lowerThresholdValueChanged( int val ) {props_->lowerThreshold = val;}
	void upperThresholdValueChanged( int val ) {props_->upperThreshold = val;}
	void maxIterationsValueChanged( int val ) {props_->maxIterations = val;}
	void initialDistanceValueChanged( double val ) {props_->initialDistance = val;}
	void seedXValueChanged( int val) {props_->seedX = val;}
	void seedYValueChanged( int val) {props_->seedY = val;}
	void seedZValueChanged( int val) {props_->seedZ = val;}
	void curvatureScalingValueChanged( double val ) {props_->curvatureScaling = val;}
	void propagationScalingValueChanged( double val ) {props_->propagationScaling = val;}
	void advectionScalingValueChanged( double val ) {props_->advectionScaling = val;}

	void ExecuteFilter();

	void EndOfExecution();
protected:
	void
	CreateWidgets();

	RemoteFilterType *filter_;
	LevelSetFilterProperties *props_;

	QWidget *_parent;
	
	QPushButton *execButton;
	
	// levelset filter properties controls
	QSpinBox *lowerThreshold;
	QSpinBox *upperThreshold;
	QSpinBox *maxIterations;
	QDoubleSpinBox *initialDistance;
	QSpinBox *seedX;
	QSpinBox *seedY;
	QSpinBox *seedZ;
	QDoubleSpinBox *curvatureScaling;
	QDoubleSpinBox *propagationScaling;
	QDoubleSpinBox *advectionScaling;
	
	const M4D::Imaging::AbstractDataSet &_decimatedDataset;
};

#endif /*_SETTINGS_BOX_H*/


