#ifndef SEGMENTATION_MANAGER_H
#define SEGMENTATION_MANAGER_H

#include <QtGui>
#include <list>
#include "TypeDeclarations.h"
#include "Imaging/Imaging.h"
#include "ManagerViewerSpecialState.h"

struct ManagerActivationInfo
{
	ManagerActivationInfo()
		: connection( NULL ), panel( NULL ){}
	ManagerActivationInfo( ImageConnectionTypePtr conn, QWidget *pan, const M4D::Viewer::SliceViewerSpecialStateOperatorPtr &sState )
		: connection( conn ), panel( pan ), specState( sState ) {}

	ImageConnectionTypePtr	connection;
	QWidget			*panel;
	M4D::Viewer::SliceViewerSpecialStateOperatorPtr	specState;
};

struct ResultsInfo
{
	ResultsInfo() {}

	ResultsInfo( const InputImageType::Ptr &im, const GDataSet::Ptr &geom )
		: image( im ), geometry( geom ) {}

	InputImageType::Ptr	image;
	GDataSet::Ptr		geometry;
};

Q_DECLARE_METATYPE( ManagerActivationInfo );
Q_DECLARE_METATYPE( ResultsInfo );

class SegmentationManager: public QObject
{
	Q_OBJECT
public:
	virtual void
	Initialize() = 0;

	virtual void
	Finalize() = 0;

	virtual void
	Activate( InputImageType::Ptr inImage ) = 0;

	virtual std::string
	GetName() = 0;
	
	QWidget *
	GetGUI()
		{ return _controlPanel; }
	
	M4D::Viewer::SliceViewerSpecialStateOperatorPtr
	GetSpecialState()
	{
		return _specialState;
	}
	
	ImageConnectionTypePtr
	GetInputConnection()
		{ return _inConnection; }

	virtual GDataSet::Ptr
	GetOutputGeometry() = 0;

	virtual InputImageType::Ptr
	GetInputImage() = 0;

public slots:

	virtual void
	ActivateManager()
		{ emit ManagerActivated( ManagerActivationInfo( GetInputConnection(), GetGUI(), GetSpecialState() ) ); }

	void
	WantsProcessResults();
signals:

	void
	ManagerActivated( ManagerActivationInfo info );

	void
	ProcessResults( ResultsInfo );

	void
	WantsViewerUpdate();
protected:
	SegmentationManager(): _controlPanel( NULL )
		{ /*empty*/ }
	
	virtual
	~SegmentationManager() 
		{ /*empty*/ }

	InputImageType::Ptr				 	_inputImage;
	bool						_wasInitialized;
	QWidget						*_controlPanel;

	M4D::Viewer::SliceViewerSpecialStateOperatorPtr _specialState;
	ImageConnectionType				*_inConnection;
};

typedef std::list< SegmentationManager* >	SegmentationManagersList;

extern SegmentationManagersList segmentationManagers;

void
InitializeSegmentationManagers();


#endif /*SEGMENTATION_MANAGER_H*/
