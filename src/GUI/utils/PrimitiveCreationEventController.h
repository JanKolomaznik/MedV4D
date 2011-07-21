#ifndef PRIMITIVE_CREATION_EVENT_CONTROLLER_H
#define PRIMITIVE_CREATION_EVENT_CONTROLLER_H

#include "GUI/widgets/GeneralViewer.h"

class APrimitiveCreationEventController: M4D::GUI::Viewer::AViewerController
{
	Q_OBJECT;
public:
	typedef boost::shared_ptr< APrimitiveCreationEventController > Ptr;
	
	Qt::MouseButton	mVectorEditorInteractionButton;

	APrimitiveCreationEventController(): mVectorEditorInteractionButton( Qt::LeftButton ), mValid( false )
	{}

	virtual void
	activate()
	{
		mValid = false;
	}

	virtual void
	deactivate()
	{
		if( mValid ) {
			cancelCreation();
		}
	}

	virtual bool
	isValid()const
	{
		return mValid;
	}
signals:
	void
	started();

	void
	finished();

	void
	canceled();
protected:
	virtual void
	beginCreation()
	{
		mValid = true;
		emit started();
	}

	virtual void
	endCreation()
	{
		ASSERT( mValid );
		emit finished();
	}

	virtual void
	cancelCreation()
	{
		mValid = false;
		emit canceled();
	}

	bool mValid;
};

template< typename TPrimitive >
class TemplatedPrimitiveCreationEventController: public APrimitiveCreationEventController
{
public:
	TPrimitive &
	getPrimitive()
	{
		return mPrimitive;
	}
protected:
	TemplatedPrimitiveCreationEventController(): mCurrentStage( 0 ) {}

	TPrimitive mPrimitive;
	size_t mCurrentStage;
};

class PointCreationEventController: public TemplatedPrimitiveCreationEventController< M4D::Point3Df >
{
public:
	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( aEventInfo.event->button() == mVectorEditorInteractionButton && state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices ) {
			
			mPrimitive = M4D::Point3Df( aEventInfo.realCoordinates );
			beginCreation();
			endCreation();

			state.viewerWindow->update();
			return true;
		} 
		return false;
	}
};

class LineCreationEventController: public TemplatedPrimitiveCreationEventController< M4D::Line3Df >
{
public:
	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices 
			&& mCurrentStage > 0 ) 
		{
			mPrimitive.secondPoint() = aEventInfo.realCoordinates;
			state.viewerWindow->update();
			return true;
		}
		return false; 
	}

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( aEventInfo.event->button() == mVectorEditorInteractionButton 
			&& state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices ) 
		{
			if ( mCurrentStage == 0 ) {
				mPrimitive = M4D::Line3Df( aEventInfo.realCoordinates, aEventInfo.realCoordinates );
				beginCreation();
				++mCurrentStage;
			} else {
				mPrimitive.secondPoint() = aEventInfo.realCoordinates;
				endCreation();
				mCurrentStage = 0;
			}
			state.viewerWindow->update();
			return true;
		} 
		if ( aEventInfo.event->button() == Qt::RightButton && mCurrentStage > 0 ) {
			cancelCreation();
			mCurrentStage = 0;
			return true;
		}
		return false;
	}
protected:
};

class SphereCreationEventController: public TemplatedPrimitiveCreationEventController< M4D::Sphere3Df >
{
public:
	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices 
			&& mCurrentStage > 0 ) 
		{
			mPrimitive.radius() = VectorSize(mPrimitive.center() - aEventInfo.realCoordinates);
			state.viewerWindow->update();
			return true;
		}
		return false; 
	}

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( aEventInfo.event->button() == mVectorEditorInteractionButton 
			&& state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices ) 
		{
			if ( mCurrentStage == 0 ) {
				mPrimitive = M4D::Sphere3Df( aEventInfo.realCoordinates, 0.0f );
				beginCreation();
				++mCurrentStage;
			} else {
				mPrimitive.radius() = VectorSize(mPrimitive.center() - aEventInfo.realCoordinates);
				endCreation();
				mCurrentStage = 0;
			}
			state.viewerWindow->update();
			return true;
		} 
		if ( aEventInfo.event->button() == Qt::RightButton && mCurrentStage > 0 ) {
			cancelCreation();
			mCurrentStage = 0;
			return true;
		}
		return false;
	}
};


#endif /*PRIMITIVE_CREATION_EVENT_CONTROLLER_H*/
