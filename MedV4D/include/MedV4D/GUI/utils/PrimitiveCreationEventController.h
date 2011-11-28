#ifndef PRIMITIVE_CREATION_EVENT_CONTROLLER_H
#define PRIMITIVE_CREATION_EVENT_CONTROLLER_H

#include "MedV4D/GUI/widgets/GeneralViewer.h"

class APrimitiveCreationEventController: public M4D::GUI::Viewer::AViewerController
{
	Q_OBJECT;
public:
	typedef boost::shared_ptr< APrimitiveCreationEventController > Ptr;
	
	Qt::MouseButton	mVectorEditorInteractionButton;

	APrimitiveCreationEventController(): mVectorEditorInteractionButton( Qt::LeftButton ), mValid( false )
	{}

	virtual void
	cancel()
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

	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool	
	mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) { return false; }

	bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event ) { return false; }
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
		mValid = false;
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
class BaseTemplatedPrimitiveCreationEventController: public APrimitiveCreationEventController
{
public:
	TPrimitive &
	getPrimitive()
	{
		return *mPrimitive;
	}

	void
	cancel()
	{
		if( mValid && mPrimitive ) {
			cancelPrimitive(mPrimitive);
		}
	}
protected:
	BaseTemplatedPrimitiveCreationEventController(): mCurrentStage( 0 ) {}

	virtual TPrimitive *
	createPrimitive( const TPrimitive & aPrimitive )
	{
		return new TPrimitive( aPrimitive );
	}

	virtual void
	primitiveFinished( TPrimitive *aPrimitive )
	{
		delete aPrimitive;
	}

	virtual void
	disposePrimitive( TPrimitive *aPrimitive )
	{
		delete aPrimitive;
	}
	//*********************************************************
	TPrimitive *
	beginPrimitive( const TPrimitive & aPrimitive )
	{
		TPrimitive * primitive = createPrimitive( aPrimitive );
		beginCreation();
		return primitive;
	}

	void
	endPrimitive( TPrimitive *aPrimitive )
	{
		ASSERT( aPrimitive );
		endCreation();
		primitiveFinished( aPrimitive );
		mCurrentStage = 0;
	}

	void
	cancelPrimitive( TPrimitive *aPrimitive )
	{
		ASSERT( aPrimitive );
		cancelCreation();		//TODO check order
		disposePrimitive( aPrimitive );
		mCurrentStage = 0;
	}

	TPrimitive *mPrimitive;
	size_t mCurrentStage;
};

template< typename TPrimitive >
class TemplatedPrimitiveCreationEventController;

template<>
class TemplatedPrimitiveCreationEventController< M4D::Point3Df >: public BaseTemplatedPrimitiveCreationEventController< M4D::Point3Df >
{
public:
	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( (aEventInfo.event->buttons() == Qt::NoButton) && (state.viewType == M4D::GUI::Viewer::vt3D) && (mCurrentStage > 0) ) 
		{
			float t1, t2;
			if ( closestPointsOnTwoLines( mPoint, mDirection, aEventInfo.point, aEventInfo.direction, t1, t2 ) ) {
				Vector3f point = mPoint + (t1 * mDirection);
				*mPrimitive = M4D::Point3Df( point );
			}

			//mPrimitive->secondPoint() = aEventInfo.realCoordinates;
			state.viewerWindow->update();
			return true;
		}
		return false; 
	}

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) 
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( aEventInfo.event->button() == mVectorEditorInteractionButton && state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices ) {
			
			mPrimitive = this->beginPrimitive( M4D::Point3Df( aEventInfo.realCoordinates ) );
			if ( mPrimitive == NULL ) {
				return true;
			}
			endPrimitive( mPrimitive );

			state.viewerWindow->update();
			return true;
		} 

		if ( aEventInfo.event->button() == mVectorEditorInteractionButton && state.viewType == M4D::GUI::Viewer::vt3D ) {
			
			LOG( "ADDING PRIMITIVE IN 3D" );
			if ( mCurrentStage == 0 ) {
				mPoint = aEventInfo.point;
				mDirection = aEventInfo.direction;
				mPrimitive = this->beginPrimitive( M4D::Point3Df( aEventInfo.point ) );
				if ( mPrimitive == NULL ) {
					return true;
				}
				//endPrimitive( mPrimitive );
				++mCurrentStage;
			} else {
				//mPrimitive->secondPoint() = aEventInfo.realCoordinates;
				this->endPrimitive( mPrimitive );
			}
			state.viewerWindow->update();
			return true;
		}
		return false;
	}

	Vector3f mPoint;
	Vector3f mDirection;
};

template<>
class TemplatedPrimitiveCreationEventController< M4D::Line3Df >: public BaseTemplatedPrimitiveCreationEventController< M4D::Line3Df >
{
public:
	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices 
			&& mCurrentStage > 0 ) 
		{
			mPrimitive->secondPoint() = aEventInfo.realCoordinates;
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
				mPrimitive = this->beginPrimitive( M4D::Line3Df( aEventInfo.realCoordinates, aEventInfo.realCoordinates ) );
				if ( mPrimitive == NULL ) {
					return true;
				}
				++mCurrentStage;
			} else {
				mPrimitive->secondPoint() = aEventInfo.realCoordinates;
				this->endPrimitive( mPrimitive );
			}
			state.viewerWindow->update();
			return true;
		} 
		if ( aEventInfo.event->button() == Qt::RightButton && mCurrentStage > 0 ) {
			this->cancelPrimitive( mPrimitive );
			return true;
		}
		return false;
	}
protected:
};

template<>
class TemplatedPrimitiveCreationEventController< M4D::Sphere3Df >: public BaseTemplatedPrimitiveCreationEventController< M4D::Sphere3Df >
{
public:
	bool
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
	{ 
		M4D::GUI::Viewer::ViewerState &state = *(boost::polymorphic_downcast< M4D::GUI::Viewer::ViewerState *>( aViewerState.get() ) );
		if ( state.viewType == M4D::GUI::Viewer::vt2DAlignedSlices 
			&& mCurrentStage > 0 ) 
		{
			mPrimitive->radius() = VectorSize(mPrimitive->center() - aEventInfo.realCoordinates);
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
				mPrimitive = this->beginPrimitive( M4D::Sphere3Df( aEventInfo.realCoordinates, 0.0f ) );
				if ( mPrimitive == NULL ) {
					return true;
				}
				++mCurrentStage;
			} else {
				mPrimitive->radius() = VectorSize(mPrimitive->center() - aEventInfo.realCoordinates);
				this->endPrimitive( mPrimitive );
				mCurrentStage = 0;
			}
			state.viewerWindow->update();
			return true;
		} 
		if ( aEventInfo.event->button() == Qt::RightButton && mCurrentStage > 0 ) {
			this->cancelPrimitive( mPrimitive );
			return true;
		}
		return false;
	}
};


#endif /*PRIMITIVE_CREATION_EVENT_CONTROLLER_H*/
