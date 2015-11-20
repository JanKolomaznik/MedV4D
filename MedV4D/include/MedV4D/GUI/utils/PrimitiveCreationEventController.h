#ifndef PRIMITIVE_CREATION_EVENT_CONTROLLER_H
#define PRIMITIVE_CREATION_EVENT_CONTROLLER_H

//Temporary workaround
#ifndef Q_MOC_RUN
#include "MedV4D/GUI/widgets/GeneralViewer.h"
#include "MedV4D/Common/GeometricAlgorithms.h"
#include "MedV4D/Common/GeometricPrimitives.h"
#include "MedV4D/Common/Sphere.h"
#endif //Q_MOC_RUN

class APrimitiveCreationEventController: public M4D::GUI::Viewer::AViewerController
{
	Q_OBJECT;
public:
	typedef std::shared_ptr< APrimitiveCreationEventController > Ptr;

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
	mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) override { return false; }

	bool
	mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) override { return false; }

	bool
	mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) override { return false; }

	bool
	mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) override { return false; }

	bool
	wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo ) override { return false; }
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
			if ( M4D::closestPointsOnTwoLines( mPoint, mDirection, fromGLM(aEventInfo.point), fromGLM(aEventInfo.direction), t1, t2 ) ) {
				Vector3f point = mPoint + (t1 * mDirection);
				*mPrimitive = M4D::Point3Df( point );
			}

			//mPrimitive->secondPoint() = aEventInfo.point;
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

			mPrimitive = this->beginPrimitive( M4D::Point3Df( fromGLM(aEventInfo.point) ) );
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
				D_PRINT("aEventInfo.point " << aEventInfo.point);
				D_PRINT("aEventInfo.direction " << aEventInfo.direction);
				mPoint = fromGLM(aEventInfo.point);
				mDirection = fromGLM(aEventInfo.direction);
				mPrimitive = this->beginPrimitive( M4D::Point3Df( fromGLM(aEventInfo.point) ) );
				if ( mPrimitive == NULL ) {
					return true;
				}
				//endPrimitive( mPrimitive );
				++mCurrentStage;
			} else {
				//mPrimitive->secondPoint() = aEventInfo.point;
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
			mPrimitive->secondPoint() = fromGLM(aEventInfo.point);
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
				mPrimitive = this->beginPrimitive( M4D::Line3Df( fromGLM(aEventInfo.point), fromGLM(aEventInfo.point) ) );
				if ( mPrimitive == NULL ) {
					return true;
				}
				++mCurrentStage;
			} else {
				mPrimitive->secondPoint() = fromGLM(aEventInfo.point);
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
			mPrimitive->radius() = VectorSize(mPrimitive->center() - fromGLM(aEventInfo.point));
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
				mPrimitive = this->beginPrimitive( M4D::Sphere3Df( fromGLM(aEventInfo.point), 0.0f ) );
				if ( mPrimitive == NULL ) {
					return true;
				}
				++mCurrentStage;
			} else {
				mPrimitive->radius() = VectorSize(mPrimitive->center() - fromGLM(aEventInfo.point));
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
