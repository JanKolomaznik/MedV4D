#include "ShoulderMeasurementModule/ShoulderMeasurementController.hpp"
#include <algorithm>
#include <numeric>


#include <eigen2/Eigen/Dense>

class PointGroupPrimitiveController: public TemplatedPrimitiveCreationEventController< M4D::Point3Df >
{
public:
	PointGroupPrimitiveController( PointSet &aPrimitives, size_t aMaxPointCount ): mPrimitives( aPrimitives ), mMaxPointCount( aMaxPointCount ) {}

protected:

	virtual M4D::Point3Df *
	createPrimitive( const M4D::Point3Df & aPrimitive )
	{
		if ( mPrimitives.size() == mMaxPointCount ) {
			return NULL;
		}
		mPrimitives.push_back( aPrimitive );
		return &(mPrimitives[mPrimitives.size()-1]);
	}

	virtual void
	primitiveFinished( M4D::Point3Df *aPrimitive )
	{
	}

	virtual void
	disposePrimitive( M4D::Point3Df *aPrimitive )
	{
		mPrimitives.resize( mPrimitives.size()-1 );
	}

	PointSet &mPrimitives;
	size_t mMaxPointCount;
};



ShoulderMeasurementController::ShoulderMeasurementController()
{
	mMeasurementHandlers[mmHUMERAL_HEAD] = APrimitiveCreationEventController::Ptr( new PointGroupPrimitiveController(mHumeralHeadPoints, 6) );

}

ShoulderMeasurementController::~ShoulderMeasurementController()
{

}

bool
ShoulderMeasurementController::mouseMoveEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mMeasurementMode != mmNONE ) {
		if ( mMeasurementHandlers[ mMeasurementMode ]->mouseMoveEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
	}
	return ControllerPredecessor::mouseMoveEvent ( aViewerState, aEventInfo );
}
bool	
ShoulderMeasurementController::mouseDoubleClickEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mMeasurementMode != mmNONE ) {
		if ( mMeasurementHandlers[ mMeasurementMode ]->mouseDoubleClickEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
		return true; //TODO - prevent plane changing
	}
	return ControllerPredecessor::mouseDoubleClickEvent ( aViewerState, aEventInfo );
}

bool
ShoulderMeasurementController::mousePressEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mMeasurementMode != mmNONE ) {
		if ( mMeasurementHandlers[ mMeasurementMode ]->mousePressEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
	}
	return ControllerPredecessor::mousePressEvent ( aViewerState, aEventInfo );
}

bool
ShoulderMeasurementController::mouseReleaseEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, const M4D::GUI::Viewer::MouseEventInfo &aEventInfo )
{
	if ( mMeasurementMode != mmNONE ) {
		if ( mMeasurementHandlers[ mMeasurementMode ]->mouseReleaseEvent( aViewerState, aEventInfo ) ) {
			return true;
		}
	}
	return ControllerPredecessor::mouseReleaseEvent ( aViewerState, aEventInfo );
}

bool
ShoulderMeasurementController::wheelEvent ( M4D::GUI::Viewer::BaseViewerState::Ptr aViewerState, QWheelEvent * event )
{
	if ( mMeasurementMode != mmNONE ) {
		if ( mMeasurementHandlers[ mMeasurementMode ]->wheelEvent( aViewerState, event ) ) {
			return true;
		}
	}
	return ControllerPredecessor::wheelEvent ( aViewerState, event );
}

unsigned
ShoulderMeasurementController::getAvailableViewTypes()const
{

}

void
ShoulderMeasurementController::render2DAlignedSlices( int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane )
{
	GL_CHECKED_CALL( glPushAttrib( GL_ALL_ATTRIB_BITS ) );

	GL_CHECKED_CALL( glEnable( GL_POINT_SMOOTH ) );
	GL_CHECKED_CALL( glEnable( GL_BLEND ) );
	GL_CHECKED_CALL( glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ) );
	GL_CHECKED_CALL( glPointSize( 4.0f ) );

	GL_CHECKED_CALL( glDisable( GL_DEPTH_TEST ) );
	
	GL_CHECKED_CALL( glColor4f( 1.0f, 0.0f, 0.0f, 1.0f ) );

	drawPointSet2D( mHumeralHeadPoints.begin(), mHumeralHeadPoints.end(), aInterval, aPlane );

	GL_CHECKED_CALL( glPopAttrib() );
}

void
ShoulderMeasurementController::preRender3D()
{
	if( !mOverlay ) {
		render3D();
	}
}

void
ShoulderMeasurementController::postRender3D()
{
	if( mOverlay ) {
		render3D();
	}
}

void
ShoulderMeasurementController::render3D()
{
	GL_CHECKED_CALL( glPushAttrib( GL_ALL_ATTRIB_BITS ) );

	GL_CHECKED_CALL( glEnable( GL_POINT_SMOOTH ) );
	GL_CHECKED_CALL( glEnable( GL_BLEND ) );
	GL_CHECKED_CALL( glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ) );
	GL_CHECKED_CALL( glPointSize( 4.0f ) );

	GL_CHECKED_CALL( glDisable( GL_LIGHTING ) );
	GL_CHECKED_CALL( glColor4f( 1.0f, 0.0f, 0.0f, 1.0f ) );

	M4D::drawPointSet( mHumeralHeadPoints.begin(), mHumeralHeadPoints.end() );

	GL_CHECKED_CALL( glColor4f( 0.0f, 1.0f, 0.0f, 1.0f ) );
	M4D::drawGrid( mHeadMeasurementData.point, mHeadMeasurementData.vDirection, mHeadMeasurementData.wDirection, 80.0f, 80.0f, 5.0f );

	M4D::drawStippledLine( mHeadMeasurementData.point + mHeadMeasurementData.normal * 70.0f, mHeadMeasurementData.point - mHeadMeasurementData.normal * 70.0f );

	GL_CHECKED_CALL( glPopAttrib() );
}

void
computeCovarianceMatrixFromPointSet( const PointSet &aPoints, Vector3f &aCenter, Eigen::Matrix3f &aCovarianceMatrix )
{
	float scale = 1.0f/aPoints.size();
	aCenter = scale * std::accumulate( aPoints.begin(), aPoints.end(), Vector3f() );

/*	//compute variances
	Vector3f variances;
	for ( size_t k = 0; k < aPoints.size(); ++k ) {
		variances += VectorCoordinateProduct( aPoints[k]-aCenter, aPoints[k]-aCenter );
	}
	variances *= scale;*/

	for ( size_t i = 0; i < 3; ++i ) {
		for ( size_t j = 0; j < 3; ++j ) {
			float covariance = 0.0f;
			for ( size_t k = 0; k < aPoints.size(); ++k ) {
				covariance += (aPoints[k][i]-aCenter[i]) * (aPoints[k][j]-aCenter[j]);
			}
			aCovarianceMatrix(i,j) = scale * covariance;
		}
	}
}

void
ShoulderMeasurementController::analyseHumeralHead()
{
	if ( mHumeralHeadPoints.size() < 3 ) {
		return;
	}

	Vector3f center;
	Eigen::Matrix3f covarianceMatrix;

	computeCovarianceMatrixFromPointSet( mHumeralHeadPoints, center, covarianceMatrix );

	typedef Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> Solver;
	Solver eigensolver(covarianceMatrix);

	Solver::RealVectorType eigenVals = eigensolver.eigenvalues();
	Eigen::Matrix3f eigenVectors = eigensolver.eigenvectors();
	D_PRINT( "Eigen values :\n" << eigenVals );
	D_PRINT( "Eigen vectors :\n" << eigenVectors );


	Vector3f v1( eigenVectors(0,2), eigenVectors(1,2), eigenVectors(2,2) );
	Vector3f v2( eigenVectors(0,1), eigenVectors(1,1), eigenVectors(2,1) );
	
/*	Vector3f v1 = mHumeralHeadPoints[0] - center;
	Vector3f v2 = mHumeralHeadPoints[1] - center;
	VectorNormalization( v1 );
	VectorNormalization( v2 );*/
	Vector3f normal = VectorProduct( v1, v2 );
	VectorNormalization( normal );
	/*v2 = VectorProduct( v1, normal );
	VectorNormalization( v2 );*/

	mHeadMeasurementData.point = center;
	mHeadMeasurementData.normal = normal;
	mHeadMeasurementData.vDirection = v1;
	mHeadMeasurementData.wDirection = v2;
	mHeadMeasurementData.available = true;

	emit updateRequest();
}
