/**
 * @ingroup TDA 
 * @author Milan Lepik
 * @file TDASliceViewerSpecialStateOperator.h 
 * @{ 
 **/

#ifndef _TDA_SLICE_VIEWER_SPECIAL_STATE_OPERATOR_H
#define _TDA_SLICE_VIEWER_SPECIAL_STATE_OPERATOR_H

class TDASliceViewerSpecialStateOperator
{
public:
	TDASliceViewerSpecialStateOperator(){};
/*
	 void
	Draw( SliceViewer & viewer, int sliceNum, double zoomRate );

	 void 
	ButtonMethodRight( int amountH, int amountV, double zoomRate );
	
	 void 
	ButtonMethodLeft( int amountH, int amountV, double zoomRate );
	
	 void 
	SelectMethodRight( double x, double y, int sliceNum, double zoomRate );
	*/
	 void 
	SelectMethodLeft( double x, double y, int sliceNum, double zoomRate );
};


#endif

/** @} */