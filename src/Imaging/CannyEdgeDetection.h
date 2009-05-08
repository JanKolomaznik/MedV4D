#ifndef CANNY_EDGE_DETECTION_H
#define CANNY_EDGE_DETECTION_H

template< typename ElementType >
void
CannyEdgeDetection( 
		const ImageRegion< ElementType, 2 >	&input,
		ImageRegion< ElementType, 2 >		&output,
		ImageRegion< Vector<float32,2>, 2 >	&gradient
		)
{
	GradientComputationAndQuantization( input, gradient );

	//NonmaximumSuppression(  );

	//Hysteresis( );
}


#endif /*CANNY_EDGE_DETECTION_H*/
