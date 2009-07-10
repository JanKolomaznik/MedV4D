#ifndef NORMALIZED_MUTUAL_INFORMATION_H
#error File NormalizedMutualInformation.tcc cannot be included directly!
#else

namespace M4D
{

namespace Imaging
{

template< typename ElementType >
double
NormalizedMutualInformation< ElementType >
::compute( MultiHistogram< ElementType, 2 >& jointHist )
{
	double H_a = 0, H_b = 0, H_joint = 0;
	int32 i, j;
	double jointFreq, separateFreq[2];

	for ( i = jointHist.GetMin(); i < jointHist.GetMax(); ++i )
	{
		separateFreq[0] = 0;
		separateFreq[1] = 0;
		for ( j = jointHist[i].GetMin(); j < jointHist[i].GetMax(); ++j )
		{
			jointFreq = jointHist[i][j];

			if ( jointFreq > 0 ) H_joint += jointFreq * std::log( jointFreq );

			separateFreq[0] += jointHist[i][j];
			separateFreq[1] += jointHist[j][i];
		}

		if ( separateFreq[0] > 0 ) H_a += separateFreq[0] * std::log( separateFreq[0] );

		if ( separateFreq[1] > 0 ) H_b += separateFreq[1] * std::log( separateFreq[1] );

	}

	return ( H_a + H_b ) / H_joint;
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif
