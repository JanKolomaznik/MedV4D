#ifndef ANALYSE_RESULTS_H
#define ANALYSE_RESULTS_H

#include "TypeDeclarations.h"
#include <QtCore>

struct AnalysisRecord
{
	float32 organVolume;
};

Q_DECLARE_METATYPE( AnalysisRecord );

void
AnalyseResults( const InputImageType &image, const M4D::Imaging::Mask3D &mask, AnalysisRecord &record );


#endif /*ANALYSE_RESULTS_H*/
