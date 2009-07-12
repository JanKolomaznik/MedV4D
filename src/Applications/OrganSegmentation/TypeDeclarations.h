#ifndef TYPE_DECLARATIONS_H
#define TYPE_DECLARATIONS_H

#include <string>
#include <vector>
#include "Imaging/Imaging.h"

typedef M4D::Imaging::AbstractImage::Ptr		AbstractImagePtr;
typedef M4D::Imaging::Image< int16, 3 >			InputImageType;
typedef M4D::Imaging::Geometry::BSpline<float32,2>	CurveType;
typedef M4D::Imaging::SlicedGeometry< CurveType >	GDataSet;

//typedef InputImageType::Ptr	InputImagePtr;

typedef M4D::Imaging::ConnectionTyped< M4D::Imaging::AbstractImage >	InImageConnection;
typedef M4D::Imaging::ConnectionTyped< InputImageType >			ImageConnectionType;
typedef ImageConnectionType 						*ImageConnectionTypePtr;
typedef M4D::Imaging::ConnectionTyped< GDataSet >			GDatasetConnectionType;
typedef M4D::Imaging::ConnectionTyped< M4D::Imaging::Mask3D >		Mask3DConnectionType;


struct ModelInfo
{
	ModelInfo() {}
	ModelInfo( std::string name, std::string filename ) : modelName( name ), modelFilename( filename )
		{}

	std::string	modelName;
	std::string	modelFilename;
};

typedef std::vector< ModelInfo >	ModelInfoVector;
#endif /*TYPE_DECLARATIONS_H*/
