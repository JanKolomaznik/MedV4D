#ifndef TYPE_DECLARATIONS_H
#define TYPE_DECLARATIONS_H

#include <string>
#include <vector>
#include "Imaging/Imaging.h"
#include "MedV4D/Common/Common.h"

typedef M4D::Imaging::AImage::Ptr			AImagePtr;
typedef M4D::Imaging::Image< int16, 3 >			InputImageType;
typedef M4D::Imaging::Geometry::BSpline<float32,2>	CurveType;
typedef M4D::Imaging::SlicedGeometry< CurveType >	GDataset;

//typedef InputImageType::Ptr	InputImagePtr;

typedef M4D::Imaging::ConnectionTyped< M4D::Imaging::AImage >		InImageConnection;
typedef M4D::Imaging::ConnectionTyped< InputImageType >			ImageConnectionType;
typedef ImageConnectionType 						*ImageConnectionTypePtr;
typedef M4D::Imaging::ConnectionTyped< GDataset >			GDatasetConnectionType;
typedef M4D::Imaging::ConnectionTyped< M4D::Imaging::Mask3D >		Mask3DConnectionType;


struct ModelInfo
{
	ModelInfo() {}
	//ModelInfo( std::string name, std::string filename ) : modelName( name ), modelFilename( filename )
	//	{}
	ModelInfo( std::string name, Path filename ) : modelName( name ), modelFilename( filename )
		{}

	std::string	modelName;
	Path		modelFilename;
};

typedef std::vector< ModelInfo >	ModelInfoVector;
#endif /*TYPE_DECLARATIONS_H*/
