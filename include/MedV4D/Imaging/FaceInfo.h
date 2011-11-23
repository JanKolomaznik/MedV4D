#ifndef _FACE_INFO_H
#define _FACE_INFO_H

#include "MedV4D/Common/Vector.h"
#include "MedV4D/Common/Color.h"
#include "MedV4D/Imaging/VertexInfo.h"


namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file FaceInfo.h 
 * @{ 
 **/

namespace Imaging
{
namespace Geometry
{


struct TriangleInfoPart
{
	Vector< unsigned, 3 >	indices;
};

template<
	typename FaceInfoPart,
	typename NormalPart = NormalDummy< float32>, 
	typename ColorPart = ColorDummy 
>
struct FaceInfo: public FaceInfoPart, public NormalPart, public ColorPart
{

};

typedef FaceInfo< TriangleInfoPart, NormalDummy<float32>, ColorDummy > SimpleTriangle;

}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/


#endif /*_FACE_INFO_H*/

