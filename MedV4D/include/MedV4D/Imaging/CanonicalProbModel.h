#ifndef CANONICAL_PROB_MODEL_H
#define CANONICAL_PROB_MODEL_H

#include "MedV4D/Common/Common.h"
#include <cmath>
#include "MedV4D/Imaging/Histogram.h"
#include <fstream>
#include <boost/shared_ptr.hpp>


namespace M4D
{
namespace Imaging {

struct Transformation {
        Vector< float32, 3 >
        operator() ( const Vector< float32, 3 > &pos ) const {
                Vector< float32, 3 > result = pos - _origin;
                result[2] *= _zScale;
                result -= result[2] * _diff;
                return result;
        }

        Vector< float32, 3 >
        GetInversion ( const Vector< float32, 3 > &pos ) const {
                Vector< float32, 3 > result = pos;
                result += result[2] * _diff;
                result[2] /= _zScale;

                return result + _origin;
        }

        float32 _zScale;
        Vector< float32, 3 > _origin;
        Vector< float32, 3 > _diff;
};

inline Transformation
GetTransformation ( Vector<int32,3> north, Vector<int32,3> south, Vector< float32, 3 > elExtents )
{
        //_THROW_ M4D::ErrorHandling::ETODO();
        Transformation tr;

        Vector<int32,3> diff = north - south;

        tr._origin = Vector< float32, 3 > ( south[0] * elExtents[0], south[1] * elExtents[1], south[2] * elExtents[2] );
        tr._diff = Vector< float32, 3 > ( diff[0] * elExtents[0], diff[1] * elExtents[1], 0.0f );
        //tr._diff = Vector< float32, 3 >();
        tr._zScale = 1.0f / ( ( north[2] - south[2] ) *elExtents[2] );
        return tr;
}

inline Transformation
GetTransformation ( Vector<float32,3> north, Vector<float32,3> south )
{
        //_THROW_ M4D::ErrorHandling::ETODO();
        Transformation tr;

        Vector<float32,3> diff = north - south;
        diff[2] = 0.0f;

        tr._origin = south;
        tr._diff = diff;
        //tr._diff = Vector< float32, 3 >();
        tr._zScale = 1.0f / ( north[2] - south[2] );
        return tr;
}

struct GridPointRecord {
        GridPointRecord()
                        : inProbabilityPos ( 0 ),
                        outProbabilityPos ( 0 ),
                        logRatioPos ( 0.0 ),
                        inHistogram ( 0, 256, false ),
                        outHistogram ( 0, 256, true ),
                        logHistogram ( 0, 256, true ) {}
        GridPointRecord (
                float32	pinProbabilityPos,
                float32	poutProbabilityPos,
                float32	plogRatioPos,
                int32 minHist,
                int32 maxHist
        )
                        : inProbabilityPos ( pinProbabilityPos ),
                        outProbabilityPos ( poutProbabilityPos ),
                        logRatioPos ( plogRatioPos ),
                        inHistogram ( minHist, maxHist, false ),
                        outHistogram ( minHist, maxHist, true ),
                        logHistogram ( minHist, maxHist, true ) {}
        void
        ToBinStream ( std::ostream &stream ) {
                BINSTREAM_WRITE_MACRO ( stream, inProbabilityPos );
                BINSTREAM_WRITE_MACRO ( stream, outProbabilityPos );
                BINSTREAM_WRITE_MACRO ( stream, logRatioPos );

                inHistogram.Save ( stream );
                outHistogram.Save ( stream );
                logHistogram.Save ( stream );
        }

        void
        FromBinStream ( std::istream &stream ) {
                BINSTREAM_READ_MACRO ( stream, inProbabilityPos );
                BINSTREAM_READ_MACRO ( stream, outProbabilityPos );
                BINSTREAM_READ_MACRO ( stream, logRatioPos );

                inHistogram.LoadTo ( stream );
                outHistogram.LoadTo ( stream );
                logHistogram.LoadTo ( stream );
        }

        float32	inProbabilityPos;
        float32	outProbabilityPos;
        float32	logRatioPos;

        Histogram< float32 >	inHistogram;
        Histogram< float32 >	outHistogram;
        Histogram< float32 >	logHistogram;
};

struct LayerStats {
        Vector< uint32, 2 > recordPos;
        float32	radius;
};

class ProbabilityGrid
{
public:
        typedef int32	IntensityType;

        typedef Vector< float32, 3 > Coordinates;

        typedef Vector< float32, 3 > Vector3F;
        typedef Vector< uint32, 3 > Vector3UI;
        typedef boost::shared_ptr< ProbabilityGrid > Ptr;

        ProbabilityGrid ( Vector< float32, 3 > origin, Vector< uint32, 3 > gridSize, Vector< float32, 3 > step ) :
                        _gridStep ( step ), _originCoordiantes ( origin ), _gridSize ( gridSize ), _strides ( 1, gridSize[0], gridSize[0]*gridSize[1] ) {
                _grid = new GridPointRecord[ _gridSize[0]*_gridSize[1]*_gridSize[2] ];
                _layerStats = new LayerStats[ _gridSize[2] ];
        }

        ~ProbabilityGrid() {
                delete [] _grid;
                delete [] _layerStats;
        }

        //***********************************************************************
        float32
        InProbabilityPosition ( const Coordinates &pos ) {
                return GetPointRecord ( GetClosestPoint ( pos ) ).inProbabilityPos;
        }

        float32
        OutProbabilityPosition ( const Coordinates &pos ) {
                return GetPointRecord ( GetClosestPoint ( pos ) ).outProbabilityPos;
        }

        float32
        LogRatioProbabilityPosition ( const Coordinates &pos ) {
                return GetPointRecord ( GetClosestPoint ( pos ) ).logRatioPos;
        }

        float32
        InProbabilityIntensityPosition ( const Coordinates &pos, IntensityType intensity ) {
                return GetPointRecord ( GetClosestPoint ( pos ) ).inHistogram[ intensity ];
        }

        float32
        OutProbabilityIntensityPosition ( const Coordinates &pos, IntensityType intensity ) {
                return GetPointRecord ( GetClosestPoint ( pos ) ).outHistogram[ intensity ];
        }

        float32
        LogRatioProbabilityIntensityPosition ( const Coordinates &pos, IntensityType intensity ) {
                return GetPointRecord ( GetClosestPoint ( pos ) ).logHistogram[ intensity ];
        }

        float32
        LogRatioCombination ( const Coordinates &pos, IntensityType intensity, float32 shapeBalance, float32 generalBalance ) {
                const GridPointRecord & rec = GetPointRecord ( GetClosestPoint ( pos ) );
                return shapeBalance * rec.logRatioPos + generalBalance * rec.logHistogram[ intensity ];
        }

        Vector< float32, 3 >
        GetLayerProbCenter ( const Coordinates &pos ) const {
                Vector< float32, 2 > tmp = _layerStats[ GetClosestPoint ( pos ) [2] ].recordPos;

                return Vector< float32, 3 > (
                               tmp[0]*_gridStep[0] - _originCoordiantes[0],
                               tmp[1]*_gridStep[1] - _originCoordiantes[1],
                               pos[2]
                       );
        }

        float32
        GetLayerProbRadius ( const Coordinates &pos ) const {
                return _layerStats[ GetClosestPoint ( pos ) [2] ].radius;
        }

        //***********************************************************************
        GridPointRecord &
        GetPointRecord ( const Vector< uint32, 3 > &pos ) {
                //D_PRINT( pos );
                if ( ! ( pos < _gridSize ) ) {
                        return _outlier;
                }
                return _grid[ _strides * pos ];
        }

        const GridPointRecord &
        GetPointRecord ( const Vector< uint32, 3 > &pos ) const {
                //D_PRINT( pos );
                if ( ! ( pos < _gridSize ) ) {
                        return _outlier;
                }
                return _grid[ _strides * pos ];
        }

        void
        Save ( std::ostream &stream ) {
                _gridSize.ToBinStream ( stream );
                _gridStep.ToBinStream ( stream );
                _originCoordiantes.ToBinStream ( stream );

                Vector< uint32, 3 > idx;
                for ( idx[0] = 0; idx[0] < _gridSize[0]; ++idx[0] ) {
                        for ( idx[1] = 0; idx[1] < _gridSize[1]; ++idx[1] ) {
                                for ( idx[2] = 0; idx[2] < _gridSize[2]; ++idx[2] ) {
                                        GetPointRecord ( idx ).ToBinStream ( stream );
                                }
                        }
                }
                /*for( unsigned i = 0; i < _gridSize[2]; ++i ) {
                	BINSTREAM_WRITE_MACRO( stream, (_layerStats[i]) );
                }*/

        }

        static Ptr
        Load ( std::istream &stream ) {
                Vector< float32, 3 >	gridStep;
                Vector< float32, 3 >	originCoordiantes;
                Vector< uint32, 3 >	gridSize;

                gridSize.FromBinStream ( stream );
                gridStep.FromBinStream ( stream );
                originCoordiantes.FromBinStream ( stream );

                ProbabilityGrid * result = new ProbabilityGrid ( originCoordiantes, gridSize, gridStep );

                Vector< uint32, 3 > idx;
                for ( idx[0] = 0; idx[0] < gridSize[0]; ++idx[0] ) {
                        for ( idx[1] = 0; idx[1] < gridSize[1]; ++idx[1] ) {
                                for ( idx[2] = 0; idx[2] < gridSize[2]; ++idx[2] ) {
                                        result->GetPointRecord ( idx ).FromBinStream ( stream );
                                }
                        }
                }
                /*for( unsigned i = 0; i < _gridSize[2]; ++i ) {
                	BINSTREAM_READ_MACRO( stream, (result->_layerStats[i]) );
                }*/
                result->ComputeLayerStats();
                return Ptr ( result );
        }

        void
        ComputeLayerStats() {
                Vector< uint32, 3 > idx;
                for ( idx[2] = 0; idx[2] < _gridSize[2]; ++idx[2] ) {
                        Vector< float32, 2 > tmp;
                        float32 sumProb = 0.0f;
                        uint32 count = 0;
                        for ( idx[0] = 0; idx[0] < _gridSize[0]; ++idx[0] ) {
                                for ( idx[1] = 0; idx[1] < _gridSize[1]; ++idx[1] ) {
                                        float32 prob = GetPointRecord ( idx ).inProbabilityPos;
                                        if ( prob > 0.85f ) {
                                                tmp += prob * Vector< float32, 2 > ( idx[0], idx[1] );
                                                sumProb += prob;
                                                ++count;
                                        }
                                }
                        }
                        _layerStats[ idx[2] ].recordPos = 1.0f/sumProb * tmp;
                        _layerStats[ idx[2] ].radius = sqrt ( count * _gridStep[0] * _gridStep[1] / 3.14f );
                }
        }

        SIMPLE_GET_METHOD ( Vector3UI, Size, _gridSize );
        SIMPLE_GET_METHOD ( Vector3F, GridStep, _gridStep );
        SIMPLE_GET_METHOD ( Vector3F, Origin, _originCoordiantes );
protected:

        Vector< uint32, 3 >
        GetClosestPoint ( const Coordinates &pos ) const {
                Coordinates pom = pos + _originCoordiantes;
                return Vector< uint32, 3 > (
                               ROUND ( pom[0]/_gridStep[0] ),
                               ROUND ( pom[1]/_gridStep[1] ),
                               ROUND ( pom[2]/_gridStep[2] )
                       );
        }


        Vector< float32, 3 >	_gridStep;

        Vector< float32, 3 >	_originCoordiantes;

        Vector< uint32, 3 >	_gridSize;
        Vector< uint32, 3 >	_strides;

        GridPointRecord		*_grid;//< Array of grid point records

        LayerStats		*_layerStats;

        GridPointRecord		_outlier;
};

class CanonicalProbModel
{
public:
        typedef int32	IntensityType;
        typedef Vector< float32, 3 > Coordinates;

        typedef boost::shared_ptr< CanonicalProbModel > Ptr;

        CanonicalProbModel ( ProbabilityGrid::Ptr grid, Histogram< float32 >::Ptr inHistogram, Histogram< float32 >::Ptr outHistogram, Histogram< float32 >::Ptr logRatioHistogram ) :
                        _inIntensity ( inHistogram ),
                        _outIntensity ( outHistogram ),
                        _logRatioIntensity ( logRatioHistogram ),
                        _grid ( grid ) {}
        //***********************************************************************
        float32
        InProbabilityIntesity ( IntensityType intensity );

        float32
        OutProbabilityIntesity ( IntensityType intensity );

        float32
        LogRatioProbabilityIntesity ( IntensityType intensity ) {
                return _logRatioIntensity->Get ( intensity );
        }
        //***********************************************************************

        //***********************************************************************
        float32
        InProbabilityIntesityPosition ( IntensityType intensity, const Coordinates &pos );

        float32
        OutProbabilityIntesityPosition ( IntensityType intensity, const Coordinates &pos );

        float32
        LogRatioProbabilityIntesityPosition ( IntensityType intensity, const Coordinates &pos, float32 balance = 0.5 ) {
                return balance * _logRatioIntensity->Get ( intensity )
                       + ( 1.0f-balance ) * _grid->LogRatioProbabilityPosition ( pos );
        }

        float32
        LogRatioProbabilityIntesityPositionDependent ( IntensityType intensity, const Coordinates &pos ) {
                return _grid->LogRatioProbabilityIntensityPosition ( pos, intensity );
        }
        //***********************************************************************

        //***********************************************************************
        float32
        InProbabilityPosition ( const Coordinates &pos );

        float32
        OutProbabilityPosition ( const Coordinates &pos );

        float32
        LogRatioProbabilityPosition ( const Coordinates &pos ) {
                return _grid->LogRatioProbabilityPosition ( pos );
        }

        float32
        LogRatioCombination ( const Coordinates &pos, IntensityType intensity, float32 distributionBalance, float32 shapeBalance, float32 generalBalance ) {
                return distributionBalance * _logRatioIntensity->Get ( intensity ) + _grid->LogRatioCombination ( pos, intensity, shapeBalance, generalBalance );
        }

        Vector< float32, 3 >
        GetLayerProbCenter ( const Coordinates &pos ) const {
                return _grid->GetLayerProbCenter ( pos );
        }

        float32
        GetLayerProbRadius ( const Coordinates &pos ) const {
                return _grid->GetLayerProbRadius ( pos );
        }
        //***********************************************************************
        const ProbabilityGrid &
        GetGrid() const {
                return *_grid;
        }

        const Histogram< float32 > &
        GetInIntesity() const {
                return *_inIntensity;
        }

        const Histogram< float32 > &
        GetOutIntesity() const {
                return *_outIntensity;
        }

        const Histogram< float32 > &
        GetLogRatioIntesity() const {
                return *_logRatioIntensity;
        }

        void
        SaveToFile ( std::string filename ) const {
                std::ofstream output ( filename.data(), std::ios::out | std::ios::binary );

                _inIntensity->Save ( output );
                _outIntensity->Save ( output );
                _logRatioIntensity->Save ( output );

                _grid->Save ( output );
        }
        static Ptr
        LoadFromFile ( std::string filename ) {
                std::ifstream input ( filename.data(), std::ios::in | std::ios::binary );

                if ( ! input.is_open() ) {
                        _THROW_ ExceptionBase ( TO_STRING ( "Could't open file " << filename ) );
                }

                Histogram< float32 >::Ptr inIntensity = Histogram< float32 >::Load ( input );
                Histogram< float32 >::Ptr outIntensity = Histogram< float32 >::Load ( input );
                Histogram< float32 >::Ptr logRatioIntensity = Histogram< float32 >::Load ( input );

                ProbabilityGrid::Ptr grid = ProbabilityGrid::Load ( input );

                CanonicalProbModel *result = new CanonicalProbModel ( grid, inIntensity, outIntensity, logRatioIntensity );

                return Ptr ( result );
        }
protected:


        Histogram< float32 >::Ptr	_inIntensity;
        Histogram< float32 >::Ptr	_outIntensity;
        Histogram< float32 >::Ptr	_logRatioIntensity;

        ProbabilityGrid::Ptr		_grid;
private:
};

struct InProbabilityAccessor {
        int16
        operator() ( const GridPointRecord & rec ) {
                return ( int16 ) ( 4096 * rec.inProbabilityPos );
        }
};

struct OutProbabilityAccessor {
        int16
        operator() ( const GridPointRecord & rec ) {
                return ( int16 ) ( 4096 * rec.outProbabilityPos );
        }
};

template< typename ElementType, typename Accessor >
typename Image< int16, 3 >::Ptr
MakeImageFromProbabilityGrid ( const ProbabilityGrid &grid, Accessor accessor )
{
        Vector< uint32, 3 > size = grid.GetSize();

        Image< int16, 3 >::Ptr image = ImageFactory::CreateEmptyImageFromExtents< int16, 3 > (
                                               Vector<int32,3>(), Vector<int32,3> ( ( int32* ) size.GetData() ), Vector< float32, 3 > ( 1.0f ) );

        Vector< uint32, 3 > idx;
        for ( idx[0] = 0; idx[0] < size[0]; ++idx[0] ) {
                for ( idx[1] = 0; idx[1] < size[1]; ++idx[1] ) {
                        for ( idx[2] = 0; idx[2] < size[2]; ++idx[2] ) {
                                const GridPointRecord &rec = grid.GetPointRecord ( idx );
                                image->GetElement ( idx ) = accessor ( rec );
                        }
                }
        }
        return image;
}

}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*CANONICAL_PROB_MODEL_H*/
