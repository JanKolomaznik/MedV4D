#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "MedV4D/Common/Common.h"
#include <vector>
#include <ostream>
#include <iomanip>
#include <boost/shared_ptr.hpp>


namespace M4D
{
namespace Imaging
{

class EIncompatibleHistograms: public ErrorHandling::ExceptionBase
{
public:
	EIncompatibleHistograms() throw() : ErrorHandling::ExceptionBase( "Operation could'nt proceed on these instances" )
		{}
};

template< typename CellType >
class Histogram;

template< typename CellType >
void
swap( Histogram< CellType > &histA, Histogram< CellType > &histB );

template< typename CellType >
class Histogram
{
public:
	typedef boost::shared_ptr< Histogram< CellType > > Ptr;

	typedef CellType* Iterator;
	typedef Iterator iterator;

	friend void swap< CellType >( Histogram< CellType > &histA, Histogram< CellType > &histB );

	Histogram( int32 min, int32 max, bool storeOutliers = true ) : _cells( NULL ),
		_minCell( min ), _maxCell( max ), _storeOutliers( storeOutliers ), _sum( 0 )
	{
		Resize( min, max );
	}
	
	Histogram( const Histogram &histogram ) :
		 _cells( NULL ), _minCell( histogram._minCell ), 
		 _maxCell( histogram._maxCell ), _storeOutliers( histogram._storeOutliers ), _sum( histogram._sum )
	{ 
		Resize( _minCell, _maxCell );
		//_cells = histogram._cells;
		std::copy( histogram._cells, histogram._size, _cells ); 
	}


	~Histogram()
	{ delete [] _cells; }
	void
	Resize( int32 min, int32 max )
	{
		_minCell = min;
		_maxCell = max;
		if ( _cells ) {
			delete [] _cells;
		}
		_size = _maxCell - _minCell + 2;
		_cells = new CellType[_size];
		Reset();
		//_cells.resize( _maxCell - _minCell + 2 );
	}

	CellType
	operator[]( int32 cell )const
		{
			return Get( cell );
		}

	const Histogram&
	operator=( const Histogram &histogram )
		{ 
			_minCell = histogram._minCell; 
		 	_maxCell = histogram._maxCell;
			_storeOutliers = histogram._storeOutliers;
			_sum = histogram._sum;
			Resize( _minCell, _maxCell );
			//_cells = histogram._cells;
			std::copy( histogram._cells, histogram._size, _cells );
			return *this;
		}

	void
	operator+=( const Histogram &histogram )
		{ 
			if( _minCell != histogram._minCell || _maxCell != histogram._maxCell ) {
				_THROW_	EIncompatibleHistograms();
			}
			CellType sum = 0;	
			for( unsigned i = 0; i < _size; ++i ) {
				_cells[i] += histogram._cells[i];
				sum += _cells[i];
			}
			if( !_storeOutliers ) {
				sum -= _cells[0] + _cells[ _size-1 ]; 
			}
			_sum = sum;
		}

	CellType
	Get( int32 cell )const
		{
			if( cell < _minCell ) {
				return _cells[ 0 ];
			}
			if( cell >= _maxCell ) {
				return _cells[ _maxCell - _minCell + 1 ];
			}
			return _cells[ cell - _minCell + 1 ];
		}
	void
	SetValueCell( int32 cell, CellType value )
		{
			int32 idx = cell - _minCell + 1;;
			if( cell < _minCell ) {
				if( _storeOutliers ) {
					idx = 0;
				} else return;
			}
			if( cell >= _maxCell ) {
				if( _storeOutliers ) {
					idx = _maxCell - _minCell + 1;
				} else return;
			}

			CellType diff = value - _cells[ idx ];
			_cells[ idx ] = value;
			_sum += diff;
		}

	void
	IncCell( int32 cell )
		{ SetValueCell( cell, Get( cell ) + 1 ); }

	/**
	 * Faster incrementation without bands checking.
	 **/
	void
	FastIncCell( int32 cell )
		{ 
			_cells[ cell - _minCell + 1 ] += 1;
			_sum+=1; 
		}

	CellType
	GetSum()const
		{ return _sum; }

	int32
	GetMin()const
		{ return _minCell; }

	int32
	GetMax()const
		{ return _maxCell; }

	Iterator
	Begin()
		{ return _cells + 1; }

	Iterator
	End()
		{ return _cells + _size - 1; }

	void
        Reset()
        {
                uint32 i;
                for ( i = 0; i < _size; ++i ) _cells[ i ] = 0;
        }


	void
	Save( std::ostream &stream ) 
	{
		BINSTREAM_WRITE_MACRO( stream, _minCell );
		BINSTREAM_WRITE_MACRO( stream, _maxCell );
		BINSTREAM_WRITE_MACRO( stream, _storeOutliers );
		BINSTREAM_WRITE_MACRO( stream, _sum );
		
		CellType tmp;
		for( unsigned i = 0; i < _size; ++i ) {
			tmp = _cells[i];
			BINSTREAM_WRITE_MACRO( stream, tmp );
		}
	}

	static Ptr
	Create( int32 min, int32 max, bool storeOutliers = true )
	{
		Histogram *result = new Histogram( min, max, storeOutliers );
		D_PRINT( "Histogram created : < " << min << "; " << max << " >" );
		return typename Histogram::Ptr( result );
	}

	static Ptr
	Load( std::istream &stream )
	{
		int32		minCell;
		int32		maxCell;
		bool		storeOutliers;
		CellType	sum;

		BINSTREAM_READ_MACRO( stream, minCell );
		BINSTREAM_READ_MACRO( stream, maxCell );
		BINSTREAM_READ_MACRO( stream, storeOutliers );
		BINSTREAM_READ_MACRO( stream, sum );
		
		typename Histogram::Ptr result = Histogram::Create( minCell, maxCell, storeOutliers );

		CellType tmp;
		for( unsigned i = 0; i < result->_size; ++i ) {
			BINSTREAM_READ_MACRO( stream, tmp );
			result->_cells[i] = tmp;
		}

		result->_sum = sum;

		return result;
	}

	void
	LoadTo( std::istream &stream )
	{
		//int32		minCell;
		//int32		maxCell;
		//bool		storeOutliers;
		//CellType	sum;

		BINSTREAM_READ_MACRO( stream, _minCell );
		BINSTREAM_READ_MACRO( stream, _maxCell );
		BINSTREAM_READ_MACRO( stream, _storeOutliers );
		BINSTREAM_READ_MACRO( stream, _sum );
		
		Resize( _minCell, _maxCell );

		CellType tmp;
		for( unsigned i = 0; i < _size; ++i ) {
			BINSTREAM_READ_MACRO( stream, tmp );
			_cells[i] = tmp;
		}
	}
protected:
	template< typename TCellType > friend TCellType HistogramGetMaxCount( const Histogram< TCellType > &aHistogram );

	typedef CellType* /*std::vector<CellType>*/ CellVector;

	CellVector	_cells;

	int32		_minCell;
	int32		_maxCell;
	bool		_storeOutliers;

	CellType	_sum;
	size_t		_size;
};

template< typename CellType >
void
swap( Histogram< CellType > &histA, Histogram< CellType > &histB )
{
	std::swap( histA._cells,  histB._cells );
	std::swap( histA._minCell,  histB._minCell );
	std::swap( histA._maxCell,  histB._maxCell );
	std::swap( histA._storeOutliers,  histB._storeOutliers );
	std::swap( histA._sum,  histB._sum );
	std::swap( histA._size,  histB._size );
}


typedef Histogram< uint32 > Histogram32;
typedef Histogram< uint64 > Histogram64;

template< typename CellType >
CellType
HistogramGetMaxCount( const Histogram< CellType > &aHistogram )
{
	if ( aHistogram._size == 0 ) {
		return static_cast< CellType >( 0 );
	}
	CellType maximum = aHistogram._cells[0];
	for( size_t i = 1; i < aHistogram._size; ++i ) {
		maximum = aHistogram._cells[i] > maximum ? aHistogram._cells[i] : maximum;
	}
	return maximum;
}

template< typename CellType >
std::ostream &
operator<<( std::ostream &stream, const Histogram< CellType > &histogram )
{
	stream << "Sum = " << histogram.GetSum() << std::endl;
	for( int32 i = histogram.GetMin() - 1; i <= histogram.GetMax(); ++i ) {
		stream << histogram[i] << std::endl;
	}
	return stream;
}

template< typename CellType >
CellType
ComputeSmoothedValue( const Histogram< CellType > &histogram, std::vector< float32 > &weights, int32 cell, unsigned radius )
{
	double tmp = 0.0;
	
	for( uint32 i = 0; i <= 2*radius; ++i ) {
		tmp += histogram[cell + i - radius] * weights[i];
	}
	return (CellType) tmp;

}

template< typename CellType >
Histogram< CellType >
HistogramPyramidSmooth( const Histogram< CellType > &histogram, unsigned radius )
{
	std::vector< float32 > weights;
	Histogram< CellType > result( histogram );
	//float32 sum = 0;

	for( unsigned i = 0; i <= 2*radius; ++i ){
		weights.push_back( 1.0f / ( 2*radius +1 ) );
	}
	//TODO - pyramid weigths
	int32 min = histogram.GetMin();
	int32 max = histogram.GetMax();

	for( int32 i = min; i < max; ++i ) {
		result.SetValueCell( i, ComputeSmoothedValue( histogram, weights, i, radius ) );
	}
	
	return result;
}

}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*HISTOGRAM_H*/
