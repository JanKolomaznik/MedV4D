#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "common/Common.h"
#include <vector>
#include <ostream>
#include <iomanip>


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
class Histogram
{
public:
	Histogram( int32 min=0, int32 max=255, bool storeOutliers = true ) :
		_minCell( min ), _maxCell( max ), _storeOutliers( storeOutliers ), _sum( 0 )
	{
		_cells.resize( _maxCell - _minCell + 2 );
	}
	
	Histogram( const Histogram &histogram ) :
		 _cells( histogram._cells ), _minCell( histogram._minCell ), 
		 _maxCell( histogram._maxCell ), _storeOutliers( histogram._storeOutliers ), _sum( histogram._sum )
	{ /*empty*/ }


	void
	Resize( int32 min, int32 max )
	{
		_minCell = min;
		_maxCell = max;
		_cells.resize( _maxCell - _minCell + 2 );
	}

	CellType
	operator[]( int32 cell )const
		{
			return Get( cell );
		}

	const Histogram&
	operator=( const Histogram &histogram )
		{ 
			_cells = histogram._cells;
			_minCell = histogram._minCell; 
		 	_maxCell = histogram._maxCell;
			_storeOutliers = histogram._storeOutliers;
			_sum = histogram._sum;
			
			return *this;
		}

	void
	operator+=( const Histogram &histogram )
		{ 
			if( _minCell != histogram._minCell || _maxCell != histogram._maxCell ) {
				_THROW_	EIncompatibleHistograms();
			}
			CellType sum = 0;	
			for( unsigned i = 0; i < _cells.size(); ++i ) {
				_cells[i] += histogram._cells[i];
				sum += _cells[i];
			}
			if( !_storeOutliers ) {
				sum -= _cells[0] + _cells[ _cells.size()-1 ]; 
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

	CellType
	GetSum()const
		{ return _sum; }

	int32
	GetMin()const
		{ return _minCell; }

	int32
	GetMax()const
		{ return _maxCell; }

	void
        Reset()
        {
                uint32 i;
                for ( i = 0; i < _cells.size(); ++i ) _cells[ i ] = 0;
        }


	void
	Save( std::ostream &stream ) 
	{
		BINSTREAM_WRITE_MACRO( stream, _minCell );
		BINSTREAM_WRITE_MACRO( stream, _maxCell );
		BINSTREAM_WRITE_MACRO( stream, _storeOutliers );
		BINSTREAM_WRITE_MACRO( stream, _sum );
		
		CellType tmp;
		for( unsigned i = 0; i < _cells.size(); ++i ) {
			tmp = _cells[i];
			BINSTREAM_WRITE_MACRO( stream, tmp );
		}
	}

	
	static Histogram *
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
		
		Histogram *result = new Histogram( minCell, maxCell, storeOutliers );

		CellType tmp;
		for( unsigned i = 0; i < result->_cells.size(); ++i ) {
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
		
		_cells.resize( _maxCell - _minCell + 2 );

		CellType tmp;
		for( unsigned i = 0; i < _cells.size(); ++i ) {
			BINSTREAM_READ_MACRO( stream, tmp );
			_cells[i] = tmp;
		}
	}
protected:
	typedef std::vector<CellType> CellVector;

	CellVector	_cells;

	int32		_minCell;
	int32		_maxCell;
	bool		_storeOutliers;

	CellType	_sum;
};


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

}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*HISTOGRAM_H*/
