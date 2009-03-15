#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "Common.h"
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
	Histogram( int32 min, int32 max, bool storeOutliers = true ) :
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
		}

	void
	operator+=( const Histogram &histogram )
		{ 
			if( _minCell != histogram._minCell || _maxCell != histogram._maxCell ) {
				_THROW_	EIncompatibleHistograms();
			}
			CellType sum = 0;	
			for( unsigned i = 0; i <= _cells.size(); ++i ) {
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
	for( int32 i = histogram.GetMin() - 1; i <= histogram.GetMax(); ++i ) {
		std::cout << histogram[i] << std::endl;
	}
	return stream;
}

}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*HISTOGRAM_H*/
