#ifndef MULTI_HISTOGRAM_H
#define MULTI_HISTOGRAM_H

#include "common/Common.h"
#include <vector>
#include <ostream>
#include <iomanip>


namespace M4D
{
namespace Imaging
{

class EIncompatibleHistogram: public ErrorHandling::ExceptionBase
{
public:
	EIncompatibleHistogram() throw() : ErrorHandling::ExceptionBase( "Operation couldn't proceed on this instance" )
		{}
};

template< typename CellType, uint32 dim >
class MultiHistogram
{
public:
	MultiHistogram( vector<int32> min, vector<int32> max, bool storeOutliers = true ) :
		_minCell( min ), _maxCell( max ), _storeOutliers( storeOutliers ), _sum( 0 )
	{
		if ( _minCell.size() != dim || _maxCell.size() != dim ) {
			_THROW_ EIncompatibleHistogram();
		}
		int32 size = 1;
		uint32 i;
		for ( i = 0; i < dim; ++i ) size *= ( _maxCell[i] - _minCell[i] + 2 );
		_cells.resize( size );
	}
	
	MultiHistogram( const MultiHistogram &histogram ) :
		 _cells( histogram._cells ), _minCell( histogram._minCell ), 
		 _maxCell( histogram._maxCell ), _storeOutliers( histogram._storeOutliers ), _sum( histogram._sum )
	{ /*empty*/ }


	void
	Resize( vector<int32> min, vector<int32> max )
	{
		_minCell = min;
		_maxCell = max;
		if ( _minCell.size() != dim || _maxCell.size() != dim ) {
			_THROW_ EIncompatibleHistogram();
		}
		int32 size = 1;
		uint32 i;
		for ( i = 0; i < dim; ++i ) size *= ( _maxCell[i] - _minCell[i] + 2 );
		_cells.resize( size );
	}

	CellType
	operator[]( vector<int32> cell )const
		{
			return Get( cell );
		}

	const MultiHistogram&
	operator=( const MultiHistogram &histogram )
		{ 
			_cells = histogram._cells;
			_minCell = histogram._minCell; 
		 	_maxCell = histogram._maxCell;
			_storeOutliers = histogram._storeOutliers;
			_sum = histogram._sum;
		}

	void
	operator+=( const MultiHistogram &histogram )
		{ 
			if( _minCell != histogram._minCell || _maxCell != histogram._maxCell ) {
				_THROW_	EIncompatibleHistogram();
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
	Get( vector<int32> cell )const
		{
			uint32 i;
			if ( cell.size() != dim ) {
				_THROW_	EIncompatibleHistogram();
			}
			for ( i = 0; i < dim; ++i )
			{
				if( cell[i] < _minCell[i] ) {
					cell[ i ] = _minCell[ i ];
				}
				if( cell[i] >= _maxCell[i] ) {
					cell[ i ] = _maxCell[ i ] - 1;
				}
			}
			
			uint32 temp = 1, idx = 0;
			for ( i = 0; i < dim; ++i )
			{
				idx += temp * ( cell[i] - _minCell[i] + 1 );
				temp *= ( _maxCell[i] - _minCell[i] + 2 );
			}
			return _cells[ idx ];
		}
	void
	SetValueCell( vector<int32> cell, CellType value )
		{
			uint32 i;
			if ( cell.size() != dim ) {
				_THROW_	EIncompatibleHistogram();
			}
			for ( i = 0; i < dim; ++i )
			{
				if( cell[i] < _minCell[i] ) {
					if ( _storeOutliers ) {
						cell[ i ] = _minCell[ i ];
					} else return;
				}
				if( cell[i] >= _maxCell[i] ) {
					if ( _storeOutliers ) {
						cell[ i ] = _maxCell[ i ] - 1;
					}
				}
			}
			
			uint32 temp = 1, idx = 0;
			for ( i = 0; i < dim; ++i )
			{
				idx += temp * ( cell[i] - _minCell[i] + 1 );
				temp *= ( _maxCell[i] - _minCell[i] + 2 );
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

	vector<int32>
	GetMin()const
		{ return _minCell; }

	vector<int32>
	GetMax()const
		{ return _maxCell; }


	void
	Save( std::ostream &stream ) 
	{
		uint32 i;
		for ( i = 0; i < dim; ++i )
		{
			BINSTREAM_WRITE_MACRO( stream, _minCell[i] );
			BINSTREAM_WRITE_MACRO( stream, _maxCell[i] );
		}
		BINSTREAM_WRITE_MACRO( stream, _storeOutliers );
		BINSTREAM_WRITE_MACRO( stream, _sum );
		
		CellType tmp;
		for( unsigned i = 0; i < _cells.size(); ++i ) {
			tmp = _cells[i];
			BINSTREAM_WRITE_MACRO( stream, tmp );
		}
	}

	
	static MultiHistogram *
	Load( std::istream &stream )
	{
		vector<int32>	minCell;
		vector<int32>	maxCell;
		bool		storeOutliers;
		CellType	sum;

		uint32 i;
		for ( i = 0; i < dim; ++i )
		{
			BINSTREAM_READ_MACRO( stream, minCell[i] );
			BINSTREAM_READ_MACRO( stream, maxCell[i] );
		}
		BINSTREAM_READ_MACRO( stream, storeOutliers );
		BINSTREAM_READ_MACRO( stream, sum );
		
		MultiHistogram *result = new MultiHistogram( minCell, maxCell, storeOutliers );

		CellType tmp;
		for( unsigned i = 0; i < result->_cells.size(); ++i ) {
			BINSTREAM_READ_MACRO( stream, tmp );
			result->_cells[i] = tmp;
		}

		result->_sum = sum;

		return result;
	}
protected:
	typedef std::vector<CellType> CellVector;

	CellVector	_cells;

	vector<int32>	_minCell;
	vector<int32>	_maxCell;
	bool		_storeOutliers;

	CellType	_sum;
};


template< typename CellType >
std::ostream &
operator<<( std::ostream &stream, const MultiHistogram< CellType > &histogram )
{
	stream << "Sum = " << histogram.GetSum() << std::endl;
	for( int32 i = histogram.GetMin() - 1; i <= histogram.GetMax(); ++i ) {
		stream << histogram[i] << std::endl;
	}
	return stream;
}

}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*MULTI_HISTOGRAM_H*/
