#ifndef MULTI_HISTOGRAM_H
#define MULTI_HISTOGRAM_H

#include "MedV4D/Common/Common.h"
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
	MultiHistogram( std::vector<int32> min, std::vector<int32> max, bool storeOutliers = true ) :
		_sum( 0 ),
		_storeOutliers( storeOutliers ),
		_minCell( min.back() ),
		_maxCell( max.back() ),
		_cells( _maxCell - _minCell + 2, MultiHistogram< CellType, dim-1 >( std::vector<int32>( min.begin(), min.end() - 1 ), std::vector<int32>( max.begin(), max.end() - 1 ) ) )
	{
		if ( min.size() != dim || max.size() != dim ) {
			_THROW_ EIncompatibleHistogram();
		}
	}
	
	MultiHistogram( const MultiHistogram< CellType, dim > &histogram ) :
		_sum( histogram._sum ),
		_storeOutliers( histogram._storeOutliers ),
		_minCell( histogram._minCell ), 
		_maxCell( histogram._maxCell ),
		_cells( histogram._cells )
	{ /*empty*/ }


	void
	Resize( const std::vector<int32> min, const std::vector<int32> max )
	{
		if ( min.size() != dim || max.size() != dim ) {
			_THROW_ EIncompatibleHistogram();
		}
		_minCell = min.back();
		_maxCell = max.back();
		_cells.resize( _maxCell - _minCell + 2 );
		uint32 i;
		max.pop_back();
		min.pop_back();
		for ( i = 0; i < _cells.size(); ++i ) _cells[ i ].Resize( min, max );
	}

	CellType
	operator[]( std::vector<int32> cell )const
		{
			return Get( cell );
		}

	MultiHistogram< CellType, dim-1 >
	operator[]( int32 cell )const
		{
			return Get( cell );
		}

	const MultiHistogram< CellType, dim > &
	operator=( const MultiHistogram< CellType, dim > &histogram )
		{ 
			_cells = histogram._cells;
			_minCell = histogram._minCell; 
		 	_maxCell = histogram._maxCell;
			_storeOutliers = histogram._storeOutliers;
			_sum = histogram._sum;
			return *this;
		}

	void
	operator+=( const MultiHistogram< CellType, dim > &histogram )
		{ 
			if( _minCell != histogram._minCell || _maxCell != histogram._maxCell ) {
				_THROW_	EIncompatibleHistogram();
			}
			CellType sum = 0;	
			for( unsigned i = 0; i < _cells.size(); ++i ) {
				_cells[i] += histogram._cells[i];
				sum += _cells[i].GetSum();
			}
			if( !_storeOutliers ) {
				sum -= _cells[0].GetSum() + _cells[ _cells.size()-1 ].GetSum(); 
			}
			_sum = sum;
		}

	CellType
	Get( std::vector<int32> cell )const
		{
			if ( cell.size() != dim ) {
				_THROW_	EIncompatibleHistogram();
			}

			int32 idx = cell.back();
			cell.pop_back();
			
			if( idx < _minCell ) {
				idx = _minCell;
			}
			if( idx >= _maxCell ) {
				idx = _maxCell - 1;
			}

			idx -= ( _minCell - 1 );

			return _cells[idx].Get( cell );
		}

	MultiHistogram< CellType, dim-1 >
	Get( int32 cell )const
		{
			if( cell < _minCell ) {
				cell = _minCell;
			}
			if( cell >= _maxCell ) {
				cell = _maxCell - 1;
			}

			return _cells[cell - _minCell + 1];
		}

	CellType
	SetValueCell( std::vector<int32> cell, CellType value )
		{
			if ( cell.size() != dim ) {
				_THROW_	EIncompatibleHistogram();
			}

			int32 idx = cell.back();
			cell.pop_back();

			if( idx < _minCell ) {
				if ( _storeOutliers ) {
					idx = _minCell;
				} else return 0;
			}
			if( idx >= _maxCell ) {
				if ( _storeOutliers ) {
					idx = _maxCell - 1;
				} else return 0;
			}

			idx -= ( _minCell - 1 );
			
			CellType diff = _cells[ idx ].SetValueCell( cell, value );
			_sum += diff;
			return diff;
		}

	void
	IncCell( std::vector<int32> cell )
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
		for ( i = 0; i < _cells.size(); ++i ) _cells[ i ].Reset();
	}


	void
	Save( std::ostream &stream ) 
	{
		BINSTREAM_WRITE_MACRO( stream, _minCell );
		BINSTREAM_WRITE_MACRO( stream, _maxCell );
		BINSTREAM_WRITE_MACRO( stream, _storeOutliers );
		BINSTREAM_WRITE_MACRO( stream, _sum );
		
		MultiHistogram< CellType, dim-1 > tmp;
		for( uint32 i = 0; i < _cells.size(); ++i ) {
			tmp = _cells[i];
			BINSTREAM_WRITE_MACRO( stream, tmp );
		}
	}

	
	static MultiHistogram< CellType, dim > *
	Load( std::istream &stream )
	{
		std::vector<int32>	minCell;
		std::vector<int32>	maxCell;
		bool		storeOutliers;
		CellType	sum;

		BINSTREAM_READ_MACRO( stream, minCell );
		BINSTREAM_READ_MACRO( stream, maxCell );
		BINSTREAM_READ_MACRO( stream, storeOutliers );
		BINSTREAM_READ_MACRO( stream, sum );
		
		MultiHistogram< CellType, dim > *result = new MultiHistogram< CellType, dim >( minCell, maxCell, storeOutliers );

		MultiHistogram< CellType, dim-1 > tmp;
		for( uint32 i = 0; i < result->_cells.size(); ++i ) {
			BINSTREAM_READ_MACRO( stream, tmp );
			result->_cells[i] = tmp;
		}

		result->_sum = sum;

		return result;
	}
protected:
	CellType								_sum;

	bool									_storeOutliers;
	int32									_minCell;
	int32									_maxCell;

	typedef typename std::vector< MultiHistogram< CellType,dim-1 > >	HistogramVector;

	HistogramVector								_cells;

};


template< typename CellType, uint32 dim >
std::ostream &
operator<<( std::ostream &stream, const MultiHistogram< CellType, dim > &histogram )
{
	stream << "Sum = " << histogram.GetSum() << std::endl;
	for( int32 i = 0; i <= histogram._cells.size(); ++i ) {
		stream << histogram._cells[i] << std::endl;
	}
	return stream;
}

template< typename CellType >
class MultiHistogram< CellType, 1 >
{
public:
	MultiHistogram( int32 min, int32 max, bool storeOutliers = true ) :
		_sum( 0 ),
		_storeOutliers( storeOutliers ),
		_minCell( min ),
		_maxCell( max ),
		_cells( max - min + 2, 0 )
	{
	}
	
	MultiHistogram( std::vector<int32> min, std::vector<int32> max, bool storeOutliers = true ) :
		_sum( 0 ),
		_storeOutliers( storeOutliers ),
		_minCell( min.back() ),
		_maxCell( max.back() ),
		_cells( _maxCell - _minCell + 2, 0 )
	{
		if ( min.size() != 1 || max.size() != 1 ) {
			_THROW_ EIncompatibleHistogram();
		}
	}
	
	MultiHistogram( const MultiHistogram< CellType, 1 > &histogram ) :
		_sum( histogram._sum ),
		_storeOutliers( histogram._storeOutliers ),
		_minCell( histogram._minCell ), 
		_maxCell( histogram._maxCell ),
		_cells( histogram._cells )
	{ /*empty*/ }


	void
	Resize( int32 min, int32 max )
	{
		_minCell = min;
		_maxCell = max;
		_cells.resize( _maxCell - _minCell + 2 );
	}

	void
	Resize( std::vector<int32> min, std::vector<int32> max )
	{
		if ( min.size() != 1 || max.size() != 1 ) {
			_THROW_ EIncompatibleHistogram();
		}
		_minCell = min.back();
		_maxCell = max.back();
		_cells.resize( _maxCell - _minCell + 2 );
		max.pop_back();
		min.pop_back();
	}

	CellType
	operator[]( int32 cell )const
		{
			return Get( cell );
		}

	CellType
	operator[]( std::vector<int32> cell )const
		{
			return Get( cell );
		}

	const MultiHistogram< CellType, 1 >&
	operator=( const MultiHistogram< CellType, 1 > &histogram )
		{ 
			_cells = histogram._cells;
			_minCell = histogram._minCell; 
		 	_maxCell = histogram._maxCell;
			_storeOutliers = histogram._storeOutliers;
			_sum = histogram._sum;
			return *this;
		}

	void
	operator+=( const MultiHistogram< CellType, 1 > &histogram )
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

	CellType
	Get( std::vector<int32> cell )const
		{
			if ( cell.size() != 1 ) {
				_THROW_	EIncompatibleHistogram();
			}

			int32 idx = cell.back();
			cell.pop_back();
			
			if( idx < _minCell ) {
				idx = _minCell;
			}
			if( idx >= _maxCell ) {
				idx = _maxCell - 1;
			}

			idx -= ( _minCell - 1 );

			return _cells[idx];
		}

	CellType
	SetValueCell( int32 cell, CellType value )
		{
			int32 idx = cell;
			if( cell < _minCell ) {
				if( _storeOutliers ) {
					idx = 0;
				} else return 0;
			}
			if( cell >= _maxCell ) {
				if( _storeOutliers ) {
					idx = _maxCell - _minCell + 1;
				} else return 0;
			}

			idx -= ( _minCell - 1 );

			CellType diff = value - _cells[ idx ];
			_cells[ idx ] = value;
			_sum += diff;
			return diff;
		}

	CellType
	SetValueCell( std::vector<int32> cell, CellType value )
		{
			if ( cell.size() != 1 ) {
				_THROW_	EIncompatibleHistogram();
			}

			int32 idx = cell.back();
			cell.pop_back();

			if( idx < _minCell ) {
				if ( _storeOutliers ) {
					idx = _minCell;
				} else return 0;
			}
			if( idx >= _maxCell ) {
				if ( _storeOutliers ) {
					idx = _maxCell - 1;
				} else return 0;
			}

			idx -= ( _minCell - 1 );

			CellType diff = value - _cells[ idx ];
			_cells[ idx ] = value;
			_sum += diff;
			return diff;
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

	
	static MultiHistogram< CellType, 1 > *
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
		
		MultiHistogram< CellType, 1 > *result = new MultiHistogram< CellType, 1 >( minCell, maxCell, storeOutliers );

		CellType tmp;
		for( unsigned i = 0; i < result->_cells.size(); ++i ) {
			BINSTREAM_READ_MACRO( stream, tmp );
			result->_cells[i] = tmp;
		}

		result->_sum = sum;

		return result;
	}
protected:
	CellType	_sum;

	bool		_storeOutliers;
	int32		_minCell;
	int32		_maxCell;

	typedef std::vector<CellType> CellVector;

	CellVector	_cells;

};

template< typename CellType >
std::ostream &
operator<<( std::ostream &stream, const MultiHistogram< CellType, 1 > &histogram )
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
