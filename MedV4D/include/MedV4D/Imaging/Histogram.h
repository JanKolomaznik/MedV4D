#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/Vector.h"
#include <vector>
#include <ostream>
#include <iomanip>
#include <memory>


namespace M4D
{
namespace Imaging {

template<typename TValue, typename TCell>
struct Histogram1DTraits
{
	typedef TCell Cell;
	typedef TValue Value;
	typedef TValue ScalarValue;
	static const int cDimension = 1;
	static const bool cIsScalar = true;

	template<int tDimIndex>
	static ScalarValue& get(Value &aVal)
	{
		return aVal;
	}

	template<int tDimIndex>
	static const ScalarValue& get(const Value &aVal)
	{
		return aVal;
	}
};
namespace detail {

template<typename TTraits, int tDimIndex>
struct FillCellCoordinates
{
	typedef typename TTraits::Value Value;
	typedef typename TTraits::ScalarValue ScalarValue;
	static const int cDimension = TTraits::cDimension;
	typedef Vector<int, cDimension> CellCoordinates;

	static void
	apply(CellCoordinates &aCoords, const Value &aValue, const Value &aMin, const Value &aRange, CellCoordinates &aResolution)
	{
		ScalarValue scalarMin = TTraits::template get<tDimIndex>(aMin);
		ScalarValue scalarRangeSize = TTraits::template get<tDimIndex>(aRange);
		ScalarValue value = TTraits::template get<tDimIndex>(aValue);
		aCoords[tDimIndex] = round(((value - scalarMin) / double(scalarRangeSize)) * aResolution[tDimIndex]);
		FillCellCoordinates<TTraits, tDimIndex - 1>::apply(aCoords, aValue, aMin, aRange, aResolution);
	}
};

template<typename TTraits>
struct FillCellCoordinates<TTraits, 0>
{
	typedef typename TTraits::Value Value;
	typedef typename TTraits::ScalarValue ScalarValue;
	static const int cDimension = TTraits::cDimension;
	typedef Vector<int, cDimension> CellCoordinates;

	static void
	apply(CellCoordinates &aCoords, const Value &aValue, const Value &aMin, const Value &aRange, const CellCoordinates &aResolution)
	{
		ScalarValue scalarMin = TTraits::template get<0>(aMin);
		ScalarValue scalarRangeSize = TTraits::template get<0>(aRange);
		ScalarValue value = TTraits::template get<0>(aValue);
		aCoords[0] = round(((value - scalarMin) / double(scalarRangeSize)) * aResolution[0]);
	}
};

} // namespace detail

template <typename TTraits>
class HistogramBase
{
public:
	typedef typename TTraits::Value Value;
	typedef typename TTraits::ScalarValue ScalarValue;
	typedef typename TTraits::Cell Cell;
	static const int cDimension = TTraits::cDimension;
	typedef Vector<int, cDimension> CellCoordinates;

	HistogramBase() = default;
	HistogramBase(Value aMin, Value aMax, CellCoordinates aResolution)
		: mMin(aMin)
		, mMax(aMax)
		, mResolution(aResolution)
		, mData(VectorCoordinateProduct(aResolution))
		, mRangeSize(aMax - aMin)
	{
		mStrides[0] = 1;
		int stride = 1;
		for (int i = 1; i < cDimension; ++i) {
			stride *= mResolution[i - 1];
			mStrides[i] = stride;
		}
	}


protected:

	CellCoordinates
	valueToCell(Value aVal) const
	{
		aVal = clamp(mMin, mMax, aVal);
		CellCoordinates coords;
		detail::FillCellCoordinates<TTraits, cDimension - 1>::apply(coords, aVal, mMin, mRangeSize, mResolution);
		return coords;
	}

	Cell &
	getCell(const Value &aValue)
	{
		return mData[valueToCell(aValue) * mStrides];
	}

	Value mMin;
	Value mMax;
	CellCoordinates mResolution;
	std::vector<Cell> mData;

	Value mRangeSize;
	CellCoordinates mStrides;
};

template<typename TValue, typename TCell = int64_t>
class Histogram1D : HistogramBase<Histogram1DTraits<TValue, TCell>>
{
public:
	typedef HistogramBase<Histogram1DTraits<TValue, TCell>> Predecessor;
	typedef typename Predecessor::Value Value;
	typedef typename Predecessor::ScalarValue ScalarValue;
	typedef typename Predecessor::Cell Cell;
	static const int cDimension = Predecessor::cDimension;
	typedef typename Predecessor::CellCoordinates CellCoordinates;

	Histogram1D(Value aMin, Value aMax, int aResolution)
		: Predecessor(aMin, aMax, CellCoordinates(aResolution))
	{}

	Histogram1D() = default;

	void
	put(const Value &aValue) {
		this->getCell(aValue) += 1;
	}
};


class Statistics {
public:
	typedef std::shared_ptr<Statistics> Ptr;
	typedef std::weak_ptr<Statistics> WPtr;

};

class EIncompatibleHistograms: public ErrorHandling::ExceptionBase
{
public:
	EIncompatibleHistograms() throw() : ErrorHandling::ExceptionBase ( "Operation could'nt proceed on these instances" ) {}
};

template< typename CellType >
class SimpleHistogram;

template< typename CellType >
void
swap ( SimpleHistogram< CellType > &histA, SimpleHistogram< CellType > &histB );

template< typename CellType >
class SimpleHistogram
{
public :
	typedef std::shared_ptr< SimpleHistogram< CellType > > Ptr;

	typedef CellType* Iterator;
	typedef Iterator iterator;

	friend void swap< CellType > ( SimpleHistogram< CellType > &histA, SimpleHistogram< CellType > &histB );

	SimpleHistogram ( int32 min, int32 max, bool storeOutliers = true ) : _cells ( NULL ),
			_minCell ( min ), _maxCell ( max ), _storeOutliers ( storeOutliers ), _sum ( 0 ) {
		resize ( min, max );
	}

	SimpleHistogram ( const SimpleHistogram &histogram ) :
			_cells ( NULL ), _minCell ( histogram._minCell ),
			_maxCell ( histogram._maxCell ), _storeOutliers ( histogram._storeOutliers ), _sum ( histogram._sum ) {
		resize ( _minCell, _maxCell );
		//_cells = histogram._cells;
		std::copy ( histogram._cells, histogram._size, _cells );
	}


	~SimpleHistogram() {
		delete [] _cells;
	}
	void
	resize ( int32 min, int32 max ) {
		_minCell = min;
		_maxCell = max;
		if ( _cells ) {
			delete [] _cells;
		}
		_size = _maxCell - _minCell + 2;
		_cells = new CellType[_size];
		reset();
		//_cells.resize( _maxCell - _minCell + 2 );
	}

	CellType
	operator[] ( int32 cell ) const {
		return get ( cell );
	}

	const SimpleHistogram&
	operator= ( const SimpleHistogram &histogram ) {
		_minCell = histogram._minCell;
		_maxCell = histogram._maxCell;
		_storeOutliers = histogram._storeOutliers;
		_sum = histogram._sum;
		resize ( _minCell, _maxCell );
		//_cells = histogram._cells;
		std::copy ( histogram._cells, histogram._size, _cells );
		return *this;
	}

	void
	operator+= ( const SimpleHistogram &histogram ) {
		if ( _minCell != histogram._minCell || _maxCell != histogram._maxCell ) {
			_THROW_	EIncompatibleHistograms();
		}
		CellType sum = 0;
		for ( unsigned i = 0; i < _size; ++i ) {
			_cells[i] += histogram._cells[i];
			sum += _cells[i];
		}
		if ( !_storeOutliers ) {
			sum -= _cells[0] + _cells[ _size-1 ];
		}
		_sum = sum;
	}

	CellType
	get ( int32 cell ) const {
		if ( cell < _minCell ) {
			return _cells[ 0 ];
		}
		if ( cell >= _maxCell ) {
			return _cells[ _maxCell - _minCell + 1 ];
		}
		return _cells[ cell - _minCell + 1 ];
	}
	void
	setValueCell( int32 cell, CellType value ) {
		int32 idx = cell - _minCell + 1;;
		if ( cell < _minCell ) {
			if ( _storeOutliers ) {
				idx = 0;
			} else return;
		}
		if ( cell >= _maxCell ) {
			if ( _storeOutliers ) {
				idx = _maxCell - _minCell + 1;
			} else return;
		}

		CellType diff = value - _cells[ idx ];
		_cells[ idx ] = value;
		_sum += diff;
	}

	void
	incCell ( int32 cell ) {
		setValueCell ( cell, get ( cell ) + 1 );
	}

	/**
	 * Faster incrementation without bands checking.
	 **/
	void
	fastIncCell ( int32 cell ) {
		_cells[ cell - _minCell + 1 ] += 1;
		_sum+=1;
	}

	CellType
	getSum() const {
		return _sum;
	}

	int32
	getMin() const {
		return _minCell;
	}

	int32
	getMax() const {
		return _maxCell;
	}

	Iterator
	begin() {
		return _cells + 1;
	}

	Iterator
	end() {
		return _cells + _size - 1;
	}

	void
	reset() {
		uint32 i;
		for ( i = 0; i < _size; ++i ) _cells[ i ] = 0;
	}


	void
	save ( std::ostream &stream ) {
		BINSTREAM_WRITE_MACRO ( stream, _minCell );
		BINSTREAM_WRITE_MACRO ( stream, _maxCell );
		BINSTREAM_WRITE_MACRO ( stream, _storeOutliers );
		BINSTREAM_WRITE_MACRO ( stream, _sum );

		CellType tmp;
		for ( unsigned i = 0; i < _size; ++i ) {
			tmp = _cells[i];
			BINSTREAM_WRITE_MACRO ( stream, tmp );
		}
	}

	static Ptr
	create ( int32 min, int32 max, bool storeOutliers = true ) {
		SimpleHistogram *result = new SimpleHistogram ( min, max, storeOutliers );
		D_PRINT ( "Histogram created : < " << min << "; " << max << " >" );
		return typename SimpleHistogram::Ptr ( result );
	}

	static Ptr
	load ( std::istream &stream ) {
		int32		minCell;
		int32		maxCell;
		bool		storeOutliers;
		CellType	sum;

		BINSTREAM_READ_MACRO ( stream, minCell );
		BINSTREAM_READ_MACRO ( stream, maxCell );
		BINSTREAM_READ_MACRO ( stream, storeOutliers );
		BINSTREAM_READ_MACRO ( stream, sum );

		typename SimpleHistogram::Ptr result = SimpleHistogram::create ( minCell, maxCell, storeOutliers );

		CellType tmp;
		for ( unsigned i = 0; i < result->_size; ++i ) {
			BINSTREAM_READ_MACRO ( stream, tmp );
			result->_cells[i] = tmp;
		}

		result->_sum = sum;

		return result;
	}

	void
	loadTo ( std::istream &stream ) {
		//int32		minCell;
		//int32		maxCell;
		//bool		storeOutliers;
		//CellType	sum;

		BINSTREAM_READ_MACRO ( stream, _minCell );
		BINSTREAM_READ_MACRO ( stream, _maxCell );
		BINSTREAM_READ_MACRO ( stream, _storeOutliers );
		BINSTREAM_READ_MACRO ( stream, _sum );

		resize ( _minCell, _maxCell );

		CellType tmp;
		for ( unsigned i = 0; i < _size; ++i ) {
			BINSTREAM_READ_MACRO ( stream, tmp );
			_cells[i] = tmp;
		}
	}
protected:
	template< typename TCellType > friend TCellType HistogramGetMaxCount ( const SimpleHistogram< TCellType > &aHistogram );

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
swap ( SimpleHistogram< CellType > &histA, SimpleHistogram< CellType > &histB )
{
	std::swap ( histA._cells,  histB._cells );
	std::swap ( histA._minCell,  histB._minCell );
	std::swap ( histA._maxCell,  histB._maxCell );
	std::swap ( histA._storeOutliers,  histB._storeOutliers );
	std::swap ( histA._sum,  histB._sum );
	std::swap ( histA._size,  histB._size );
}


typedef SimpleHistogram< uint32 > SimpleHistogram32;
typedef SimpleHistogram< uint64 > SimpleHistogram64;

template< typename CellType >
CellType
histogramGetMaxCount( const SimpleHistogram< CellType > &aHistogram )
{
	if ( aHistogram._size == 0 ) {
		return static_cast< CellType > ( 0 );
	}
	CellType maximum = aHistogram._cells[0];
	for ( size_t i = 1; i < aHistogram._size; ++i ) {
		maximum = aHistogram._cells[i] > maximum ? aHistogram._cells[i] : maximum;
	}
	return maximum;
}

template< typename CellType >
std::ostream &
operator<< ( std::ostream &stream, const SimpleHistogram< CellType > &histogram )
{
	stream << "Sum = " << histogram.getSum() << std::endl;
	for ( int32 i = histogram.getMin() - 1; i <= histogram.getMax(); ++i ) {
		stream << histogram[i] << std::endl;
	}
	return stream;
}

template< typename CellType >
CellType
ComputeSmoothedValue ( const SimpleHistogram< CellType > &histogram, std::vector< float32 > &weights, int32 cell, unsigned radius )
{
	double tmp = 0.0;

	for ( uint32 i = 0; i <= 2*radius; ++i ) {
		tmp += histogram[cell + i - radius] * weights[i];
	}
	return ( CellType ) tmp;

}

template< typename CellType >
SimpleHistogram< CellType >
HistogramPyramidSmooth ( const SimpleHistogram< CellType > &histogram, unsigned radius )
{
	std::vector< float32 > weights;
	SimpleHistogram< CellType > result ( histogram );
	//float32 sum = 0;

	for ( unsigned i = 0; i <= 2*radius; ++i ) {
		weights.push_back ( 1.0f / ( 2*radius +1 ) );
	}
	//TODO - pyramid weigths
	int32 min = histogram.getMin();
	int32 max = histogram.getMax();

	for ( int32 i = min; i < max; ++i ) {
		result.setValueCell ( i, ComputeSmoothedValue ( histogram, weights, i, radius ) );
	}

	return result;
}

}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*HISTOGRAM_H*/
