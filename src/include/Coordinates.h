


template< typename CoordType, unsigned Dim >
class Coordinates;

template< typename CoordType  >
class Coordinates< CoordType, 2 >
{
public:
	Coordinates( const CoordType &x, const CoordType &y );
};

template< typename CoordType  >
class Coordinates< CoordType, 3 >
{
public:
	Coordinates( const CoordType &x, const CoordType &y, const CoordType &z );
};
