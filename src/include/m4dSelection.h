#ifndef _m4dSelection_h
#define _m4dSelection_h

namespace M4D
{

namespace Selection
{

template< typename ElementType >
class m4dPoint
{

public:
    
    m4dPoint( ElementType x );
    
    m4dPoint( ElementType x, ElementType y );

    m4dPoint( ElementType x, ElementType y, ElementType z );
    
    m4dPoint( ElementType x, ElementType y, ElementType z, ElementType t );


    unsigned
    GetDimension();
    

    ElementType*
    GetAllValues();
    
    void
    SetAllValues( ElementType* v );


    ElementType&
    GetParticularValue( unsigned i ) const;
    
    ElementType&
    GetParticularValue( unsigned i );

private:
    
    unsigned _dim;
    
    ElementType* _pointValue;

};

typedef m4dPoint m4dVector;

template< typename ElementType >
class m4dLine
{

public:
    
    m4dLine( m4dPoint< ElementType >& p1, m4dPoint< ElementType >& p2 );

    m4dPoint< ElementType >&
    GetHoldingPoint();

    m4dPoint< ElementType >&
    GetHoldingPoint() const;

    m4dVector< ElementType >&
    GetDirectionVector();

    m4dVector< ElementType>&
    GetDirectionVector() const;

    bool
    PointLiesOnLine( m4dPoint< ElementType >& p );

private:

    m4dPoint< ElementType >  _HoldingPoint;
    m4dVector< ElementType > _DirectionVector;
    
};

template< typename ElementType >
class m4dPlane
{

public:

    m4dPlane( m4dPoint< ElementType >& p1, m4dPoint< ElementType >& p2, m4dPoint< ElementType >& p3 );
    
    m4dPlane( m4dPoint< ElementType >& p, m4dLine< ElementType >& l );

    m4dPoint< ElementType >&
    GetHoldingPoint();
    
    m4dPoint< ElementType >&
    GetHoldingPoint() const;
    
    m4dVector< ElementType >&
    GetFirstDirectionVector();
    
    m4dVector< ElementType >&
    GetFirstDirectionVector() const;
    
    m4dVector< ElementType >&
    GetSecondDirectionVector();
    
    m4dVector< ElementType >&
    GetSecondDirectionVector() const;

    bool
    PointLiesOnPlane( m4dPoint< ElementType >& p );

    bool
    LineLiesOnPlane( m4dLine< ElementType >& l );

private:
    m4dPoint< ElementType >  _HoldingPoint;
    m4dVector< ElementType > _FirstDirectionVector;
    m4dVector< ElementType > _SecondDirectionVector;
    
};

}
}

#endif
