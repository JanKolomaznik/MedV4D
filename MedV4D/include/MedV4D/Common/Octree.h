#ifndef MEDV4D_OCTREE_H
#define MEDV4D_OCTREE_H

template< typename TCoordinate, typename TData, size_t tBucketSize = 5 >
class PointBucketNode
{
public:
	typedef Vector< TCoordinate, 3 > Coords;

	PointBucketNode(): mIsLeaf( true )
	{ }

	PointBucketNode *
	FindChild( Coords aCoords );

	bool
	isLeaf()const
	{
		return mIsLeaf;
	}
protected:
	void
	Split()
	{
		ASSERT( isLeaf() );
	}

	bool mIsLeaf;
	PointBucketNode *mChildren[8];
	std::vector< std::pair< Coords, TData > > *mData;

};

template< typename TCoordinate, TNode >
class Octree
{
public:
	typedef TNode Node;
	typedef Vector< TCoordinate, 3 > Coords;
protected:
	Node *
	FindLeaf( Coords aCoords )
	{
		if ( aCoords < mMin && aCoords >= mMax ) {
			return NULL;
		}
		Node *node = mRoot;
		while ( ! node->isLeaf() ) {
			node->FindChild( aCoords );
		}
		return node;
	}
private:
	Node *mRoot;
	Coords mMin;
	Coords mMax;
};


#endif /*MEDV4D_OCTREE_H*/
