#ifndef TF_INDEXER
#define TF_INDEXER

namespace M4D {
namespace GUI {
namespace TF {

class Indexer{

	typedef std::vector<TF::Size> Indexes;
	typedef Indexes::iterator IndexesIt;

public:

	Indexer(): nextIndex_(0){}
	~Indexer(){}

	TF::Size getIndex(){

		TF::Size index = nextIndex_;
		if(!released_.empty())
		{
			index = released_[released_.size()-1];
			released_.pop_back();
		}
		else
		{
			++nextIndex_;
		}
		return index;
	}

	void releaseIndex(const TF::Size index){

		if(index == (nextIndex_-1))
		{
			--nextIndex_;

			IndexesIt newBegin = released_.begin();
			IndexesIt end = released_.end();
			while( (newBegin != end) && (*newBegin == (nextIndex_-1)) )
			{
				++newBegin;
				--nextIndex_;
			}
			if(newBegin == end) released_.clear();
			else released_ = Indexes(newBegin, end);
		}
		else
		{
			IndexesIt it = released_.begin();
			IndexesIt end = released_.end();

			while( (it != end) && (*it > index) ) ++it;
			released_.insert(it, index);
		}
	}

private:

	TF::Size nextIndex_;
	Indexes released_;
};

}	//namespace TF
}	//namespace GUI
}	//namespace M4D

#endif	//TF_INDEXER