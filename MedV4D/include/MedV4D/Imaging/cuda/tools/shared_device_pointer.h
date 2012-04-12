/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

/**
 * @author Bjoern Knafla <bknafla@uni-kassel.de>
 * @author Jens Breitbart modified the smart pointer to support device pointers
 * @version 0.1
 * @date 24.07.06
 *
 * Reference counting smart pointer for device pointer.
 */
#ifndef CUPP_shared_device_pointer_H
#define CUPP_shared_device_pointer_H

#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif


// Include std::swap
#include <algorithm>

// Include assert
#include <cassert>

// cupp::free()
#include "cupp/runtime.h"


namespace cupp {

/**
* Helper class for @c shared_device_pointer.
*/
struct SharedPointerReferenceCount {
	typedef size_t size_type;

	explicit SharedPointerReferenceCount(): referenceCount_( 1 ) {
		// Nothing to do.
	}

	size_type referenceCount_;
};


/**
* Smart pointer implementation using reference counting. Dereferencing is
* as fast as it can get and might reach the speed of dereferencing raw
* pointers if a good optimizing compiler is used.
*
* See Andrei Alexandrescu, Modern C++ Design, Addison-Wesley, 2002,
* pp. 165--167.
*
* See http://Boost.org shared_ptr
* ( http://www.boost.org/libs/smart_ptr/shared_ptr.htm ).
*
* @attention Beware of cycles of smart pointers as these will lead to
* memory leaks.
*/
template< typename T >
class shared_device_pointer {
public:
	typedef size_t             size_type;
	typedef T                  value_type;
	typedef value_type&        reference;
	typedef value_type const&  const_reference;
	typedef value_type*        pointer;
	typedef value_type const*  const_pointer;

	template< typename U > friend class shared_device_pointer;


	/**
	* Constructs an empty @c shared_device_pointer. The use count is unspecified.
	*
	* @post <code>get() == 0</code>
	*
	* @throw @c std::bad_alloc if memory could not be obtained.
	*/
	shared_device_pointer() : data_( 0 ), referenceCount_( new SharedPointerReferenceCount() ) {
		// Nothing to do.
	}

	/**
	* Constructs a @c shared_device_pointer that owns the pointer @a _data.
	*
	* @post <code> use_count() == 1 </code> and <code>get() == _data )</code>
	*
	* @throw @c std::bad_alloc if memory could not be obtained.
	*/
	explicit shared_device_pointer( T* _data ) : data_( _data ), referenceCount_( new SharedPointerReferenceCount() ) {
		// Nothing to do.
	}

	/**
	* Constructs a @c shared_device_pointer that shares ownership with @a other.
	*
	* @post <code>get() == other.get()</code> and
	*       <code>useCount() == other.useCount()</code>
	*
	* @throw Nothing.
	*/
	shared_device_pointer( shared_device_pointer const& other ) : data_( other.data_ ), referenceCount_( other.referenceCount_ ) {
		retain();
	}

	/**
	* Constructs a @c shared_device_pointer that shares ownership with @a other.
	* A pointer of type @c U must be assignable to a pointer of type @c T.
	*
	* @post <code>get() == other.get()</code> and
	*       <code>useCount() == other.useCount()</code>
	*
	* @throw Nothing.
	*/
	template< typename U >
	shared_device_pointer( shared_device_pointer< U > const& other ) : data_( other.data_ ), referenceCount_( other.referenceCount_ ) {
		retain();
	}

	/**
	* Decreases the use count by one. If the use count hits @c 0 the
	* managed pointer is deleted.
	*
	* @throw Might throw an exception if the destructor of the managed
	*        pointer throws an exception if called.
	*/
	~shared_device_pointer() {
		release();
	}

	/**
	* Shares the ownership of the pointer managed by @a other with @a other.
	* The old managed pointer is deleted if ownership decreases to @c 0.
	*
	* @throw Nothing.
	*/
	shared_device_pointer& operator=( shared_device_pointer other ) {
		swap( other );
		return *this;
	}

	/**
	* Swaps the managed data and the ownership of it with @a other.
	*
	* @throw Nothing.
	*/
	void swap( shared_device_pointer& other ) {
		std::swap( data_, other.data_ );
		std::swap( referenceCount_, other.referenceCount_ );
	}

	size_type useCount() const {
		return referenceCount_->referenceCount_;
	}

	/**
	* Get direct control of the raw managed pointer.
	*
	* @attention Don't delete the pointer behind the back of the smart
	*            pointer because this leads to undefined behavior.
	*
	* @attention Don't put the pointer under the management of this or
	*            another smart pointer!
	*/
	pointer get() const {
		return data_;
	}

	/**
	* Sets a new pointer to be managed by the smart pointer.
	*
	* @attention Don't add a pointer that is already managed by another
	*            smart pointer. Otherwise behavior is undefined.
	*/
	void reset( T* _data = 0 ) {
		shared_device_pointer( _data ).swap( *this );
	}


	/**
	* See http://Boost.org shared_ptr
	* ( http://www.boost.org/libs/smart_ptr/shared_ptr.htm ).
	*/
	typedef T* (shared_device_pointer::*unspecified_bool_type)() const;

	/**
	* Automatic cast operator to enable use of a smart pointer inside a
	* conditional, for example to test if the smart pointer manages a @c 0
	* pointer.
	*
	* See http://Boost.org shared_ptr
	* ( http://www.boost.org/libs/smart_ptr/shared_ptr.htm ).
	*/
	operator unspecified_bool_type () const {
		return 0 == data_ ? 0 : &shared_device_pointer::get;
	}


	template< typename U >
	bool operator<( shared_device_pointer< U > const& rhs ) {
		// Because of sub-typing two different pointers might
		// nonetheless point to the same class instance. Therefore use
		// the reference count pointer for comparisons.
		return referenceCount_ < rhs.referenceCount_;
	}


	/**
	* Calls @c static_cast for the pointer shared by @a sp while
	* preserving the right use count.
	*
	* Naming scheme resembles that of the Boost library.
	*/
	template< typename U >
	inline friend shared_device_pointer< U > static_pointer_cast( shared_device_pointer const& sp ) {
		return sp.static_pointer_cast< U >();
	}


	/**
	* Calls @c const_cast for the pointer shared by @a sp while
	* preserving the right use count.
	*
	* Naming scheme resembles that of the Boost library.
	*/
	template< typename U >
	inline friend shared_device_pointer< U > const_pointer_cast( shared_device_pointer const& sp ) {
		return sp.const_pointer_cast< U >();
	}

	/**
	* Calls @c dynamic_cast for the pointer shared by @a sp while
	* preserving the right use count.
	*
	* Naming scheme resembles that of the Boost library.
	*/
	template< typename U >
	inline friend shared_device_pointer< U > dynamic_pointer_cast( shared_device_pointer const& sp ) {
		return sp.dynamic_pointer_cast< U >();

	}


private:

	/**
	* Decreases the reference count of the managed pointer. If the
	* reference count reaches @c 0 the managed pointer is deleted, too.
	*
	* @attention @c release and @c retain are used internally and only
	*            public to allow developers to gain more control over
	*            the smart pointer. But beware, if release is called and
	*            the managed pointer is destroyed further calls to the
	*            dereference operators or even the destructor show
	*            undefined behavior and might crash the application.
	*/
	void release() {
		assert( 0 < referenceCount_->referenceCount_ && "Only call release for reference counts greater than 0." );

		--( referenceCount_->referenceCount_ );
		if ( 0 == referenceCount_->referenceCount_ ) {
			cupp::free(data_);
			data_ = 0;
			delete referenceCount_;
			referenceCount_ = 0;
		}
	}

	/**
	* Increases the reference count for the managed pointer.
	*
	* Mostly used internally, therefor beware of wrong usage that might
	* lead to memory leaks.
	*/
	void retain() {
		++( referenceCount_->referenceCount_ );
	}


	/**
	* Helper constructor used by the friend cast functions.
	*/
	shared_device_pointer( T* _data, SharedPointerReferenceCount* _referenceCount ) : data_( _data ), referenceCount_( _referenceCount ) {
		retain();
	}


	/**
	* Helper function for the friend function @c static_pointer_cast.
	*
	* It is needed because two @c shared_device_pointer classes with different
	* template parameters grand friendship to each other while a friend
	* function only is the friend of one of both classes and doesn't get
	* access to the special constructor needed for casts of the other
	* class.
	*/
	template< typename U >
	shared_device_pointer< U > static_pointer_cast() const {
		return shared_device_pointer< U >( static_cast< U* >( get() ), referenceCount_ );
	}

	/**
	* Helper function for the friend function @c const_pointer_cast.
	*
	* It is needed because two @c shared_device_pointer classes with different
	* template parameters grand friendship to each other while a friend
	* function only is the friend of one of both classes and doesn't get
	* access to the special constructor needed for casts of the other
	* class.
	*/
	template< typename U >
	shared_device_pointer< U > const_pointer_cast() const {
		return shared_device_pointer< U >( const_cast< U* >( get() ), referenceCount_ );
	}

	/**
	* Helper function for the friend function @c dynamic_pointer_cast.
	*
	* It is needed because two @c shared_device_pointer classes with different
	* template parameters grand friendship to each other while a friend
	* function only is the friend of one of both classes and doesn't get
	* access to the special constructor needed for casts of the other
	* class.
	*/
	template< typename U >
	inline shared_device_pointer< U > dynamic_pointer_cast() const {
		U* castPointer = dynamic_cast< U* >( get() );

		if ( 0 == castPointer ) {
			return shared_device_pointer< U >();
		}

		return shared_device_pointer< U >( castPointer, referenceCount_ );
	}


private:
	pointer data_;
	SharedPointerReferenceCount* referenceCount_;
}; // class shared_device_pointer


template< typename T, typename U >
bool operator==( shared_device_pointer< T > const& lhs, shared_device_pointer< U > const& rhs );

template< typename T, typename U >
bool operator!=( shared_device_pointer< T > const& lhs, shared_device_pointer< U > const& rhs );

template< typename T >
void swap( shared_device_pointer< T >& lhs, shared_device_pointer< T >& rhs );




template< typename T, typename U >
bool operator==( shared_device_pointer< T > const& lhs, shared_device_pointer< U > const& rhs ) {
	return lhs.get() == rhs.get();
}

// template< typename T, typename U >
//     bool operator==( shared_device_pointer< T > const& lhs, U const* rhs ) {
//         return lhs.get == rhs;
//     }

// template< typename T, typename U >
//     bool operator==( T const* lhs, shared_device_pointer< U > const& rhs ) {
//         return lhs == rhs.get();
//     }


template< typename T, typename U >
bool operator!=( shared_device_pointer< T > const& lhs, shared_device_pointer< U > const& rhs ) {
	return !( lhs == rhs );
}

// template< typename T, typename U >
//     bool operator!=( shared_device_pointer< T > const& lhs, U const* rhs ) {
//         return !( lhs == rhs );
//     }

// template< typename T, typename U >
//     bool operator!=( T const* lhs, shared_device_pointer< U > const& rhs ) {
//         return !( lhs == rhs );
//     }



template< typename T >
void swap( shared_device_pointer< T >& lhs, shared_device_pointer< T >& rhs ) {
	lhs.swap( rhs );
}


} // namespace cupp


#endif
