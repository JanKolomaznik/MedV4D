#ifndef DEFINITION_MACROS_H
#define DEFINITION_MACROS_H

#define BINSTREAM_WRITE_MACRO( STREAM, VARIABLE ) \
	STREAM.write( (char*)&VARIABLE, sizeof(VARIABLE) );

#define BINSTREAM_READ_MACRO( STREAM, VARIABLE ) \
	STREAM.read( (char*)&VARIABLE, sizeof(VARIABLE) );


#define SIMPLE_GET_METHOD( TYPE, NAME, PARAM_NAME ) \
	TYPE Get##NAME ()const{ return PARAM_NAME ; }

#define SIMPLE_SET_METHOD( TYPE, NAME, PARAM_NAME ) \
	void Set##NAME ( TYPE value ){ PARAM_NAME = value; }
		
#define SIMPLE_GET_SET_METHODS( TYPE, NAME, PARAM_NAME ) \
	SIMPLE_GET_METHOD( TYPE, NAME, PARAM_NAME ) \
	SIMPLE_SET_METHOD( TYPE, NAME, PARAM_NAME )

//TODO - move
#define PROHIBIT_COPYING_OF_OBJECT_MACRO( ClassName ) \
	ClassName( const ClassName& ); \
	ClassName& \
	operator=( const ClassName& ); 

#define SMART_POINTER_TYPEDEFS( TYPE ) \
	typedef std::shared_ptr< TYPE > Ptr; \
	typedef std::shared_ptr< const TYPE > ConstPtr; \
	typedef boost::weak_ptr< TYPE > WPtr; \
	typedef boost::weak_ptr< const TYPE > ConstWPtr;

#endif /*DEFINITION_MACROS_H*/
