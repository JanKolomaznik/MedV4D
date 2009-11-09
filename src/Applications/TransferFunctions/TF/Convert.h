#ifndef CONVERT
#define CONVERT

#include <sstream>

template<typename From, typename To>
static To convert(const From &s){

    stringstream ss;
    To d;
    ss << s;
    if(ss >> d)
	{
        return d;
	}
    /*
    cerr << endl
         << "error: conversion failed, used default" << endl;
    */
    return NULL;
}

#endif //CONVERT