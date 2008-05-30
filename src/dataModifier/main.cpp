
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
#include <iostream>

#include "Common.h"
#include "entryContainer.h"

namespace fs = boost::filesystem;
using namespace std;
using namespace M4D::DataModifier;

typedef vector<fs::path> Paths;

Paths paths;

EntryContainer entryCont;

void SolveDir( fs::path & dirName)
{
  // Get all files in this dir
  // loop through them
  LOG( "Entering DIRECTORY: " << dirName);

  fs::directory_iterator end_iter;

  for ( fs::directory_iterator dir_itr( dirName );
        dir_itr != end_iter;
        ++dir_itr )
  {
    
    // if it is subdir, call itself on subdir
    if ( fs::is_directory( dir_itr->status() ) )
    {
      paths.push_back( *dir_itr);
    }
    else
    {
      entryCont.SolveFile( dir_itr->string(), dirName.string() );
    }
  }
}

///////////////////////////////////////////////////////////////////////

void PrintUsage( void)
{
  cout << "Usage:" << endl;
  cout << "dataModifier <path to images>" << endl;
}

///////////////////////////////////////////////////////////////////////

int
main( int argc, char** argv)
{
  fs::path full_path( fs::initial_path<fs::path>() );

  if ( argc > 1 )
    full_path = fs::system_complete( fs::path( argv[1], fs::native ) );
  else
    PrintUsage();

  // specification of hi, low date of studies
  if( argc > 2)
  {
    string s(argv[2]);
    if( s.compare( "--info") == 0 )
    {
      entryCont.infoOnly = true;
    }
    else
    {
      if( argc > 3)
      {
        entryCont.dateFrom = argv[2];
        entryCont.dateTo = argv[3];
      }
      else
        PrintUsage();
    }
  }

  // recursively (through queue) go through all files in subtree
  // of specified directory specified

  try {
    if ( fs::is_directory( full_path ) )
    {
      paths.push_back( full_path);
    }
    else // must be a file
    {
      entryCont.SolveFile( full_path.string(), "buddy" );
    }

    fs::path currPath;
    while( ! paths.empty() )
    {
      currPath = paths.back();
      paths.pop_back(); // remove this dir from queue

      SolveDir( currPath);
    }
    
  } catch( std::exception &ex) {
    LOG( ex.what());
  }

  // flush info
  {
    ofstream o("output.txt");
    entryCont.FlushMaps( o);
    o.close();
  }
  
  return 0;
}

///////////////////////////////////////////////////////////////////////