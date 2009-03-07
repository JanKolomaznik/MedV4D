#ifndef DICOMREADER_H_
#define DICOMREADER_H_

#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkGDCMImageIO.h"

#include "itkIntegration/itkFilter.h"

typedef int16 PixelType;
typedef itk::Image<PixelType, 3> InternalImageType;
typedef itk::ImageSeriesReader< InternalImageType > ReaderType;

namespace M4D
{
namespace ITKIntegration
{

class DICOMReader
{
public:
	DICOMReader()
	{
		reader = ReaderType::New();
	}
	
	template< typename MedvedImage>
	static void ReadImage(MedvedImage &medImage, const std::string &path)
	{
	  // create IO object that can handle DICOM
	  typedef itk::GDCMImageIO       ImageIOType;
	  ImageIOType::Pointer dicomIO = ImageIOType::New();
	 
	  reader->SetImageIO( dicomIO );

	  // reading DICOM issues
	  typedef itk::GDCMSeriesFileNames NamesGeneratorType;
	  NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

	  nameGenerator->SetUseSeriesDetails( true );
	  nameGenerator->AddSeriesRestriction("0008|0021" );

	  nameGenerator->SetDirectory( path.c_str() );

	  typedef std::vector< std::string >    SeriesIdContainer;
	  
	  try
	  {
	    // read the specified directory end find out what DICOM series are within
		// note: there should be only ONE !!
	    const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
	    if(seriesUID.size() > 1)
	    {
	    	std::cout << "There should be only ine serie within specified directory !!";
	    }
	    
	    std::cout << std::endl << "The directory: " <<  path  << std::endl;
	    std::cout << "contains the following DICOM Series: ";
	   
	    SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
	    SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
	    while( seriesItr != seriesEnd )
	      {
	      std::cout << seriesItr->c_str() << std::endl;
	      seriesItr++;
	      }

	    typedef std::vector< std::string >   FileNamesContainer;
	    FileNamesContainer fileNames;

	    // get the file names of files that compose the serie
	    fileNames = nameGenerator->GetFileNames( seriesUID[0].c_str() );
	    // and set them to reader
	    reader->SetFileNames( fileNames );
	    
	    // update the reader to load the image
	    reader->Update();
	    
	    CopyITKToMedvedImage< InternalImageType, MedevedImage >(
	    		*reader->GetOutput(), medImage) );
	    		
	  } catch (itk::ExceptionObject &ex) {
		std::cout << "Nejde nacist DIka:" << std::endl;
		std::cout << ex << std::endl;
	  }
	}
private:
	ReaderType::Pointer reader;
};

}
}
#endif /*DICOMREADER_H_*/
