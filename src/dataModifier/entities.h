#ifndef ENTITIES_H
#define ENTITIES_H

/**
 * Dictionary of names
 */
namespace M4D {
namespace DataModifier {

struct PatientInfo
{
  std::string patName;
  std::string patSex;
  std::string born;
};

struct StudyInfo
{
  std::string date;
};

}
}

#endif
