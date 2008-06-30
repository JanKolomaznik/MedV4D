#ifndef JOB_MANAGER_H
#define JOB_MANAGER_H

#include <map>

namespace M4D
{
namespace CellBE
{

class JobManager
{
public:

  typedef std::map<JobID, ServerJob *> JobMap;
  JobMap m_jobs;

  inline ServerJob *FindJob( JobID id)
  {
    JobMap::iterator it = m_jobs.find(id);
    if( it == m_jobs.end() )
      throw ExceptionBase("Job not found");
    else
      return it->second;
  }

  inline void AddJob( ServerJob *j)
  {
    m_jobs.insert( JobMap::value_type(j->primHeader.id, j) );
  }

  inline void RemoveJob( JobID id)
  {
    JobMap::iterator found = m_jobs.find( id);
    if( found == m_jobs.end() )
      throw ExceptionBase("Job not found");
    else
      m_jobs.erase( found);
  }

};

}}
#endif