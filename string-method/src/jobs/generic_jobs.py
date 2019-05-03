from __future__ import absolute_import, division, print_function

import logging
import os
import subprocess
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


class AbstractJob(object):
    """Wrapper object for jobs, for example bash executions"""

    def __init__(self, id, type=""):
        super(AbstractJob, self).__init__()
        self.id = id
        self.logger = logging.getLogger("Job")
        self.child_jobs = []
        self.parent_jobs = []
        self.submission_count = 0
        self.type = type

    def add_child_job(self, child_job):
        self.child_jobs.append(child_job)
        child_job.parent_jobs.append(self)

    def add_child_jobs(self, child_jobs):
        for cj in child_jobs:
            self.add_child_job(cj)

    def submit(self):
        """Submit this job to som executor, for example a slurm queue or directly to bash"""
        self.submission_count += 1
        self.logger.warn("submit() not implemented")

    # def resubmit(self):
    #     """Submit the job again. By default simply a new submit"""
    #     self.submit()

    def stop(self):
        """Stops the job"""
        self.logger.warn("stop() not implemented")

    def done(self):
        """Returns true when the job has finished and has shutdown"""
        self.logger.warn("done() not implemented")
        return False


class BashJob(AbstractJob):
    """Call bash commands"""

    def __init__(self, id, args, done_file=None, type=""):
        super(BashJob, self).__init__(id, type=type)
        self.args = [str(a) for a in args]
        self.done_file = done_file
        self._process = None

    def submit(self):
        # self.logger.debug("calling bash jobs %s with args: %s", self.id, self.args)
        self.submission_count += 1
        process = subprocess.Popen(self.args)
        if self.done_file is None:
            self._process = process
            # Else don't save a reference to the process since it will generate zombie jobs if the Job object is not garbage collected properly

    def done(self):
        if self.done_file is None:
            if self._process is None:
                return True
            elif self._process.poll() is not None:
                # allow it to be garbage collected to avoid zombie processes
                # see https://stackoverflow.com/questions/2760652/how-to-kill-or-avoid-zombie-processes-with-subprocess-module
                self._process = None
                return True
            return False
        else:
            return os.path.exists(self.done_file)
