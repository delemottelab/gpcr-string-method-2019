from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import time
from Queue import Queue

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


class JobExecutor(object):
    """
    Submits jobs and checks if they have finished.
    It ensures that not more than a certain number of jobs are running at the same time
    """

    def __init__(self, id, pool_size, sleep_time=10, logging_time=3000, resubmission_time=10000, append_mode=False,
                 callbacks=None):
        super(JobExecutor, self).__init__()
        self.id = id
        self.logger = logging.getLogger(id)
        self.logging_time = logging_time
        self.sleep_time = sleep_time
        self.pool_size = pool_size
        self.queue = Queue()
        self.active_jobs = []
        self.finished_jobs = []
        self.resubmission_time = resubmission_time
        self.append_mode = append_mode
        self.callbacks = [] if callbacks is None else callbacks

    def add(self, job):
        if job in self.queue.queue:  # queue.queue is a list
            self.logger.warn("Job %s already in queue. Not adding it again", job.id)
            return
        self.on_job_queued(job)
        self.queue.put(job)

    def stop(self):
        current_count = len(self.active_jobs) + self.queue.qsize()
        if current_count > 0:
            raise Exception(
                "There are still %s active jobs. Are you sure you're ready to change the iteration?" % current_count)
        for job in self.active_jobs:
            job.stop()

    def run(self):
        """
        Wait until there are no jobs in queue and all jobs' done() method has returned True.
        """
        finished = False
        counter = 0
        # last_running_count = len(self.active_jobs)
        # last_finished_count = len(self.finished_jobs)
        # resubmitted_count = 0
        # max_jobs_to_resubmit = 6
        while not finished:
            active_jobs = []
            for job in self.active_jobs:
                if job.done():
                    self.on_job_finished(job)
                else:
                    active_jobs.append(job)
            while len(active_jobs) < self.pool_size and self.queue.qsize() > 0:
                job = self.queue.get()
                if self.append_mode and job.done():
                    self.logger.debug("Job %s was already completed from before. Not starting it again", job.id)
                    self.on_job_finished(job, already_finished=True)
                else:
                    self.on_job_began(job)
                    job.submit()
                    active_jobs.append(job)
            self.active_jobs = active_jobs
            finished = len(active_jobs) == 0 and self.queue.empty()
            if not finished:
                time.sleep(self.sleep_time)
            if counter * self.sleep_time % self.logging_time == 0:
                self.logger.debug(
                    "%s jobs have been submitted and are still in progress. %s jobs have finished. %s jobs are in queue",
                    len(self.active_jobs),
                    len(self.finished_jobs),
                    self.queue.qsize())
            # if self.resubmission_time > 0 and counter * self.sleep_time % self.resubmission_time == 0:
            #     """CODE TO RESUBMIT JOBS IF NECESSARY"""
            #     finished_count = len(self.finished_jobs)
            #     running_count = len(self.active_jobs)
            #     nothing_changed = last_finished_count == finished_count and last_running_count == running_count
            #     if nothing_changed and max_jobs_to_resubmit - resubmitted_count >= running_count:
            #         self.logger.warn(
            #             "Nothing changed for a long time. Some jobs might have failed. Will try to resubmit jobs: %s",
            #             [j.id for j in self.active_jobs])
            #         for j in self.active_jobs:
            #             self.add(j)
            #             resubmitted_count += 1
            #     last_finished_count = finished_count
            #     last_running_count = running_count
            counter += 1
        self.logger.info("Finished. All simulations completed")

    def on_job_finished(self, job, already_finished=False):
        self.logger.debug("Job %s finished. %s new jobs will be submitted", job.id, len(job.child_jobs))
        self.finished_jobs.append(job)
        for cj in job.child_jobs:
            child_ready = True
            for p in cj.parent_jobs:
                if not p.done():
                    child_ready = False
                    break
            if child_ready:
                self.add(cj)
        for cb in self.callbacks:
            cb.on_job_finished(job, already_finished=already_finished)

    def on_job_began(self, job):
        for cb in self.callbacks:
            cb.on_job_began(job)

    def on_job_queued(self, job):
        for cb in self.callbacks:
            cb.on_job_queued(job)


class FinishedJobExecutor(object):
    """
    Does not submit jobs, raises exception if job is not already done only delegates jobs. To use
    """

    def __init__(self, id, callbacks=None):
        super(FinishedJobExecutor, self).__init__()
        self.callbacks = [] if callbacks is None else callbacks
        self.queue = Queue()
        self.id=id
        self.active_jobs = []
        self.logger = logging.getLogger(id)

    def add(self, job):
        if job in self.queue.queue:  # queue.queue is a list
            # self.logger.warn("Job %s already in queue. Not adding it again", job.id)
            return
        self.on_job_queued(job)
        self.queue.put(job)

    def stop(self):
        current_count = len(self.active_jobs) + self.queue.qsize()
        if current_count > 0:
            raise Exception(
                "There are still %s active jobs. Are you sure you're ready to change the iteration?" % current_count)
        for job in self.active_jobs:
            job.stop()

    def run(self):
        """
        Wait until there are no jobs in queue and all jobs' done() method has returned True.
        """
        while self.queue.qsize() > 0:
            job = self.queue.get()
            if job.done():
                # self.logger.debug("Job %s was already completed from before. Not starting it again", job.id)
                self.on_job_finished(job, already_finished=True)
            else:
                raise Exception("Job {} not done".format(job.id))
        self.logger.info("Finished. All simulations completed")

    def on_job_finished(self, job, already_finished=False):
        # self.logger.debug("Job %s finished. %s new jobs will be submitted", job.id, len(job.child_jobs))
        for cj in job.child_jobs:
            self.add(cj)
        for cb in self.callbacks:
            cb.on_job_finished(job, already_finished=already_finished)

    def on_job_began(self, job):
        for cb in self.callbacks:
            cb.on_job_began(job)

    def on_job_queued(self, job):
        for cb in self.callbacks:
            cb.on_job_queued(job)
