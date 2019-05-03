from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import os
import shutil

logger = logging.getLogger("FileMigrator")


def replace_filenames_in_path(path, oldtext, newtext):
    if os.path.exists(path):
        for f in os.listdir(path):
            if oldtext in f:
                newname = f.replace(oldtext, newtext)
                logger.debug("Moving %s to %s", path + f, path + newname)
                if os.path.exists(path + newname):
                    raise Exception("File " + path + newname + " already exists.")
                shutil.move(path + f, path + newname)


def migrate_files(simu_dir):
    for i in reversed(range(100)):
        for p in reversed(range(100)):
            for s in reversed(range(100)):
                replace_filenames_in_path(simu_dir + "/%s/%s/" % (i, p), "_swarm%s" % s, "s%s" % s)
            replace_filenames_in_path(simu_dir + "/%s/%s/" % (i, p), "iter%s_point%s" % (i, p), "i%sp%s" % (i, p))


if __name__ == "__main__":
    migrate_files("../gpcr/.string_simu")
