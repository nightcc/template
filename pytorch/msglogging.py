import logging
import logging.config
import time
import os


logger = logging.getLogger("app_cfg")

def config_logger(log_cfg_file, experiment_name, output_dir='logs'):
    timestr = time.strftime("%Y.%m.%d-%H%M%S")
    exp_full_name = timestr if experiment_name is None else experiment_name + '__' + timestr
    logdir = os.path.join(output_dir, exp_full_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_filename = os.path.join(logdir, exp_full_name + '.log')
    if os.path.isfile(log_cfg_file):
        logging.config.fileConfig(log_cfg_file, defaults={'logfilename':log_filename})

    msglogger = logging.getLogger()
    msglogger.logdir = logdir
    msglogger.log_filename = log_filename
    msglogger.info('Log file for this run: ' + os.path.realpath(log_filename))

    try:
        os.unlink('latest_log_file')
    except FileNotFoundError:
        pass
    try:
        os.unlink('latest_log_dir')
    except FileNotFoundError:
        pass
    try:
        os.symlink(logdir, "latest_log_dir")
        os.symlink(log_filename, "latest_log_file")
    except OSError:
        msglogger.debug("Failed to create symlinks to latest logs")
    return msglogger




