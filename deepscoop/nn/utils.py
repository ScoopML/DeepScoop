import logging

def configure_logging(name=None, logfile='main.log'):
	if name is None:
		name = __name__
	logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w',
						format='%(name)s - %(levelname)s - %(message)s')

def get_logger():
	return logging.getLogger(__name__)
