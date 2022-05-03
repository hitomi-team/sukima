import logging

FORMAT = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) - %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

