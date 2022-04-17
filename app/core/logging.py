import logging

# example: 2022-02-17 23:27:47,031 INFO tensorizer.py(96645) - Loading fairseq-2.7b/fairseq-2.7b.model
FORMAT = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) - %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

