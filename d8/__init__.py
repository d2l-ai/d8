__version__ = "0.0.2"

from . import core
# from . import object_detection
# from . import image_classification
# from . import tabular_classification

import logging
logging.basicConfig(format='[d8:%(filename)s:L%(lineno)d] %(levelname)-6s %(message)s')
logging.getLogger().setLevel(logging.INFO)
