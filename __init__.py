import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# registers the routes
# noinspection PyUnresolvedReferences
from . import uiapi



module_js_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")
NODE_CLASS_MAPPINGS = { }
WEB_DIRECTORY = "js"
