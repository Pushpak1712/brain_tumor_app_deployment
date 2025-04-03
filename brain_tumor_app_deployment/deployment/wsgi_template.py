import os
from flask import Flask

# WSGI Application
# File: /var/www/yourusername_pythonanywhere_com_wsgi.py
# This file contains the WSGI configuration required to serve up your
# web application at http://yourusername.pythonanywhere.com/
# It works by setting the variable 'application' to a WSGI handler of some
# description.

# +++++++++++ FLASK +++++++++++
# Flask works like any other WSGI-compatible framework, with the app object
# being passed to the WSGI handler.

path = '/home/yourusername/brain_tumor_app'
if path not in sys.path:
    sys.path.append(path)

from app import app as application  # noqa

# The PythonAnywhere WSGI server uses this to serve the application
# application = app
