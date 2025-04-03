# Brain Tumor Classification Web App Deployment Guide

This document provides instructions for deploying the Brain Tumor Classification web application to PythonAnywhere.

## Application Overview

This Flask web application provides:
- User authentication
- MRI scan upload and processing
- Brain tumor classification using a deep learning model
- Explainable AI visualizations (LIME and SHAP)

## Deployment Steps

### 1. PythonAnywhere Account Setup
- Log in to your PythonAnywhere account
- Navigate to the Web tab

### 2. Upload and Extract Files
- Upload the provided ZIP file to your PythonAnywhere account
- Extract the files to a directory (e.g., brain_tumor_app)

### 3. Virtual Environment Setup
- Open a Bash console
- Create a virtual environment:
  ```
  mkvirtualenv --python=/usr/bin/python3.10 brain_tumor_env
  ```
- Activate the environment:
  ```
  workon brain_tumor_env
  ```
- Install requirements:
  ```
  cd ~/brain_tumor_app
  pip install -r requirements.txt
  ```

### 4. Web App Configuration
- In the Web tab, create a new web app
- Select "Manual configuration" and Python 3.10
- Set the source code directory to your app directory
- Set the working directory to the same
- Configure the WSGI file using the provided wsgi.py template
- Set up static files mapping:
  - URL: /static/ -> Directory: /home/yourusername/brain_tumor_app/static
  - URL: /uploads/ -> Directory: /home/yourusername/brain_tumor_app/uploads
- Set the virtual environment path to your created environment

### 5. Final Steps
- Reload your web app
- Access your application at yourusername.pythonanywhere.com

## Troubleshooting
- Check error logs in the Web tab
- Ensure all directories exist and have proper permissions
- Verify that all required packages are installed correctly

## Login Credentials
- Username: doctor
- Password: doctor123
