import setuptools

import os
print(os.path.curdir)
with open('README.md', "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="phm_diagnosis",
  version="1.0.4",
  author="qin_hai_ning",
  author_email="2364839934@qq.com",
  description="failure diagnosis",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/three-quark/phm-diagnosis.git",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: Apache Software License",
  "Operating System :: OS Independent",
  ],
)
