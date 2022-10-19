import setuptools

with open("README-zh.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="phm_diagnosis",
  version="0.0.1",
  author="qin_hai_ning",
  author_email="2364839934@qq.com",
  description="failure diagnosis",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/three-quark/phm-diagnosis.git",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: Apache License 2.0",
  "Operating System :: OS Independent",
  ],
)
