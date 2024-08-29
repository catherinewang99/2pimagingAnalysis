import setuptools
import os

loc = os.path.abspath(os.path.dirname(__file__))

#with open(os.path.join(loc,"../","README.md"),"r") as fh:
#    long_description = fh.read()

setuptools.setup(
        name = "alm_2p",
        version = "0.0.1",
        author = "Catherine Wang and Taiga Abe",
        author_email = "taigaabe@stanford.edu",
        description = "tools developed by Catherine for 2p imaging in ALM", 
        long_description = "",
        long_description_content_type = "test/markdown", 
        url = "https://github.com/cellistigs/alm_2pimagingAnalysis",
        packages = setuptools.find_packages(),
        include_package_data=True,
        package_data={},
        classifiers = [
            "License :: OSI Approved :: MIT License"],
        python_requires=">=3.6",
        )


