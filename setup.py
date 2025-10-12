"""
File Name: setup.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: GenAI
Latest Version: <<projectversion>>
Relative Path: /setup.py
File Created: Sunday, 12th October 2025 10:35:36 am
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 12th October 2025 1:46:13 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2025 - 2025, Panyi Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import json
import subprocess
import logging
from setuptools import setup, find_packages

logging.basicConfig()
log = logging.getLogger(__file__)

# Automatically get release version
version = (
    subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()[1:]  # remove the first character 'v'
)
# get only the version release number in case push is attached
version = version.split("-")[0] if "-" in version else version

INSTALL_LIST = [
    "torch",
    "transformers",
]

EXCLUDE_LIST = [
    "tests",
    "Dockerfiles",
    "example",
    "experiments",
    "archive",
    "data",
    "docs",
    ".github",
    "build",
    "dist",
]

SETUP_REQUIRES = [
    "setuptools",
]

SETUP_ARGS = {
    "name": "InsurGenAI",
    "version": version,
    "author": "Panyi Dong",
    "url": "https://github.com/PanyiDong/GenAI",
    "author_email": "panyid2@illinois.edu",
    "description": "GenAI toolkit for insurance applications.",
    "license": "MIT",
    # "cmdclass": {"build_ext": build_ext},
    "ext_modules": [],
}
EXT_MODULES = []

DATA_LIST = ["data/*", "example/*"]

EXTRA_DICT = {}


def setup_package():
    # global dep_list

    setup(
        packages=find_packages(
            exclude=EXCLUDE_LIST,
        ),
        package_dir={"InsurGenAI": "InsurGenAI"},
        include_package_data=True,
        package_data={"InsurGenAI": DATA_LIST},
        platforms=["Linux", "Windows"],
        long_description_content_type="text/markdown",
        python_requires=">=3.11",
        install_requires=INSTALL_LIST,
        extras_require=EXTRA_DICT,
        zip_safe=False,
        setup_requires=SETUP_REQUIRES,
        **SETUP_ARGS,
    )


# prepare for package.json file
def get_package():
    result = {}
    result["name"] = SETUP_ARGS["name"]
    result["author"] = {
        "name": SETUP_ARGS["author"],
        "email": SETUP_ARGS["author_email"],
    }
    result["repository"] = {
        "type": "git",
        "url": SETUP_ARGS["url"],
    }
    result["main"] = "main.py"
    result["private"] = False
    result["version"] = SETUP_ARGS["version"]
    result["description"] = SETUP_ARGS["description"]
    result["install_requires"] = INSTALL_LIST

    if os.path.exists("package.json"):
        os.remove("package.json")

    with open("package.json", "w") as f:
        json.dump(result, f, indent=4)


def main():

    # setup package
    setup_package()
    log.info(
        "{}-{} installation succeeded.".format(
            SETUP_ARGS["name"], SETUP_ARGS["version"]
        )
    )


if __name__ == "__main__":
    main()
    get_package()
