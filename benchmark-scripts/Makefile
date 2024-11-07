# Copyright © 2024 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
.PHONY: consolidate python-test python-integration python-coverage

ROOT_DIRECTORY ?= results

init-packages:
	pip3 install -r requirements.txt

consolidate: init-packages
	python3 consolidate_multiple_run_of_metrics.py --root_directory $(ROOT_DIRECTORY)/ --output $(ROOT_DIRECTORY)/summary.csv

python-test:
	python -m coverage run -m unittest benchmark_test.py stream_density_test.py

python-integration:
	python -m coverage run -m unittest benchmark_integration.py

# for more up-to-date coverage, run python unit test first
python-coverage: python-test
	coverage report -m

# this one is just used for easier deployment of changes on benchmark-scripts
# to loss-prevention LOCALLY without reaching out to github
# assuming loss-prevention/ repo is already git cloned before and is
# in the same parallel directory structure as performance-tools/ repo
copy-to-loss-prevention:
	cp -r ./ ../../loss-prevention/performance-tools/benchmark-scripts/
