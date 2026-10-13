PYTHON ?= python3

.PHONY: version-check audit python-check shell-check test check release

version-check:
	@$(PYTHON) -c 'import sys; assert sys.version_info >= (3, 10), "SkillScaffold requires Python >= 3.10"'

audit:
	$(PYTHON) scripts/audit_release.py

python-check:
	$(PYTHON) -m compileall -q benchmarks integrations framework scripts

shell-check:
	bash scripts/check_shell.sh

test:
	$(PYTHON) -m unittest discover -s tests -v

check: version-check audit python-check shell-check test

release:
	$(PYTHON) scripts/build_release.py
