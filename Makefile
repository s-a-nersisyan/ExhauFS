setup_pip:
	python3 setup.py bdist_wheel

publish_pip:
	make setup_pip
	python3 -m twine upload dist/*


local_pip:
	make setup_pip
	pip3 install dist/ExhauFS-0.1-py3-none-any.whl --force-reinstall
