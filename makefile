run%:
	@pushd $* && make run && popd

c%:
	@pushd $* && make && popd


