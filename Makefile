# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined variable: $1$(if $2, ($2))))

.PHONY: release

release:
	@:$(call check_defined, version, The release version)
	git checkout main
	git tag -fa "v$(version)" -m "Release v$(version)"
	git push origin "v$(version)" --force
