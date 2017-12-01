DATASET=../dataset/images
BUILD=build

build: kernel.cu
	@mkdir -p $(BUILD)
	nvcc kernel.cu -o $(BUILD)/p2.exe

run: build
	@$(BUILD)/p2.exe $(DATASET)/logo-insa-lyon.ppm

