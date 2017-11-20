DATASET=../dataset/pictures
BUILD=build

build: kernel.cu
	@mkdir -p $(BUILD)
	nvcc kernel.cu -o $(BUILD)/2.exe

run: build
	@$(BUILD)/2.exe $(DATASET)/input.ppm $(DATASET)/output.ppm

