DATASET=../dataset/histo
BUILD=build

build: kernel.cu
	@mkdir -p $(BUILD)
	nvcc kernel.cu -o $(BUILD)/p1.exe

run: build
	@$(BUILD)/p1.exe $(DATASET)/input.raw

