DATASET=../dataset/vector
BUILD=build

build: kernel.cu
	@mkdir -p $(BUILD)
	nvcc kernel.cu -o $(BUILD)/1.exe

run: build
	@$(BUILD)/1.exe $(DATASET)/input0.raw $(DATASET)/input1.raw $(DATASET)/output.raw

