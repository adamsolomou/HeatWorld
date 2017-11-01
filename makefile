CPPFLAGS += -isystem include
CPPFLAGS += -W -Wall
CPPFLAGS += -std=c++11
CPPFLAGS += -O3

LDLIBS += -framework OpenCL

#LDLIBS += -lOpenCL

all : bin/make_world bin/render_world bin/step_world bin/aes414/step_world_v1_lambda bin/aes414/step_world_v3_opencl bin/aes414/step_world_v4_double_buffered bin/aes414/step_world_v5_packed_properties

bin/% : src/%.cpp src/heat.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

bin/aes414/% : src/aes414/%.cpp src/heat.cpp
		mkdir -p $(dir $@)
		$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) -framework OpenCL

test : bin/make_world bin/step_world bin/aes414/step_world_v1_lambda bin/aes414/step_world_v2_function
	@mkdir -p out/
	@echo "******** Make_world with step_world ********"
	bin/make_world 10 0.1 | bin/step_world 0.1 1 > out/world_with_step_world.out
	@echo "******** Make_world with step_world_v1_lambda ********"
	bin/make_world 10 0.1 | bin/aes414/step_world_v1_lambda 0.1 1 > out/world_with_step_world_v1_lamda.out
	@echo "******** Make_world with step_world_v2_function ********"
	bin/make_world 10 0.1 | bin/aes414/step_world_v2_function 0.1 1 > out/world_with_step_world_v2_function.out
	@echo "******** Make_world with step_world_v3_opencl ********"
	bin/make_world 10 0.1 | bin/aes414/step_world_v3_opencl 0.1 1 > out/world_with_step_world_v3_opencl.out
	@echo "******** Make_world with step_world_v4_double_buffered ********"
	bin/make_world 10 0.1 | bin/aes414/step_world_v4_double_buffered 0.1 1 > out/world_with_step_world_v4_double_buffered.out
	@echo "******** Comparing different worlds ********"
	@diff out/world_with_step_world.out out/world_with_step_world_v1_lamda.out; 
	@if [ $$? -eq 0 ] ; then echo "* no differences for v1 *"; fi
	@diff out/world_with_step_world.out out/world_with_step_world_v2_function.out; 
	@if [ $$? -eq 0 ] ; then echo "* no differences for v2 *"; fi
	@diff out/world_with_step_world.out out/world_with_step_world_v3_opencl.out; 
	@diff out/world_with_step_world.out out/world_with_step_world_v4_double_buffered.out; 
	@if [ $$? -eq 0 ] ; then echo "* no differences for v4 *"; fi;



