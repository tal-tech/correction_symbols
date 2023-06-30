LIBDIR = -Wl,-rpath=./lib\
	 -L./lib -lpthread -lstdc++ -lregpg -lopencv_imgproc -lopencv_imgcodecs -lopencv_core\

	
CPPFLAGS = -Wall -Winline -pipe -ffast-math -D_LINUX_64_ -DNO_DLL -rdynamic -mavx -mavx2 -msse 
INCLUDEDIR = -I. -I./ -I./include -I./include/opencv2


define mkObjDir
    @ test -d $(1) || mkdir -p $(1)
endef

define mkGitInfo
	@echo `git log | head -3` > output/version.txt
	@echo "------" >> output/version.txt
	@echo `git diff` >> output/version.txt
endef

GCC := g++ -std=gnu++11 -w -fopenmp
NVCC := /usr/local/cuda/bin/nvcc

OBJDIR := obj

TARGET1 := test_demo
TARGET2 := test_demo_debug


COREOBJ :=

COREOBJ_DEBUG = $(patsubst %.o,%_debug.o,$(COREOBJ))

OBJ1 = $(addprefix $(OBJDIR)/, $(COREOBJ) ./demo_reg.o)
OBJ2 = $(addprefix $(OBJDIR)/, $(COREOBJ_DEBUG) ./demo_reg_debug.o)

all: $(TARGET1) $(TARGET2)
	#rm -rf output
	#mkdir -p output
	#mv ${TARGET1}  /output/
	#mv ${TARGET2}  /output/

$(TARGET1) : $(OBJ1) 
	$(GCC) -o $@ $^ $(LIBDIR) $(INCLUDEDIR) 
$(TARGET2) : $(OBJ2) 
	$(GCC) -g -o $@ $^ $(LIBDIR) $(INCLUDEDIR)


$(OBJDIR)/%.o : %.cpp
	@ test -d $(OBJDIR) || mkdir -p $(OBJDIR)
	$(call mkObjDir,$(dir $@))
	$(GCC) -O2 $(CPPFLAGS) -c $< -o $@ $(INCLUDEDIR)

$(OBJDIR)/%_debug.o : %.cpp
	@ test -d $(OBJDIR) || mkdir -p $(OBJDIR)
	$(call mkObjDir,$(dir $@))
	$(GCC) $(CPPFLAGS) -g -c $< -o $@ $(INCLUDEDIR)

clean:
	rm -rf ./obj
	rm -rf $(TARGET1)
	rm -rf $(TARGET2)
