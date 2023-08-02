#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "malloc.h"
#include <stdlib.h>

#define Ceil(x, y) ((x+y-1)/y)

int main(void) {
	int a = Ceil(10, 3);
	printf("%d\n", a);
	return 0;
}