/* LD_PRELOAD shim: report IPC-imported peer pointers as belonging to the
 * current device, so torch's from_blob device check passes in flux's
 * cudaipc_create_tensor_list. Only attribute REPORTING changes; data paths
 * (memcpy, kernels, NCCL - which statically links cudart) are untouched. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>

struct cudaPointerAttributes_v2 {
  int type;            /* cudaMemoryType */
  int device;
  void *devicePointer;
  void *hostPointer;
};

typedef int cudaError_t_i;
typedef cudaError_t_i (*real_fn_t)(struct cudaPointerAttributes_v2 *, const void *);
typedef cudaError_t_i (*getdev_fn_t)(int *);

cudaError_t_i cudaPointerGetAttributes(struct cudaPointerAttributes_v2 *attr, const void *ptr) {
  static real_fn_t real = NULL;
  static getdev_fn_t getdev = NULL;
  if (!real) {
    real = (real_fn_t)dlsym(RTLD_NEXT, "cudaPointerGetAttributes");
    getdev = (getdev_fn_t)dlsym(RTLD_DEFAULT, "cudaGetDevice");
  }
  cudaError_t_i err = real(attr, ptr);
  if (err == 0 && attr->type == 2 /* cudaMemoryTypeDevice */ && getdev) {
    int cur = -1;
    if (getdev(&cur) == 0 && cur >= 0 && attr->device != cur)
      attr->device = cur;  /* peer IPC import: lie so from_blob accepts it */
  }
  return err;
}
