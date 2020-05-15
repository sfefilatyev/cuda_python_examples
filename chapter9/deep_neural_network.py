dense_eval_code='''
__global__ void dense_eval(int num_outputs, int num_inputs, int relu, int sigmoid, float* w, float* b, float* x, float* y, int batch_size, int w_t, int_bt, float delta)
'''
