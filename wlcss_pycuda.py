import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void wlcss_cuda_kernel(int32_t *d_mss, int32_t *d_mss_offsets, int32_t *d_ts, int32_t *d_ss, int32_t *d_tlen, int32_t *d_toffsets, int32_t *d_slen, int32_t *d_soffsets, int32_t *d_params){
    
    int params_idx = threadIdx.x;
    int template_idx = blockIdx.x;
    int stream_idx = blockIdx.y;
    
    int t_len = d_tlen[template_idx];
    int s_len = d_slen[stream_idx];
    
    int t_offset = d_toffsets[template_idx];
    int s_offset = d_soffsets[stream_idx];
    
    int d_mss_offset = d_mss_offsets[params_idx*gridDim.x*gridDim.y+template_idx*gridDim.y+stream_idx];
    
    int32_t *tmp_window = new int32_t[(t_len + 2)]();

    int32_t *t = &d_ts[t_offset];
    int32_t *s = &d_ss[s_offset];

    int32_t *mss = &d_mss[d_mss_offset];

    int32_t reward = d_params[params_idx*3];
    int32_t penalty = d_params[params_idx*3+1];
    int32_t accepteddist = d_params[params_idx*3+2];

    int32_t tmp = 0;

    for(int32_t j=0;j<s_len;j++){
        for(int32_t i=0;i<t_len;i++){
            int32_t distance = abs(s[j]-t[i]);
            if (distance <= accepteddist){
                tmp = tmp_window[i]+reward;
            } else{
                tmp = max(tmp_window[i]-penalty*distance,
                            max(tmp_window[i+1]-penalty*distance,
                                tmp_window[t_len+1]-penalty*distance));
            }
            tmp_window[i] = tmp_window[t_len+1];
            tmp_window[t_len+1] = tmp;
        }
        tmp_window[t_len] = tmp_window[t_len+1];
        mss[j] = tmp_window[t_len+1];
        tmp_window[t_len+1] = 0;
    }
    delete [] tmp_window;

}
""")


def compute_wlcss(templates, streams, params):
    wlcss_pycuda = mod.get_function("wlcss_cuda_kernel")

    h_t = templates
    h_s = streams
    h_params = np.array(params).astype(np.int32)

    num_templates = len(h_t)  # Num block on X
    num_streams = len(h_s)  # Num block on Y
    num_params_sets = len(h_params)  # Num thread per block

    h_tlen = np.array([len(t) for t in h_t]).astype(np.int32)
    h_toffsets = np.cumsum(h_tlen).astype(np.int32)
    h_toffsets = np.insert(h_toffsets[0:-1], 0, 0)

    h_slen = np.array([len(s) for s in h_s]).astype(np.int32)
    h_soffsets = np.cumsum(h_slen).astype(np.int32)
    h_soffsets = np.insert(h_soffsets[0:-1], 0, 0)

    h_ts = np.array([item for sublist in h_t for item in sublist]).astype(np.int32)  # Template as numpy array
    h_ss = np.array([item for sublist in h_s for item in sublist]).astype(np.int32)  # Stream as numpy array

    h_mss = np.zeros((len(h_ss) * num_params_sets * num_templates)).astype(np.int32)
    d_mss = drv.mem_alloc(h_mss.nbytes)
    drv.memcpy_htod(d_mss, h_mss)
    h_mss_offsets = np.cumsum(np.tile(h_slen, num_params_sets * num_templates)).astype(np.int32)
    h_mss_offsets = np.insert(h_mss_offsets, 0, 0)

    wlcss_pycuda(d_mss, drv.In(h_mss_offsets),
                 drv.In(h_ts), drv.In(h_ss),
                 drv.In(h_tlen), drv.In(h_toffsets),
                 drv.In(h_slen), drv.In(h_soffsets),
                 drv.In(h_params),
                 block=(num_params_sets, 1, 1), grid=(num_templates, num_streams))

    h_mss = np.empty_like(h_mss).astype(np.int32)
    drv.memcpy_dtoh(h_mss, d_mss)
    tmp_mss = np.array([h_mss[offset - 1] for offset in h_mss_offsets[1:]])
    mss = [np.reshape(np.ravel(x), (num_streams, num_templates), order='F') for x in
           np.reshape(tmp_mss, (num_params_sets, num_streams, num_templates))]
    return mss
