import sys, time, torch
mode = sys.argv[1]  # 'eager' | 'inductor'
if len(sys.argv) > 2 and sys.argv[2] == 'import_vllm_omni':
    import vllm_omni  # noqa
    print('[note] imported vllm_omni')
from diffusers import AutoencoderKLWan
dev='xpu:0'
vae=AutoencoderKLWan().eval().to(dev, torch.float32)
lm=torch.tensor(vae.config.latents_mean).view(1,-1,1,1,1).to(dev)
lis=(1.0/torch.tensor(vae.config.latents_std)).view(1,-1,1,1,1).to(dev)
x=torch.zeros(1,3,33,352,640,device=dev,dtype=torch.bfloat16)
def core(inp):
    with torch.amp.autocast(dtype=torch.bfloat16, device_type='xpu'):
        h=vae._encode(inp); mu,_=h.chunk(2,dim=1); return (mu-lm)*lis
if mode=='inductor':
    fn=torch.compile(core, backend='inductor')
else:
    fn=core
# COLD: first ever call (pays compile+codegen+JIT for inductor; kernel JIT for eager)
torch.xpu.synchronize(); t=time.perf_counter()
with torch.no_grad(): fn(x)
torch.xpu.synchronize(); cold=(time.perf_counter()-t)*1000
# 2nd call
t=time.perf_counter()
with torch.no_grad(): fn(x)
torch.xpu.synchronize(); second=(time.perf_counter()-t)*1000
# WARM steady (mean of 8 after a few more warmups)
for _ in range(4):
    with torch.no_grad(): fn(x)
torch.xpu.synchronize()
ts=[]
for _ in range(8):
    t=time.perf_counter()
    with torch.no_grad(): fn(x)
    torch.xpu.synchronize(); ts.append((time.perf_counter()-t)*1000)
warm=sum(ts)/len(ts)
print(f'MODE={mode} COLD_MS={cold:.1f} SECOND_MS={second:.1f} WARM_MEAN_MS={warm:.1f} WARM_MIN={min(ts):.1f}')
