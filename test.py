from models.beamforming import Net
import torch
model_params = {
    "stft_chunk_size": 192,
    "stft_pad_size": 96,
    "stft_back_pad": 0,
    "num_ch": 6,
    "D": 16,
    "L": 4,
    "I": 1,
    "J": 1,
    "B": 4,
    "H": 64,
    "E": 2,
    "local_atten_len": 50,
    "use_attn": False,
    "lookahead": True,
    "chunk_causal": True,
    "use_first_ln": True,
    "merge_method": "early_cat",
    "directional": True
}
device = torch.device('cpu') ##('cuda')
model = Net(**model_params).to(device)

num_chunk = 50
test_num = 10
chunk_size = model_params["stft_chunk_size"]
look_front = model_params["stft_pad_size"]
look_back = model_params["stft_back_pad"] #model_params["lookback"]
x = torch.rand(4, 6, look_back + chunk_size*num_chunk + look_front)
x = x.to(device)
x2 = x[..., :look_back + chunk_size*test_num + look_front]
inputs = {"mixture": x}
inputs2 = {"mixture": x2}
y = model(inputs, pad=False)['output']
y2 = model(inputs2, pad=False)['output']

print(x.shape, x2.shape, y.shape, y2.shape)
_id  = 3
check_valid = torch.allclose(y2[:, 0, :chunk_size*test_num], y[:, 0, :chunk_size*test_num], atol=1e-2 )
print((y2[:, 0, :chunk_size*test_num] - y[:, 0, :chunk_size*test_num]).abs().max())
print(check_valid)

next_state = None
for chunk_start in range(0, num_chunk, test_num):
    chunk_end = chunk_start + test_num
    x3 = x[..., chunk_size*chunk_start: look_back + chunk_size*chunk_end + look_front]
    inputs = {"mixture": x3}
    y3 = model(inputs, input_state=next_state, pad=False)
    output = y3['output']; next_state = y3['next_state']

    _output = output[:, :, :chunk_size*test_num]
    _y = y[:, :, chunk_size*chunk_start:chunk_size*chunk_end]
    check_valid = torch.allclose(_output, _y, atol=1e-2)
    print((_y[:] - _output[:]).abs().max(), check_valid)
    # break
# import matplotlib.pyplot as plt 
# plt.figure()
# plt.plot( y[_id, 0, :chunk_size*test_num].detach().numpy())
# plt.plot( y2[_id, 0, :chunk_size*test_num].detach().numpy(), linestyle = '--', color = 'r')
# plt.show()