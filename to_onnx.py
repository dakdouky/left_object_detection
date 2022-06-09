from models.FCCDN import FCS
import torch 
import numpy as np

# model = FCS(num_band=3, use_se = True)
import os 
dev = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = dev
print("Using GPU: ", dev) 
model_name = 'saved_models/model_epoch_FCS.mdl'

torch_model = torch.load(model_name)

torch_model.eval()

batch_size = 1 

x = torch.randn(batch_size, 6, 256, 320, dtype = torch.float32 , requires_grad=True).cuda()
#x2 = torch.randn(batch_size, 1, 224, 224, requires_grad=True)

torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "change_detection.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

import onnx

onnx_model = onnx.load("change_detection.onnx")
onnx.checker.check_model(onnx_model)


import onnxruntime

ort_session = onnxruntime.InferenceSession("change_detection.onnx", 
                                           providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

