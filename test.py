import torch
import torch_operator_extension


if __name__ == '__main__':
    x = torch.ones(16, 16, dtype=torch.float32).cuda()
    out = torch_operator_extension.custom_mul2(x)
    assert torch.allclose(out, x * 2)
    print("PASSED")