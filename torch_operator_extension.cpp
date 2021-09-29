#include <torch/extension.h>

at::Tensor my_custom_operator_mul2(at::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("custom_mul2", &my_custom_operator_mul2, "custom_mul2");
}
