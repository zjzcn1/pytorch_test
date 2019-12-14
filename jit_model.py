import torch

class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))
        print(self.weight)

    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output

my_module = MyModule(10,20)
input = torch.randn(20)
print(input)
print(my_module.forward(input))
traced_script_module = torch.jit.script(my_module)

traced_script_module.save("traced_model.pt")