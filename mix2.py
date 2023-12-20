
class Mix2(nn.Module):
    def __init__(self,m=-0.80):
        super(Mix2, self).__init__()
        w=torch.nn.Parameter(torch.FloatTensor([m]),requires_grad=True)
        w=torch.nn.Parameter(w,requires_grad=True)
        self.w=w
        self.mix_block=nn.Sigmoid()

    def forward(self,x):
        mix_factor=self.mix_block(self.w)
        out=x*mix_factor
