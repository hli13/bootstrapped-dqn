import torch
import torch.autograd as autograd
import torch.nn as nn

Variable = autograd.Variable

class DQN(nn.Module):

    def __init__(self,num_actions=3):

        super(DQN, self).__init__()
        
        self.num_actions = num_actions
        self.main = nn.Sequential(
                nn.Conv2d(4, 32, 8, 4, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 4, 2, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 512, 7, 4, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, num_actions, 1, 1, 0)
		)
        
#        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
#        self.relu = nn.ReLU(inplace=True)
#        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
#        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)       
#        self.conv4 = nn.Conv2d(64, 512, kernel_size=7, stride=4, padding=0)
#        self.conv5 = nn.Conv2d(512, 3, kernel_size=1, stride=1, padding=0)
        
        

    def forward(self, x):
        out = self.main(x).squeeze()
        
#        x = self.conv1(x)
#        x = self.relu(x)
#        x = self.conv2(x)
#        x = self.relu(x)
#        x = self.conv3(x)
#        x = self.relu(x)
#        x = self.conv4(x)
#        x = self.relu(x)
#        x = self.conv5(x)
#        out = x.squeeze()
        
        return out
