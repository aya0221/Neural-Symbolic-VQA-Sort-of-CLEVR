import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class BiggerConvInputModel(nn.Module):
    def __init__(self):
        super(BiggerConvInputModel, self).__init__()
        
        #CNN Layers for sort of clevr task from the original paper
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(256)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x



class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()
        
        #CNN Layers for sort of clevr task for a smaller network
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x


#FC layers used by RN and CNN_MLP
class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


#Base class with handles for training, testing, and saving model
class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss
        
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))


class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')
        
        self.conv = ConvInputModel()
        
        self.relation_type = args.relation_type
        
        #(number of filters per object(conv output)+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((24+2)*2+11, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1) # (64 x 25 x 24)
        
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2) #(64 x 25 x (24+2))
        

        # add question everywhere
        qst = torch.unsqueeze(qst, 1) # 64 x 1 x 11
        qst = qst.repeat(1, 25, 1) # 64 x 25 x 11
        qst = torch.unsqueeze(qst, 2) # 64 x 25 x 1 x 11

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (64 x 1 x 25 x (24+2))
        x_i = x_i.repeat(1, 25, 1, 1)  # (64 x 25 x 25 x (24+2))
        x_j = torch.unsqueeze(x_flat, 2)  # (64 x 25 x 1 x (24+2))
        x_j = torch.cat([x_j, qst], 3) # (64 x 25 x 1 x (24+2 + 11))
        x_j = x_j.repeat(1, 1, 25, 1)  # (64 x 25 x 25 x (24+2 + 11))
        
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64 x 25 x 25 x ((24+2)*2 + 11))
    
        # reshape for passing through network
        x_ = x_full.view(mb * (d * d) * (d * d), 63)  # (64*25*25x(2*26+11)) = (40.000, 63)
            
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(mb, (d * d) * (d * d), 256)

        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)


class BiggerRN(BasicModel):
    def __init__(self, args):
        super(BiggerRN, self).__init__(args, 'BiggerRN')
        
        self.conv = BiggerConvInputModel()
        
        self.relation_type = args.relation_type
        
        #(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((256+2)*2+11, 2000)

        self.g_fc2 = nn.Linear(2000, 2000)
        self.g_fc3 = nn.Linear(2000, 2000)
        self.g_fc4 = nn.Linear(2000, 2000)

        self.f_fc1 = nn.Linear(2000, 2000)
        self.f_fc2 = nn.Linear(2000, 1000)
        self.f_fc3 = nn.Linear(1000, 500)
        self.f_fc4 = nn.Linear(500, 100)
        self.f_fc5 = nn.Linear(100, 10) #final output length depends on the answer embedding (10 in this case)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 256 x 5 x 5)
        
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1) # (64 x 25 x 256)
        
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2) #(64 x 25 x (256 + 2))
        

        # add question everywhere
        qst = torch.unsqueeze(qst, 1) #(64,11) -> (64,1,11)
        qst = qst.repeat(1, 25, 1) # 64 x 25 x 11
        qst = torch.unsqueeze(qst, 2) #64 x 25 x 1 x 11

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # 64 x 1 x 25 x (256 + 2)
        x_i = x_i.repeat(1, 25, 1, 1)  # 64 x 25 x 25 x (256 + 2)
        x_j = torch.unsqueeze(x_flat, 2)  #  64 x 25 x 1 x (256 + 2)
        x_j = torch.cat([x_j, qst], 3) # 64 x 25 x 1 x (256 + 2 + 11)
        x_j = x_j.repeat(1, 1, 25, 1)  #  64 x 25 x 25 x (256 + 2 + 11)
        
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # 64 x 25 x 25 x ((256 + 2) + (256 + 2 + 11))
    
        # reshape for passing through network
        x_ = x_full.view(mb * (d * d) * (d * d), 2*(256 + 2) + 11)  # (64*25*25 x (2*258+11)) = (40000, 527)
            
        x_ = self.g_fc1(x_) # 64*25*25 x 2000
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_) # 64*25*25 x 2000
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_) # 64*25*25 x 2000
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_) # 64*25*25 x 2000
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(mb, (d * d) * (d * d), 2000) # 64 x 25*25 x 2000

        x_g = x_g.sum(1).squeeze() # 64 x 2000
        
        """f"""
        x_f = self.f_fc1(x_g) # 64 x 2000
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f) # 64 x 1000
        x_f = F.relu(x_f)
        x_f = self.f_fc3(x_f) # 64 x 500
        x_f = F.relu(x_f)
        x_f = self.f_fc4(x_f) # 64 x 100
        x_f = F.relu(x_f)
        x_f = self.f_fc5(x_f) # 64 x 10
        x_f = F.log_softmax(x_f, dim=1) # 64 x 10
        
        return x_f


class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv  = ConvInputModel()
        self.fc1   = nn.Linear(5*5*24 + 11, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
  
    def forward(self, img, qst):
        x = self.conv(img) # x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1) # (64 x 24 * 5 * 5)
        
        x_ = torch.cat((x, qst), 1)  # Concat question (64 x (24 * 5 * 5 + 11))
        
        x_ = self.fc1(x_) # (64 x 256)
        x_ = F.relu(x_) # (64 x 256)
        
        return self.fcout(x_) # (64 x 10)