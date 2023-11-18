import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function

        W1=self.parameters["W1"]
        b1=self.parameters["b1"]
        W2=self.parameters["W2"]
        b2=self.parameters["b2"]

        s1=torch.mm(x,W1.t())+b1    #Linear layer 1
        # Non linear function f
        if self.f_function=="relu":
          a1=torch.relu(s1)
        elif self.f_function=="sigmoid":
          a1=torch.sigmoid(s1)
        else:
          a1=s1
        
        s2= torch.matmul(a1,W2.t())+b2    #linear layer 2
        # Non linear function g
        if self.g_function=="relu":
          y_hat=torch.relu(s2)
        elif self.g_function=="sigmoid":
          y_hat=torch.sigmoid(s2)
        else:
          y_hat=s2
        
        self.cache['a1']=a1
        self.cache['s1']=s1
        self.cache['s2']=s2
        self.cache['x']=x
        self.cache['y_hat']=y_hat
        
        return y_hat

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function

        W1=self.parameters["W1"]
        b1=self.parameters["b1"]
        W2=self.parameters["W2"]
        b2=self.parameters["b2"]
        a1=self.cache['a1']
        s1=self.cache['s1']
        s2=self.cache['s2']
        x=self.cache['x']
        y_hat=self.cache['y_hat']
      
      # derivative of los wrt output of linear layer 2
        if self.g_function=="relu":
          dJds2=(dJdy_hat*(s2>0).float())
        elif self.g_function=="sigmoid":
          dJds2=(dJdy_hat*(y_hat*(1-y_hat)))
        else:
          dJds2=dJdy_hat

        dJdb2=torch.sum(dJds2,0)
        dJdW2=torch.mm(dJds2.t(),a1)
        dJda1=torch.mm(dJds2,W2)

        # derivtive of loss wrt output of linear layer 1
        if (self.f_function == "relu"):
            dJds1 = (dJda1*(s1 > 0).float())
        elif (self.f_function == "sigmoid"):
            dJds1 = (dJda1*(a1*(1-a1)))
        else:
            dJds1 = dJda1
       

        dJdb1 = torch.sum(dJds1, 0)
        dJdW1= torch.mm(dJds1.t(), x)
       
        self.grads['dJdb1']=dJdb1
        self.grads['dJdb2']=dJdb2
        self.grads['dJdW1']=dJdW1
        self.grads['dJdW2']=dJdW2

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
  
    J= (torch.sum((y-y_hat)**2))/(y.size(dim=0)*y.size(dim=1))
    dJdy_hat= (2*(y_hat-y))/(y.size(dim=0)*y.size(dim=1))
    return J, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
  
    J= torch.sum(y* torch.log(y_hat)+ (1-y)*torch.log(1-y_hat))/(-y.size(dim=0)*y.size(dim=1))
    dJdy_hat = (-1/(y.size(dim=0)*y.size(dim=1)))*(y/y_hat - (1-y)/(1-y_hat))
    return J, dJdy_hat
