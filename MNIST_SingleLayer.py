import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from sklearn.utils.extmath import softmax


seed = 7
np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_train)

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1. - x * x


def compute_multiclass_loss(Y, Y_hat):

    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1/m) * L_sum
    return L



def predict(model, x,y):
    W1, b1, W2, b2 = model['wh'], model['bh'], model['wout'], model['bout']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    predict = np.argmax(probs,axis=1)
    count=0
    for i in range(len(y)):
        if(predict[i]==y[i]):
            count+=1
    accuracy = count/len(y)
    accuracy = accuracy*100
    return accuracy,count

def build_model(X,Y,lr,epoch):
    np.random.seed(1)
    
    
    m=X.shape[0] #Number of Training examples
    w1= np.random.randn(n_x,n_h)*0.01
    b1 = np.zeros((1,n_h))
    w2 = np.random.randn(n_h,n_y)*0.01
    b2 = np.zeros((1,n_y))

    model = {}
    for i in range(epoch):
        #Forward Propogation
        hidden_layer_input1=np.dot(X,w1)
        z1=hidden_layer_input1 + b1
        a1= tanh(z1)
        output_layer_input1=np.dot(a1,w2)
        z2= output_layer_input1+ b2
        a2 = softmax(z2)
        
        #cost = compute_multiclass_loss(Y, a2)

        #print('Cost after Epoch'+ str(i+1)+' is '+ str(cost))
       
        #Backpropagation
        
        dZ2 = a2-Y         
        dW2 = (1./m) * np.matmul(a1.T,dZ2)
        db2 = (1./m) * np.sum(dZ2, axis=0, keepdims=True)
        
        
        #dA1 = np.matmul(dZ2, w2.T)
        Error_at_hidden_layer = np.dot(dZ2,w2.T)
        slope_hidden_layer = dtanh(a1)
        dZ1 = Error_at_hidden_layer * slope_hidden_layer
        #dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))        
        dW1 = (1./m) * np.matmul(X.T,dZ1)
        db1 = (1./m) * np.sum(dZ1, axis=0, keepdims=True)
        
        
        
        w2 = w2 - lr * dW2
        b2 = b2 - lr * db2
        w1 = w1 - lr * dW1
        b1 = b1 - lr * db1
        
        model = { 'wh': w1, 'bh': b1, 'wout': w2, 'bout': b2}
        pred,count = predict(model, X_test,y_test)
        if(i%10==0):
            print(i," ",pred)
    return model



epoch=200                      
lr=0.25                          
n_x = X_train.shape[1]
n_h = 20         
n_y = 10              

model = build_model(X_train,Y_train, lr, epoch)
print("\nNo of testing samples : ",len(y_test))

