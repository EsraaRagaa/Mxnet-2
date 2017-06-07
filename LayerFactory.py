import mxnet as mx
import mxpth
def convFactory(data,num_filter,kernel,strid = (1,1),pad = (0,0) ,name = None, suffix = ''):
    conv = mx.sym.Convolution(data = data,num_filter = num_filter,kernel = kernel,strid = strid,pad = pad,name = 'conv_%%'%(name,suffix))
    bn = bn = mx.sym.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    act = mx.sym.Activation(data=bn, act_type='relu', name='relu_%s%s'%(name,suffix))
    pooling = mx.sym.Pooling(data = act, pool_type = 'max', kernel = (2,2), strid = (2,2), name = 'pooling_%s%s'%(name,suffix))

def fcFactory(data,)