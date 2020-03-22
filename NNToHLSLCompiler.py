import numpy as np
import tensorflow as tf
import math

def compileDenseLayerToHLSL(layer : tf.keras.layers.Dense, functionName, activations):
    '''
    Compiles a specific dense layer to an HLSL funcion.
    Input vectors are split in several float4 vectors, and a remainer
    Dense matrix weights are decomposed as float4x4 matrices and remainers.
    Matrix multiplication is implemented using Strassen's method
    '''
    w = layer.get_weights()
    M = w[0] # dense weights
    W = w[1] # bias weights
    
    i = [] # input variables
    it = [] # input types
    input_size = len(M) # number of rows of the matrix
    remain = input_size
    index = 0
    while remain > 0:
        varSize = min(4, remain)
        i.append("i"+str(index))
        it.append("float"+str(varSize))
        remain -= varSize
        index += 1

    o = [] # output variables
    ot = [] # output types
    output_size = len(M[0]) # number of columns of the matrix
    remain = output_size
    index = 0
    while remain > 0:
        varSize = min(4, remain)
        o.append("o"+str(index))
        ot.append("float"+str(varSize))
        remain -= varSize
        index += 1
    
    code = ""
    code += "void "+functionName+" ("\
            +','.join("in "+t+" "+varName for t, varName in zip(it, i)) +","    \
            +','.join("out "+t+" "+varName for t, varName in zip(ot, o))        \
            +") { \n"

    m = [] # matrices variables
    mt = [] # matrices types
    mv = [] # matrices values

    row = 0
    remain_rows = input_size
    while remain_rows > 0:
        col = 0
        remain_cols = output_size
        rows = min(4, remain_rows)
        while remain_cols > 0:
            cols = min(4, remain_cols)
            m.append("m_"+str(row)+"_"+str(col))
            mt.append("float"+str(rows)+"x"+str(cols))
            mv.append([M[i][j] for i in range(row*4, row*4+rows) for j in range(col*4, col*4 + cols)])
            remain_cols -= cols
            col += 1
        remain_rows -= rows
        row += 1

    # add matrix definitions to code
    for mName, mType, mValues in zip(m, mt, mv):
        code += mType +" "+mName +" = "+mType+" ("+','.join([str(v)+"f" for v in mValues])+"); \n"

    b = [] # bias vectors
    bt = [] # bias types
    bv = [] # bias values
    index = 0
    remain = output_size
    while remain > 0:
        varSize = min(4, remain)
        b.append("w"+str(index))
        bt.append("float"+str(varSize))
        bv.append([W[i] for i in range(index*4, index*4 + varSize)])
        remain -= varSize
        index += 1
    
    # add bias definitions to code
    for bName, bType, bValues in zip(b, bt, bv):
        code += bType + " " + bName + " = "+bType+" ("+','.join([str(v)+"f" for v in bValues])+"); \n"

    # add output assignaments
    index = 0
    for oVar,oType in zip(o,ot):
        code += oVar+" = " + activations[layer.activation] +"(" +'+'.join(["mul("+i[x]+","+m[x * math.ceil(output_size/4)+index]+")" for x in range(0, len(i))])+" + "+b[index]+"); \n"
        index += 1

    code += "}\n"
    return code

def compileLinearActivation(vecType):
    return " "+vecType+" linearActivation("+vecType+" x){ return x; }\n"

def compileSigmoidActivation(vecType):
    return " "+vecType+" sigmoidActivation("+vecType+" x){ return 1.0f / (1.0f + exp(-x)); }\n"

def compileSoftPlusActivation(vecType):
    return " "+vecType+" softplusActivation("+vecType+" x){ return log(1 + exp(x)); }\n"

def compileTanHActivation(vecType):
    return " "+vecType+" tanhActivation("+vecType+" x){ return tanh(x); }\n"

def compileRELUActivation(vecType):
    return " "+vecType+" reluActivation("+vecType+" x){ return max(0, x); }\n"

def compileActivation(activation):
    compileFunction = None
    functionName = "unknown"
    if activation == tf.keras.activations.linear:
        compileFunction = compileLinearActivation
        functionName = "linearActivation"

    if activation == tf.keras.activations.sigmoid:
        compileFunction = compileSigmoidActivation
        functionName = "sigmoidActivation"

    if activation == tf.keras.activations.softplus:
        compileFunction = compileSoftPlusActivation
        functionName = "softplusActivation"

    if activation == tf.keras.activations.tanh:
        compileFunction = compileTanHActivation
        functionName = "tanhActivation"
    
    if activation == tf.keras.activations.relu:
        compileFunction = compileRELUActivation
        functionName = "reluActivation"

    code = ""
    # compile all functions possible overloads
    for i in range(1, 5):
        code += compileFunction("float"+str(i))+"\n"
    return (code, functionName)

def compileModelToHLSL(model, modelName):
    code = ""

    activations = {}
    layerLengths = []
    layerNames  = []
    index = 0
    for l in model.layers:
        if index == 0: # add input layer
            layerLengths.append(l.input_shape[1]) # initial input size
        layerLengths.append(l.output_shape[1]) # output of layer index and input of layer index+1
        layerName = modelName+"_layer_"+str(index)
        layerNames.append(layerName)

        if not ( l.activation in activations ):
            activationCode, name = compileActivation(l.activation)
            activations[l.activation] = name
            code += activationCode

        # add layer function
        code += compileDenseLayerToHLSL(l, layerName, activations)
        code += "\n"
        index += 1

    # create variables for neurons
    l_n = [] # neuron vector list by layer
    l_nt = [] # neuron vector types

    index = 0
    for ll in layerLengths:
        remain = ll
        n = []
        nt = []
        varIndex = 0
        while remain > 0:
            varSize = min(4, remain)
            n.append("n_"+str(index)+"_"+str(varIndex))
            nt.append("float"+str(varSize))
            remain -= varSize
            varIndex += 1
        l_n.append(n)
        l_nt.append(nt)
        index += 1

    # append model main function to code
    # function signature
    code += "void "+modelName+"(float _input["+str(layerLengths[0])+"], out float _output["+str(layerLengths[len(layerLengths)-1])+"]) { \n"
    # neuron vector definitions
    for n, nt in zip(l_n, l_nt):
        for nVar, nType in zip(n, nt):
            code += nType +" "+nVar+" = ("+nType+")0;\n"

    # retrieve values from input
    for i in range(0, layerLengths[0]):
        inputIndex = int(i/4)
        inputComponent = i % 4
        code += l_n[0][inputIndex]+"["+str(inputComponent)+"] = _input["+str(i)+"];\n"

    # calls for each layer
    for i in range(0, len(layerLengths)-1):
        code += layerNames[i]+"("+ ",".join(l_n[i]) + ", "+",".join([""+vn for vn in l_n[i+1]]) + "); \n"

    lastLayer = len(layerLengths)-1
    # set values to output
    for i in range(0, layerLengths[lastLayer]):
        outputIndex = int(i/4)
        outputComponent = i % 4
        code += "_output["+str(i)+"] = "+l_n[lastLayer][outputIndex]+"["+str(outputComponent)+"];\n"

    code += "}\n"
    return code

