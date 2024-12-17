# (1) Can you determine the mininum memory demand to execute each model for each batch size, when no data is allowed to
# be evicted from the GPU before a tensor is destroyed?
# (2) What is the distribution of the tensor size and lifetime? Plot the total size of the tensors against lifetime in milliseconds.
# (3) Plot a figure to show the distribution of the active/inactive time of tensors in each model. Explain the potential for data
# offloading with the figure.
# (4) If we can swap tensors off GPU memory when not used, what is the minimal memory requirement for executing each
# model (assuming data transfer takes no time)?

import os
import matplotlib.pyplot as plt

tensorList = {} # <TensorID, [Size, Global, Lifetime]>
kernelList = {} # <KernelID, [Name, Time, InputTensor, OutputTensor]>

def minimumLifetime():
    srtEnd = {}
    graphData = {} # <KernelId, Memory>
    for id, kern in kernelList.items():
        name, time, inputTensor, outputTensor = kern
        graphData[id] = 0
        for i in inputTensor:
            if i not in srtEnd:
                srtEnd[i] = [id, id]
            else:
                srtEnd[i][1] = id
        for i in outputTensor:
            if i not in srtEnd:
                srtEnd[i] = [id, id]
            else:
                srtEnd[i][1] = id

    for id, rng in srtEnd.items():
        for i in range(int(rng[0]), int(rng[1])+1):
            graphData[i] += tensorList[id][0]
    
    return graphData

def tensorSizeLifetime():
    srtEnd = {}
    graphData = [] # <Size, Lifetime(ms)>

    for id, kern in kernelList.items():
        name, time, inputTensor, outputTensor = kern
        for i in inputTensor:
            if i not in srtEnd:
                srtEnd[i] = [id, id]
            else:
                srtEnd[i][1] = id
        for i in outputTensor:
            if i not in srtEnd:
                srtEnd[i] = [id, id]
            else:
                srtEnd[i][1] = id
                
    for id, rng in srtEnd.items():
        for i in range(int(rng[0]), int(rng[1])+1):
            tensorList[id][2] += kernelList[i][1]

    for id, tensor in tensorList.items():
        graphData.append((tensor[0], tensor[2]))
    return graphData

def activeInactiveTime():
    srtEnd = {}
    graphData = {} # <KernelId, (Active, Inactive)>
    for id, kern in kernelList.items():
        name, time, inputTensor, outputTensor = kern
        exists = {}
        graphData[id] = [0, 0]
        for i in inputTensor:
            graphData[id][0] += time
            exists[i] = True
            if i not in srtEnd:
                srtEnd[i] = [id, id]
            else:
                srtEnd[i][1] = id
        for i in outputTensor:
            if i not in exists:
                graphData[id][0] += time
            if i not in srtEnd:
                srtEnd[i] = [id, id]
            else:
                srtEnd[i][1] = id

    for id, rng in srtEnd.items():
        for i in range(int(rng[0]), int(rng[1])+1):
            graphData[i][1] += kernelList[i][1]
    
    return graphData

def minimumMemory():
    srtEnd = {}
    graphData = {} # <KernelId, Memory>
    for id, kern in kernelList.items():
        name, time, inputTensor, outputTensor = kern
        graphData[id] = 0
        for i in inputTensor:
            graphData[id] += tensorList[i][0]
        for i in outputTensor:
            graphData[id] += tensorList[i][0]

    return graphData

def createGraph(x_name, y_name, graphData, title, filename, type, separate=False, AI=False):
    plt.clf()
    for key, value in graphData.items():
        if separate:
            plt.clf()
        if type == 'scatter':
            if AI:
                y_values = []
                y_values = [item[0] for item in value.values()]
                plt.scatter(value.keys(), y_values, label=f'Active', s=1)
                y_values = [item[1] for item in value.values()]
                plt.scatter(value.keys(), y_values, label=f'Inactive', s=1)
            else:
                x_values = [item[0] for item in value]
                y_values = [item[1] for item in value]
                plt.scatter(x_values, y_values, label=f'Batch {key}', s=10)
                plt.xscale('log')
        elif type == 'plot':
            plt.plot(value.keys(), value.values(), label=f'Batch {key}')
        
        if separate:
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.legend()
            plt.title(title + ' Batch ' + key)
            os.makedirs(os.path.dirname(filename + '_' + key + '.png'), exist_ok=True)
            plt.savefig(filename + '_' + key + '.png')

    if not separate:
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.legend()
        plt.title(title)
        os.makedirs(os.path.dirname(filename + '.png'), exist_ok=True)
        plt.savefig(filename + '.png')
        #plt.show()

def parseTensor(file):
    with open(file) as fp:
        for line in fp:
            parts = line.strip().split()
            id, size, glob = int(parts[0]), int(parts[1]), parts[2]
            tensorList[id] = [size, glob, 0]


def parseKernel(file):
    with open(file) as fp:
        for line in fp:
            parts = line.strip().split()
            id, name, time, inputTensor, outputTensor = int(parts[0]), parts[1], float(parts[2]), parts[3], parts[4]
            inputTensor = stripInOut(inputTensor)
            outputTensor = stripInOut(outputTensor)
            kernelList[id] = (name, time, inputTensor, outputTensor)

def stripInOut(arr):
    new_arr = []
    if(arr == '[]'):
        return new_arr
    stripInput = arr.strip().strip("[]")
    new_arr.extend(map(int, stripInput.split(',')))
    return new_arr

if __name__ == "__main__":
    wls = ['BERT', 'Inceptionv3', 'ResNet152', 'SENet154', 'VIT']
    batchSizes = ['128', '256', '384', '512', '640', '768', '1024', '1152', '1280', '1536']

    for wl in wls:
        grph = {}
        szLiGrph = {}
        minGrph = {}
        aITime = {}
        for batch in batchSizes:
            tensorList.clear()
            kernelList.clear()
            if os.path.exists('./results/' + wl + '/sim_input/' + batch + 'Kernel.info'):
                parseTensor('./results/' + wl + '/sim_input/' + batch + 'Tensor.info')
                parseKernel('./results/' + wl + '/sim_input/' + batch + 'Kernel.info')
                grph[batch] = minimumLifetime()
                szLiGrph[batch] = tensorSizeLifetime()
                minGrph[batch] = minimumMemory()
                aITime[batch] = activeInactiveTime()
            else:
                print("Path does not exist")

        createGraph('Kernel ID', 'Memory', grph, wl + ' Minimum Lifetime Memory Demand', './graphs/' + wl + '/' + wl + '_minLifetime', 'plot')
        createGraph('Size of Tensor (log)', 'Lifetime (ms)', szLiGrph, wl + ' Tensor Size Lifetime', './graphs/' + wl + '/' + wl + '_tensorSizeLifetime', 'scatter')
        createGraph('Kernel ID', 'Memory', minGrph, wl + ' Minimum Memory Demand', './graphs/' + wl + '/' + wl + '_minMemory', 'plot', True)
        createGraph('Tensor ID', 'Lifetime (ms)', aITime, wl + ' Active Inactive Time', './graphs/' + wl + '/' + wl + '_activeInactiveTime', 'scatter', True, True)