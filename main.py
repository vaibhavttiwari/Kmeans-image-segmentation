from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import click
import os
import warnings

warnings.filterwarnings("ignore")

def calculate_sd(x, y, im):
    mean = 0
    count = 0
    for i in range(x-2, x+3):
        for j in range(y-2, y+3):
            if(i >= 0 and i < im.shape[0] and j >= 0 and j < im.shape[1] and i is not x and j is not y):
                mean += im[i, j, 0] + im[i, j, 1] + im[i, j, 2]
                count += 1
    mean = mean/count
    #print(mean)
    
    sd = 0
    count = 0
    for i in range(x-2, x+3):
        for j in range(y-2, y+3):
            if(i >= 0 and i < im.shape[0] and j >= 0 and j < im.shape[1] and not (i is x and j is y)):
                sd += (im[i, j, 0] + im[i, j, 1] + im[i, j, 2] - mean)**2
                #print(i)
                #print(j)
                count += 1
             
    #print(sd)
    sd = math.sqrt(sd/count)
    
    return sd
    
def KCenter(P, k):
    c = []
    l = len(P)
    max_ind = random.randint(0, l-1)
    distance = [float("inf")]*(len(P)-1)
    c.append(P.pop(max_ind))
    
    for i in range(k-1):
        max_dist = float(0)
        for j in range(l-1-i):
            a = np.asarray(P[j])
            b = np.asarray(c[i])
            dist = np.linalg.norm(a-b)
            distance[j] = min(distance[j], dist)
            
            if distance[j]>=max_dist:
                max_dist=distance[j]
                max_ind=j
        c.append(P.pop(max_ind))
        del distance[max_ind]
        
    return c

def KMeans(n, data, n_iterations, threshold):
    #s = random.sample(range(0, data.shape[0]), n)
    #means = []
    
    #for i in range(n):
    #    means.append(data[s[i]])
        
    #means = np.asarray(means)
    means = np.asarray(KCenter(data.copy(), n))
    
    data = np.asarray(data)
    
    y = np.zeros((data.shape[0],))
    
    _ = 0
    epsilon = 1000
    
    while(_ < n_iterations and epsilon > threshold):
        avg = np.zeros(means.shape)
        count = np.zeros((means.shape[0],))
        
        for i in range(data.shape[0]):
            min_dist = 1000000
            min_index = 1000000
            for j in range(means.shape[0]):
                dist = np.linalg.norm(data[i]-means[j])
                
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
                    
            y[i] = min_index
            avg[min_index] += data[i]
            count[min_index] += 1
            
        for i in range(avg.shape[0]):
            avg[i] = avg[i]/count[i]

        epsilon = 0
        
        for i in range(n):
            epsilon = max(epsilon, np.linalg.norm(avg[i]-means[i]))
        
        print("Iteration : "+str(_)+" \tEpsilon : "+str(epsilon))     
        means = avg
        _ += 1
        
    return y

@click.command()
@click.option('--input_path',
              type=click.STRING,
              default=None,
              help='The path of the input image. This argument is mandatory.')
@click.option('--output_dir',
              type=click.STRING,
              default="",
              help='The directory where the output segmented image will be stored. Default : Current working directory.')
@click.option('--n_clusters',
              type=click.INT,
              default=4,
              help='The number of segments required in the image. Default : 4')
@click.option('--features',
              type=click.STRING,
              default='11111',
              help='Choose the feature vector from the following pool : R,G,B,POS,Texture. Eg: 11101 will use all the features except the position of pixels. Default : 11111')
@click.option('--n_iterations',
              type=click.INT,
              default=10,
              help='Maximum number of iterations for K-Means algorithm. Default : 10')
@click.option('--epsilon',
              type=click.FLOAT,
              default=0.5,
              help='Threshold for the cluster center distances between two consecutive iterations. Default : 0.5')
def main(input_path, output_dir, n_clusters, features, n_iterations, epsilon):

    print("\n-----------------Starting execution with following parameters---------------------\n")
    print("Input image path : " + str(input_path))
    print("Output directory : " + str(output_dir))
    print("Number of clusters : " + str(n_clusters))
    print("Feature vector : " + str(features))
    print("Maximum number of iterations : " + str(n_iterations))
    print("Epsilon : " + str(epsilon))
    
    print("\n")
    #if not os.path.isdir(ouptut_dir):
    #    os.makedirs(output_dir)
    
    if not len(features) is 5:
        print("Invalid features argument.")
        exit()
        
    print("Reading image...\n")
    im = Image.open(input_path)
    im = np.asarray(im)

    data = []
    sd = []
    avg = 0
    
    print("Processing Image...\n")
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            a = calculate_sd(i, j, im)
            avg += a
            sd.append(a)

    avg = avg/(im.shape[0]*im.shape[1])
    
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            _ = []
            if features[0] == '1':
                _.append((im[i, j, 0]-128))
            if features[1] == '1':
                _.append((im[i, j, 1]-128))
            if features[2] == '1':
                _.append((im[i, j, 2]-128))
    	    
            if features[3] == '1':
                _.append((i-im.shape[0])/im.shape[0])
                _.append((j-im.shape[1])/im.shape[1])
    	    
            if features[4] == '1':
                #print(str(i)+" "+str(j)+" "+str(im.shape[0]*i + j))
                _.append((sd[im.shape[1]*i + j]-avg))
            data.append(_)

    kmeans = KMeans(n_clusters, data, n_iterations, epsilon)

    ans = np.reshape(kmeans, (im.shape[0], im.shape[1]))

    print("\nSaving output image...")
    plt.imsave(output_dir+'ans.jpg', ans)

if __name__ == '__main__':
    main()
