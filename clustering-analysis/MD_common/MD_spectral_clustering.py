import argparse
import sys
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-colorblind")
from numpy import linalg as LA
from sklearn import cluster, datasets
from scipy.optimize import minimize
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import SpectralEmbedding
from scipy.spatial.distance import squareform, pdist
from scipy.sparse.linalg import eigsh, lobpcg
from mpl_toolkits.mplot3d import Axes3D
from spherical_kmeans import SphericalKMeans
from sklearn.utils.graph import graph_shortest_path
from sklearn.metrics import silhouette_score
import networkx as nx
import os


class SpectralClustering:
    file_end_name = '';
    save_folder = '';
    distances = 0;
    validationDistances = 0;
    nDimensions = 1;
    
    def initialization(self,parser, input_args=None):
        parser.add_argument('-d','--distance_array',type=str,help='Input distances or input coordinates.',default='');        
        parser.add_argument('-id','--in_directory',type=str,help='Directory with input data (optional)',default='');
        parser.add_argument('-fe','--file_end_name',type=str,help='Output file end name (optional)', default='');
        parser.add_argument('-od','--out_directory',type=str,help='The directory where data should be saved (optional)',default='');
        parser.add_argument('-cdist','--compressed_distances',help='Flag for using precomputed distances (optional)',action='store_true');
        parser.add_argument('-nDims','--number_dimensions',type=float,help='Setting the dimension of the data (optional)',default=1);
        args = parser.parse_args() if input_args is None else parser.parse_args(input_args)
        self.file_end_name = args.file_end_name;
        self.save_folder = args.out_directory;
        print('Spectral clustering.');
        print('Reading distance matrix.');
        self.distances = np.loadtxt(args.in_directory+args.distance_array);        
        
        if args.compressed_distances:
            self.distances = squareform(self.distances);
        else:
            self.distances = squareform(pdist(self.distances));
        
        print(self.distances.shape);
        
        self.nDimensions = args.number_dimensions;
        return;
    
    def getKNNgraph(self, k):
        print('Construct graph with k = ' + str(k) + ' neighbors');
        nDist = self.distances.shape[0];
        
        # Construct kNN network
        adjacencyMatrix = np.zeros((nDist,nDist));
        distance_sort_ind = np.argsort(self.distances);
        
        for i in range(0, nDist):
            iConnected = distance_sort_ind[i,0:k];
            adjacencyMatrix[i,iConnected] = self.distances[i,iConnected];
            adjacencyMatrix[iConnected,i] = self.distances[i,iConnected];
        
        # Symmetric adjacency matrix
        adjacencyMatrix = 0.5 * (adjacencyMatrix + adjacencyMatrix.T);

        return adjacencyMatrix;

    def computeClusterPropensity(self, adjacencyMatrix):
        
        nNodes = len(adjacencyMatrix[::,0]);
        degree = float(np.count_nonzero(adjacencyMatrix))/float(nNodes);
        print('Actual graph degree: ' + str(degree));
        
        # Use networkx
        graph = nx.Graph(adjacencyMatrix);
        cluster_coeff = nx.average_clustering(graph);
        cluster_coeff_random = float(degree)/(float(nNodes)-1.0);
        
        print('Clustering coefficient: ' + str(cluster_coeff));
        print('dC = ' + str(cluster_coeff-cluster_coeff_random));
        return cluster_coeff-cluster_coeff_random;
        
    def getGraphMatrix(self, k_list):
        c_diff = -1;
        adjMat = [];
        k_opt = -1;
        for k in k_list:
            A = self.getKNNgraph(k);
            #cTmp = self.computeClusterPropensity(A);
            #if c_diff < cTmp:
            #c_diff = cTmp;
            adjMat = A;
            k_opt = k;
        
        affinityMatrix = graph_shortest_path(adjMat,directed=False);
        # Make similarity matrix
        #affinityMatrix = np.exp(-affinityMatrix**2/2);
        #for i in range(0,affinityMatrix.shape[0]):
        #    affinityMatrix[i,i] = 0;
        
        #affinityMatrix = -adjMat**2/2;
        print('Optimal k = ' + str(k_opt));
        return affinityMatrix;
            
    
    def computeSpectralEigenvectors(self, degree, affinity, n_components):
        print('Compute spectral eigenvectors');
        nDims = affinity.shape[0];

        D = 1/np.sqrt(degree)*np.eye(nDims);
        
        L = np.eye(nDims)-np.dot(np.dot(D,affinity),D);
        eigenvalues,eigenvectors = LA.eig(L);
        
        sort_ind = np.argsort(eigenvalues);
        X = eigenvectors[::,sort_ind[0:n_components]];
        
        for i in range(0,len(X[::,0])):
            X[i,::] /= np.linalg.norm(X[i,::]);
        
        return X;

    def getGraphFromAdjacencyMatrix(self, adjacencyMatrix):
        G=nx.Graph();

        for i in range(0,len(adjacencyMatrix[::,0])):
            for j in range(i+1,len(adjacencyMatrix[0,::])):
                if adjacencyMatrix[i,j] > 0:
                    G.add_edge(str(i),str(j),weight=adjacencyMatrix[i,j]);
        return G;

    def drawNetwork(self, adjacencyMatrix):
        # Add shortest paths/geodesic distances + isomap projection before plotting
        graph = nx.Graph(adjacencyMatrix);        
        pos = nx.spring_layout(graph);
        nx.draw_networkx_nodes(graph,pos,node_size=100);
        nx.draw_networkx_edges(graph,pos);
        return;
    

    def sphericalDistance(self, x1, x2):
        # Distance along arc: r*theta (with r = 1 on unit n-sphere). 
        # Get theta from theta=arccos(u'*v/(norm(u)*norm(v))).
        argument = np.dot(x1,x2)/(LA.norm(x1)*LA.norm(x2));
        if argument > 1:
            argument = 1;
        elif argument < -1:
            argument = -1;
        return np.arccos(argument);

    def sampleSphericalPoints(self, nPoints, nDimensions):
        points = np.random.randn(nPoints, nDimensions);
        for i in range(0,points.shape[0]):
                points[i,::] /= LA.norm(points[i,::]);
        #points[nDimensions-1,::] = np.abs(points[nDimensions-1,::]);
        return points;
    
    def qualityScore(self, x, centers):
        nDimensions = len(centers[::,0]);
        tmpDist = np.zeros(nDimensions);
        dist_fraction = np.zeros(len(x[::,0]));
        
        for i in range(0,len(x[::,0])):
            x1 = x[i,::];
            for j in range(0,nDimensions):
                tmpDist[j] = self.sphericalDistance(x1, centers[j,::]);
            
            tmpDist = np.sort(tmpDist);
            if tmpDist[0] < 1e-5:
                tmpDist[0] = 1e-5;
            
            dist_fraction[i] = tmpDist[1]/tmpDist[0];

        qualityScore = np.median(dist_fraction);
        return qualityScore;

    def normalizedQualityScore(self, x, centers):
        
        nDimensions = len(centers[::,0]);    
        nPoints = len(x[::,0]);    
        
        # Un-normalized Q score
        Q = self.qualityScore(x, centers);
        
        ## Compute the normalization factor based on randomly sampled points
        tmpQ = np.zeros(50);
        for i in range(0,len(tmpQ)):
            rand_points = self.sampleSphericalPoints(nPoints, nDimensions);
            
            # Cluster with spherical k-means
            kmeansObj = SphericalKMeans()#cluster.KMeans());
            kmeansObj.__init__(n_clusters=nDimensions,init='random');
            kmeansObj.fit(rand_points);
            
            # Get cluster centers
            rand_centers = kmeansObj.cluster_centers_;
            tmpQ[i] = self.qualityScore(rand_points, rand_centers);
        
        # Compute quality score for random points        
        Q_rand = np.mean(tmpQ);

        print('Q = ' + str(Q));
        print('Q_rand = ' + str(Q_rand));
        return Q/Q_rand;

    def silhouetteScore(self, x, cluster_indices):
        nPoints = x.shape[0];
        allDistances = np.zeros((nPoints,nPoints));
        dist_fraction = np.zeros(len(x[::,0]));
        
        for i in range(0,nPoints):
            x1 = x[i,::];
            for j in range(i+1,nPoints):
                allDistances[i,j] = self.sphericalDistance(x1, x[j,::]);
            
        allDistances += allDistances.T;

        qualityScore = silhouette_score(allDistances,labels=cluster_indices,metric='precomputed');
        print('Silhouette score: ' + str(qualityScore));
        return qualityScore;
    
    def saveData(self, points, cluster_indices, index):
        if self.save_folder[-1] != '/':
            self.save_folder += '/';
        
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder);

        np.savetxt(self.save_folder + 'cluster_indices_' + self.file_end_name + '.txt', cluster_indices[::,index]);
        np.savetxt(self.save_folder + 'sphere_points_' + self.file_end_name + '.txt', points);
        print('Data saved to files');
        return;

    def evaluateKDE(self,sigma, distances):
        variance = sigma**2;
        nTrainingPoints = float(self.validationDistances.shape[1]);
        # Dimension of self.validationDistances: [nValPoints x nTrainPoints]
        #print(nTrainingPoints*sigma**self.nDimensions);
        scalingFactor = -self.nDimensions*np.log(sigma*np.sqrt(2*np.pi));
        allBasisFunctions = np.exp(-self.validationDistances**2/(2*variance));
        tmpKDE = np.sum(allBasisFunctions,axis=1);
        logLikelihood = np.sum(np.log(tmpKDE)+scalingFactor);
        return logLikelihood,scalingFactor;
    
    def optimalSigma(self):
        print('Inferring Gaussian standard deviation.');
        maxSigma = np.sqrt(np.mean(self.distances**2));
        minSigma = 0.71*np.min(self.distances[self.distances > 0]);
        print('Min sigma = ' + str(minSigma));
        
        # Let 60 % of points (randomly chosen) be in training set
        nPoints = int(self.distances.shape[0]);
        nTrainingPoints = int(np.floor(0.8*nPoints));
        nValidationPoints = nPoints - nTrainingPoints;
        
        permutedPoints = np.random.permutation(nPoints);
        trainingIndices = permutedPoints[0:nTrainingPoints];
        validationIndices = permutedPoints[nTrainingPoints::];
        
        self.validationDistances = np.zeros((nValidationPoints,nTrainingPoints));
        for i in range(0,nTrainingPoints):
            for j in range(0,nValidationPoints):
                self.validationDistances[j,i] = self.distances[trainingIndices[i],validationIndices[j]];
        
        constr = ({'type': 'ineq','fun': lambda sigma: sigma-minSigma},{'type': 'ineq','fun' : lambda sigma: maxSigma-sigma})
        
        bestSigma = 1;
        bestLogLikelihood = 0;
        optResult = 0;
        for sigma in np.arange(minSigma,maxSigma,(maxSigma-minSigma)/100):
            initialGuess = np.random.rand(1)*maxSigma;
            tmpLoglikelihood,scalingFactor = self.evaluateKDE(sigma);#minimize(self.evaluateKDE,initialGuess,constraints=constr);
            
            #print tmpResult
            
            if bestLogLikelihood == 0 and not(np.isnan(tmpLoglikelihood)):
                #bestOpt = optResult.fun;
                bestLogLikelihood = tmpLoglikelihood;
                bestSigma = sigma;
                
            elif tmpLoglikelihood > bestLogLikelihood:
                print('Found better solution: ');
                print(tmpLoglikelihood);
                print(scalingFactor);
                #print tmpResult.fun
                #print bestOpt;
                #print tmpResult.x;
                #print initialGuess;
                #bestOpt = optResult.fun;
                bestLogLikelihood = tmpLoglikelihood;
                bestSigma = sigma;
        
        print('Inferred sigma: ' + str(bestSigma));
        return bestSigma;
    
    def KDE_density(self, sigma, distances):
        variance = sigma**2;
        nPoints = float(distances.shape[1]);
        scalingFactor = -self.nDimensions*np.log(sigma*np.sqrt(2*np.pi));
        allBasisFunctions = np.exp(-distances**2/(2*variance));
        tmpKDE = np.sum(allBasisFunctions,axis=1);
        logLikelihood = np.sum(log(tmpKDE) + scalingFactor);
        return logLikelihood, scalingFactor;

    def leaveOneOutSigma(self):

        maxSigma = np.sqrt(np.mean(self.distances**2));
        minSigma = np.min(self.distances[self.distances > 0]);

        #print minSigma;
        #print maxSigma;        
        
        nPoints = self.distances.shape[0];
        minDiff = 1000000000;
        bestSigma = minSigma;
        for sigma in np.arange(minSigma,maxSigma,(maxSigma-minSigma)/100):
            variance = sigma**2;
            projections = np.exp(-self.distances**2/(2*variance));
            
            total = 0;
            for i in range(nPoints):
                tmpDistances = self.distances[i,0:i-1];
                tmpDistances = np.append(tmpDistances,self.distances[i,i+1::]);
                tmpProj = projections[i,0:i-1];
                tmpProj = np.append(tmpProj,projections[i,i+1::]);
                prob_i = 1/(nPoints-1)*np.sum(tmpProj)+1e-5;
                K_i = np.sum(tmpProj*tmpDistances**2);
                total += 1/prob_i*K_i;
            
            total = np.log(total) - self.nDimensions*np.log(sigma*np.sqrt(2*np.pi))
            diff = (np.log(variance*(self.nDimensions*(nPoints-1)*nPoints)) - total)**2;
            
            if diff < minDiff:
                minDiff = diff;
                bestSigma = sigma;
        
        return bestSigma;

    def graphLaplacian(self, adjecencyMatrix):
        D = np.sum(adjecencyMatrix,axis=1);
        D_inv = np.diag(1/np.sqrt(D));
        laplacian = np.dot(D_inv,np.dot(adjecencyMatrix,D_inv));
        return laplacian;

    def cluster(self, number_clusters=np.array([2,3,4,5,6,7,8])):
        
        k_list = [4];
        nCompList = number_clusters;
        
        # Construct affinity matrix from interframe distances
        #A = self.getGraphMatrix(k_list);
        sigma = 1;#self.optimalSigma(); 
        distSort = np.sort(self.distances,axis=1);
        
        sigma = 0.7*np.mean(distSort[::,1]); #self.leaveOneOutSigma(); #
        print('Sigma = ' + str(sigma));
        
        dist_squared = np.multiply(self.distances,self.distances);
        variance = np.mean(dist_squared);
        
        A = np.exp(-dist_squared/(2*sigma**2));
        for i in range(0,A.shape[0]):
            A[i,i] = 0;        
    
        # First project onto a bunch of dimensions, then pick optimal dim/clusters
        print('Compute Laplacian');
        laplacian = self.graphLaplacian(A);
        print('Spectral embedding');
        eigenValues, eigenVectors = eigsh(laplacian,k=(nCompList[-1]+1));
        
        # Sort in descending order
        eigSortedInds = np.argsort(-eigenValues);
        eigenValues = eigenValues[eigSortedInds];
        proj = eigenVectors[::, eigSortedInds];
        
        #proj = SpectralEmbedding(n_components=20, eigen_solver='arpack',
        #                affinity='precomputed').fit(A).embedding_;
        print('Clustering in all dimensions');
    
        counter = 1;        
        qualityScores = np.zeros(len(nCompList));
        silhouetteScores = np.zeros(len(nCompList));
        eigenGaps = np.zeros(len(nCompList));
        BICs = np.zeros(len(nCompList));

        cluster_indices = np.zeros((A.shape[0],len(nCompList)));
        
        # Do clustering and vary the number of clusters
        for nComponents in nCompList:        
            
            print('Number of clusters: ' + str(nComponents));
            X = np.copy(proj[::,0:nComponents]);
            for i in range(0,len(X[::,0])):
                X[i,::] /= LA.norm(X[i,::]);
            
            
            # Cluster with spherical k-means
            kmeansObj = SphericalKMeans();
            kmeansObj.__init__(n_clusters=nComponents,init='random');
            kmeansObj.fit(X);
            
            # Get cluster indices and centers
            cl_ind = kmeansObj.labels_;
            cluster_indices[::,counter-1] = cl_ind+1;
            
            centers = kmeansObj.cluster_centers_;
            
            # Compute clustering quality score
            qualityScores[counter-1] = self.normalizedQualityScore(X, centers);
            #silhouetteScores[counter-1] = self.silhouetteScore(X, cl_ind);
            eigenGaps[counter-1] = eigenValues[nComponents-1]-eigenValues[nComponents];
            print('Eigen gap: ' + str(eigenGaps[counter-1]));
            #BICs[counter-1] = 0.5*nComponents*X.shape[1]*np.log(X.shape[0]);
            #print('BIC: ' + str(BICs[counter-1]));
            
            colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
            colors = np.hstack([colors] * 20)
            
    
            if X.shape[1] > 2 and X.shape[1] < 4:
                fig = plt.figure(counter);
                ax = fig.add_subplot(111,projection='3d',aspect='equal');
                ax.scatter(X[::,0],X[::,1],X[::,2],color=colors[cl_ind].tolist());
            elif X.shape[1] == 2:
                fig = plt.figure(counter);
                ax = fig.add_subplot(111,aspect='equal');
                ax.scatter(X[::,0],X[::,1],color=colors[cl_ind].tolist());
                plt.axis([-1, 1, -1, 1])
    
            counter += 1;
            
        silhouetteScores -= np.min(silhouetteScores);
        silhouetteScores /= np.max(silhouetteScores) if np.max(silhouetteScores) != 0 else 1;

        fig = plt.figure(counter);
        ax = fig.add_subplot(111);
        ax.plot(nCompList,qualityScores/np.max(qualityScores),marker='o', linewidth=2.0,label='SPECTRUS Q');
        #ax.plot(nCompList,silhouetteScores,marker='o',color='r',linewidth=2.0,label='Silhouette');
        ax.plot(nCompList,eigenGaps/np.max(eigenGaps),marker='o', linewidth=2.0,label='Eigen gap');
        #ax.plot(nCompList,BICs/np.max(BICs),marker='o',color='k',linewidth=2.0,label='BIC');
        #plt.title('Normalized quality scores')#: Pick number of clusters');
        plt.xlabel('# clusters');
        #plt.xticks(range(min(number_clusters),max(number_clusters)));
        plt.legend();
        plt.grid();
        plt.ylabel('Normalized score');
        
        # Decide on clusters
        # nClusters = int(np.round(plt.ginput(1)[0][0]));
        plt.tight_layout(pad=0.3)
        
        print("Saving spectrus graph svg and metadata in ", self.save_folder)
        plt.savefig(self.save_folder + "*/spectrus_score.svg")
        #Row 0 is the x-axis, row 1-2 are the quality scores
        spectrus_data = np.append(nCompList, qualityScores, axis=1)
        spectrus_data = np.append(spectrus_data, eigenGaps, axis=1)        
        spectrus_data.save(self.save_folder + "/spectrus_data")
        
        plt.show();
        print("Ready to read input?")
        nClusters = int(raw_input("Pick number of clusters:"))
        print("Number of clusters: " + str(nClusters));
        index = np.where(nCompList==nClusters)[0][0];

        X = np.copy(proj[::,0:nClusters]);
        for i in range(0,len(X[::,0])):
            X[i,::] /= np.linalg.norm(X[i,::]);

        self.saveData(X, cluster_indices, index);
        return self;



    def clusterFindSigma(self):
        
        k_list = [4];
        nCompList = np.array([2,3,4,5,6,7,8]);
        
        # Construct affinity matrix from interframe distances
        #A = self.getGraphMatrix(k_list);
        sigma = 1;#self.optimalSigma(); #self.leaveOneOutSigma();
        distSort = np.sort(self.distances,axis=1);
        
        sigma_min = 0.5*np.min(distSort[::,1]); #self.leaveOneOutSigma(); #
        print('Sigma min = ' + str(sigma_min));
        
        dist_squared = np.multiply(self.distances,self.distances);
        sigma_max = 0.5*np.max(distSort[::,1]);
        print('sigma max: ' + str(sigma_max));

        sigma_best = sigma_min;
        minDistortion = 100000;
        bestX = 0;

        for nComponents in nCompList:    
            print('Clustering in all dimensions');
            for sigma in np.arange(sigma_min,sigma_max,(sigma_max-sigma_min)/20):
                A = np.exp(-dist_squared/(2*sigma**2));
                for i in range(0,A.shape[0]):
                    A[i,i] = 0;        
                
                # First project onto a bunch of dimensions, then pick optimal dim/clusters
                print('Spectral embedding with sigma = '+str(sigma));
                laplacian = self.graphLaplacian(A);
                eigenValues, eigenVectors = LA.eig(laplacian);

                # Sort in descending order
                eigSortedInds = np.argsort(-eigenValues);
                eigenValues = eigenValues[eigSortedInds];
                proj = eigenVectors[:, eigSortedInds];
        
                #proj = SpectralEmbedding(n_components=20, eigen_solver='arpack',
                #                affinity='precomputed').fit(A).embedding_;
    
                counter = 1;        
                qualityScores = np.zeros(len(nCompList));
                silhouetteScores = np.zeros(len(nCompList));
                eigenGaps = np.zeros(len(nCompList));

                cluster_indices = np.zeros((A.shape[0],len(nCompList)));
        
                # Do clustering and vary the number of clusters
                X = np.copy(proj[::,0:nComponents]);
                
                print('Number of clusters: ' + str(nComponents));
                if sigma==sigma_min:
                    bestX = X;
                
                for i in range(0,len(X[::,0])):
                    X[i,::] /= LA.norm(X[i,::]);
            
                
                # Cluster with spherical k-means
                kmeansObj = SphericalKMeans(cluster.KMeans());
                kmeansObj.__init__(n_clusters=nComponents,init='random');
                kmeansObj.fit(X);
                
                # Get cluster indices and centers
                cl_ind = kmeansObj.labels_;
                
                clusterDistortion = kmeansObj.inertia_;
                
                if clusterDistortion < minDistortion:
                    minDistortion = clusterDistortion;
                    bestX = X;
                    sigma_best = sigma;
                    cluster_indices[::,counter-1] = cl_ind+1;
                    centers = kmeansObj.cluster_centers_;
                else:
                    break;
                    
            print('Chosen sigma: ' + str(sigma_best));
            # Compute clustering quality score
            qualityScores[counter-1] = self.normalizedQualityScore(X, centers);
            silhouetteScores[counter-1] = self.silhouetteScore(X, cl_ind);
            eigenGaps[counter-1] = eigenValues[nComponents-1]-eigenValues[nComponents];
            print('Eigen gap: ' + str(eigenGaps[counter-1]));
            
            colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
            colors = np.hstack([colors] * 20)
            
    
            if X.shape[1] > 2 and X.shape[1] < 4:
                fig = plt.figure(counter);
                ax = fig.add_subplot(111,projection='3d',aspect='equal');
                ax.scatter(X[::,0],X[::,1],X[::,2],color=colors[cl_ind].tolist());
            elif X.shape[1] == 2:
                fig = plt.figure(counter);
                ax = fig.add_subplot(111,aspect='equal');
                ax.scatter(X[::,0],X[::,1],color=colors[cl_ind].tolist());
                plt.axis([-1, 1, -1, 1])
    
            counter += 1;
            
        silhouetteScores -= np.min(silhouetteScores);
        silhouetteScores /= np.max(silhouetteScores);

        fig = plt.figure(counter);
        ax = fig.add_subplot(111);
        ax.plot(nCompList,qualityScores/np.max(qualityScores),marker='o',color='b',linewidth=2.0,label='SPECTRUS Q');
        ax.plot(nCompList,silhouetteScores,marker='o',color='r',linewidth=2.0,label='Silhouette');
        ax.plot(nCompList,eigenGaps/np.max(eigenGaps),marker='o',color='g',linewidth=2.0,label='Eigen gap');
        plt.title('Normalized quality scores: Pick number of clusters');
        plt.xlabel('# clusters');
        plt.ylabel('Score');
        
        # Decide on clusters
        nClusters = int(np.round(plt.ginput(1)[0][0]));
        print("Number of clusters: " + str(nClusters));
        index = np.where(nCompList==nClusters)[0][0];

        X = np.copy(proj[::,0:nClusters]);
        for i in range(0,len(X[::,0])):
            X[i,::] /= np.linalg.norm(X[i,::]);

        self.saveData(X, cluster_indices, index);
        plt.show();
        return;

# parser = argparse.ArgumentParser(epilog='Perform spectral clustering with spherical k-means on given data. Annie Westerlund 2017.');

# clustering = SpectralClustering();
# clustering.initialization(parser);
# clustering.cluster();
