import numpy as np 
import unittest
import time
import os
import cv2
import matplotlib.pyplot as plt

cur_path = os.path.dirname(__file__)

class Cluster_Labeling():
    def __init__(self, grid):
        # The Cluster_Labeling Class is used to find "clusters" in a 2d array (aka grid)
        # The grid is a 2d array of cells where each cell is either occupied or unoccupied 
        # and every group of cells that neighbor each other form a cluster.
        # Attributes:
        #       self.grid: is input occupany "grid"
        #       self.labels: a 2d np array, like the occupany grid, but the values are the cluster labels  
        #       self.assignments: a dictionary storing a list of all coords (r,c) of points in each cluster. {cluster: [(r1,c1),(r2,c2) ...]}
        #       self.cluster_sizes: a dictionary mapping clusters to num of points in the cluster
        #       self.sorted_clusters: a list of [(clusters,size) pairs in order of size ([more points ...... less points])
        #       self.cluster_centroids: a dictionary mapping centroid loc (item) to cluster (key) {cluster: (r_centroid, c_centroid) }
        #       self.cluster_name_int: this is the last name used for a cluster. Everytime a new cluster is made, we will use self.cluster_name_int++ as the name
        # # Inputs:
        #       grid (2d numpy array with 1 (or any other nonzero number) = occupied 0 = not occupied.) : The grid that we want to find clusters in.
        self.grid = grid
        self.labels = np.zeros(np.shape(self.grid))
        self.assignments = dict()
        self.cluster_sizes = dict()
        self.sorted_clusters = []
        self.cluster_centroids = dict()
        self.cluster_name_int = 0

        # Do cluster finding to update self.labels and self.assigments
        self.find_clusters()
        self.find_centroids()
        self.calc_cluster_sizes()
        self.rename_clusters()

    def process_new_labels(self,new_labels,new_cluster_centroids = None):
        # takes in a 2d array of a labels object as an input. Changes self.assignments, self.cluster sizes, sorted clusters, and cluster centroids dict
        # Input:
        #           new_labels: an np 2d array representing a labels object

        # initialize
        new_cluster_ids = []
        new_assignments = dict()
        new_cluster_sizes = dict()
        new_sorted_clusters = []
        new_cluster_name_int = 0

        # Check for all clusters
        new_cluster_name_int = np.max(new_labels) + 1
        for cluster_id in range(1,int(new_cluster_name_int)):
            row_coords, col_coords = np.where(new_labels == cluster_id)
            new_assignments[cluster_id] = list(zip(row_coords,col_coords))
            new_cluster_sizes[cluster_id] = len(new_assignments[cluster_id])
            new_sorted_clusters  = sorted(new_cluster_sizes.items(), key = lambda x : x[1], reverse = True)

        # set values
        self.labels = new_labels
        self.assignments = new_assignments
        self.cluster_sizes = new_cluster_sizes
        self.sorted_clusters = new_sorted_clusters
        self.cluster_name_int = new_cluster_name_int 

        # get cluster centroids
        if new_cluster_centroids is not None:
            self.cluster_centroids = new_cluster_centroids

        else:
            self.find_centroids()



    def union(self, cluster_a, cluster_b):
        # When 2 originally distinct clusters turns out to be neigbors, this function unions the clusters. This function does two things
        # 1) This func updates the labels so the the value of coord at cluster b is changed to the value for cluster a
        # 2) This updates the assigments dictionary so that the points assigned to key value for cluster 2 are moved to key values for cluster a. The cluster b key is deleted
        # # Inputs: 
        #       cluster_a, cluster_b: the names for the two clusters (the value in the labels cell / also the key name in assignments dict.)
        # # Output:
        #       None
        
        # update self.labels and self.assignments
        for r,c in self.assignments[cluster_b]:
            self.labels[r,c] = cluster_a
            self.assignments[cluster_a].append((r,c))

        # del cluster b
        del self.assignments[cluster_b]

    def rename_clusters(self):
        # After the process of finding clusters, cluster names are unbounded and sometimes skip integers. ie: (1,4,100...)
        # change cluster names to (1,2,3....)
        new_assignments = dict()
        new_cluster_sizes = dict()
        new_sorted_clusters = []
        new_cluster_centroids = dict()
        new_labels = np.zeros(np.shape(self.labels))

        if len(self.sorted_clusters) > 0:
            for i in range(len(self.sorted_clusters)):
                prev_cluster_name = self.sorted_clusters[i][0]
                new_cluster_name = i+1
                if prev_cluster_name != new_cluster_name:
                    # change all instances of prev names to new name
                    new_assignments[new_cluster_name] = self.assignments.pop(prev_cluster_name)
                    new_cluster_sizes[new_cluster_name] = self.cluster_sizes.pop(prev_cluster_name)
                    new_sorted_clusters.append((new_cluster_name, self.sorted_clusters[i][1]))
                    new_cluster_centroids[new_cluster_name] = self.cluster_centroids.pop(prev_cluster_name)
                    new_labels[np.where(self.labels == prev_cluster_name)] = new_cluster_name
                
            # set naming variable
            self.cluster_name_int = self.sorted_clusters[-1][0]
        # assigning class attributes to new varaibels
        self.assignments = new_assignments
        self.cluster_sizes = new_cluster_sizes
        self.sorted_clusters = new_sorted_clusters
        self.cluster_centroids = new_cluster_centroids
        self.labels = new_labels


    def add_point(self,coord, cluster = None):
        # If cluster is None, create a new cluster then add point
        # If cluster is specified, adds a coordinate to that cluster
        # updates self.labels and self.assignments
        # Input: 
        #       coord: a tuple (r,c) as the point to be added
        #       cluster: The name for the cluster the point is to be added in to. 
        # Output: None

        r,c = coord
        
        # create new cluster if neccesary
        if cluster == None:
            cluster = self.cluster_name_int + 1
            self.cluster_name_int = cluster
            self.assignments[cluster] = []
        
        # update labels and assignments
        self.labels[r,c] = cluster
        self.assignments[cluster].append(coord) 

    def find_clusters(self):
        # Main clustering algorithm. Does a for loop over every cell in self.grid. 
        # Checks if occupied or not. 
        # If occ. then will check neighbors, if no labeled neighbors then it creates a new cluster.
        # If there exists labeled neighbors, then union neccessary clusters and join cluster.

        R,C = np.shape(self.grid)

        for r in range(R):
            for c in range(C):
                # Check if coord is occupied or not
                # if its is occupied, then we have to make some updates
                # if not, then we conintue onto the next point
                is_occupied = self.grid[r,c]

                if is_occupied:
                    # check neighbors above and left of coord (r,c) and filter out coords that are out of bounds
                    neighbors_coords = [(r_,c_) for r_,c_ in [(r-1,c-1),(r-1,c),(r-1,c+1),(r,c-1)] if (r_ >= 0 and r_ < R) and (c_ >= 0 and c_ < C)]
    
                    # check if neighbors are labeled. If exists labels, then add coord to that cluster. Union all the neighbors if multiple clusters are present.
                    # neighbor labels is a list of all cluster labels
                    neighbor_labels = []
                    for r_,c_ in neighbors_coords:
                        neighbor_cluster_label = self.labels[r_,c_]
                        # Check neighbor labels
                        # take action if occupied, continue if not
                        if (neighbor_cluster_label != 0) and (neighbor_cluster_label not in neighbor_labels):
                            neighbor_labels.append(neighbor_cluster_label)
                        else:
                            pass

                    # if no occupied neighbors, then create a new cluster
                    if len(neighbor_labels) == 0:
                        # make new label and assign (r,c) to that label
                        self.add_point((r,c))
                    else:
                        # get any cluster label and assign all the other points to this cluster
                        main_cluster_label = neighbor_labels[0]
                        neighbor_labels = neighbor_labels[1:]
                        # add point r,c to this cluster
                        self.add_point((r,c), cluster=main_cluster_label)
                        # union all other points
                        for other_cluster_label in neighbor_labels:
                            self.union(main_cluster_label,other_cluster_label)
                else:
                    pass

    def find_centroids(self):
        # Uses the assignments dictionary to calculate the centroid of each cluster using the center of mass equations. 
        # Output:
        #       cluster_centroids: a dictionary mapping centroid loc to cluster {cluster: (r_centroid, c_centroid) }

        cluster_centroids = dict()

        for cluster, cluster_points in self.assignments.items():

            centroid_coord = sum(coord[0] for coord in cluster_points)/len(cluster_points), sum(coord[1] for coord in cluster_points)/len(cluster_points)
            cluster_centroids[cluster] = centroid_coord

        self.cluster_centroids = cluster_centroids

        return cluster_centroids

    def calc_cluster_sizes(self):
        # Computes the size of clusters and ranks them by size
        # Output:
        #       self.cluster_sizes: a dictionary mapping clusters to num of points in the cluster
        #       self.sorted_clusters: a list of (clusters,size) pairs in order of size ([more points ...... less points])
        cluster_sizes = dict()
        for cluster in self.assignments:
            size = len(self.assignments[cluster])
            cluster_sizes[cluster] = size
        self.cluster_sizes = cluster_sizes
        self.sorted_clusters = sorted(cluster_sizes.items(), key = lambda x : x[1], reverse = True)

    def filter_by_features_mask(self,mask,density_threshold = .01, draw = False):
        # Filters out clusters that don't have enough features. 
        # Modifies self.labels, self.assigments, self.cluster_sizes, etc
        # Input:
        #       mask: a 2d np array that is 1 where a feature exists and 0 o.w.
        #       density_threshold: if the density (#features in cluster/cluster area) is less than the threshold, remove cluster
        # 
        if draw == True:
            plt.figure()
            plt.subplot(221)
            im_to_show = np.copy(self.labels)
            im_to_show = np.where(mask == 1, self.cluster_name_int + 1, im_to_show)
            plt.imshow(im_to_show)
            plt.subplot(223)
            plt.imshow(np.where(self.labels != 0, im1_ir,0))

        clusters_to_remove = []

        # calc feature density for each cluster
        for cluster in self.assignments:
            cluster_area = len(self.assignments[cluster])
            num_features = np.shape(np.where((self.labels == cluster) & (mask != 0)))[1]
            density = num_features/cluster_area

            # remove cluster if not enough features
            if density < density_threshold:
                clusters_to_remove.append(cluster)

        # Do removal
        for cluster in clusters_to_remove:
            self.labels = np.where(self.labels == cluster,0,self.labels)
            del self.assignments[cluster]
            del self.cluster_sizes[cluster]
            del self.cluster_centroids[cluster]

        self.sorted_clusters = sorted(self.cluster_sizes.items(), key = lambda x : x[1], reverse = True)
        self.rename_clusters()

        if draw == True:
            plt.subplot(222)
            plt.imshow(self.labels)
            plt.subplot(224)
            im_to_show = np.copy(self.labels)
            im_to_show = np.where(self.labels != 0, im1_ir,0)
            plt.imshow(im_to_show)
            plt.show()        

class Test_Cluster_Labeling(unittest.TestCase):

    def test_clustering(self):
        # unoccupied
        test = np.array([[ 0, 0, 0, 0 ],
                         [ 0, 0, 0, 0 ],
                         [ 0, 0, 0, 0 ]])

        corr = np.array([[ 0, 0, 0, 0 ],
                         [ 0, 0, 0, 0 ],
                         [ 0, 0, 0, 0 ]])

        result = Cluster_Labeling(test).labels
        np.testing.assert_array_equal(result,corr)
        
        # two seperate clusters, adding new points
        test = np.array([[ 1, 1, 1, 0 ],
                         [ 0, 0, 0, 0 ],
                         [ 0, 1, 1, 1 ]])

        corr = np.array([[ 1, 1, 1, 0 ],
                         [ 0, 0, 0, 0 ],
                         [ 0, 2, 2, 2 ]])

        result = Cluster_Labeling(test).labels
        np.testing.assert_array_equal(result,corr)

        # different numbers and str [0,1,2]
        test = np.array([[ 1, 2, 1000, 0.0 ],
                         [ 0, 0, 0.00, 0.0 ],
                         [ 0, 2, 1.89, -1  ]])

        corr = np.array([[ 1, 1, 1, 0 ],
                         [ 0, 0, 0, 0 ],
                         [ 0, 2, 2, 2 ]])

        result = Cluster_Labeling(test).labels
        np.testing.assert_array_equal(result,corr)

        # test simple union of 2 clusters 
        test = np.array([[ 1, 1, 0, 1 ],
                         [ 0, 0, 1, 0 ],
                         [ 1, 0, 0, 1 ]])

        corr = np.array([[ 1, 1, 0, 1 ],
                         [ 0, 0, 1, 0 ],
                         [ 2, 0, 0, 1 ]])

        result = Cluster_Labeling(test).labels
        np.testing.assert_array_equal(result,corr)

        # test all occupied
        test = np.array([[ 1, 1, 1, 1 ],
                         [ 1, 1, 1, 1 ],
                         [ 1, 1, 1, 1 ]])

        corr = np.array([[ 1, 1, 1, 1 ],
                         [ 1, 1, 1, 1 ],
                         [ 1, 1, 1, 1 ]])

        result = Cluster_Labeling(test).labels
        np.testing.assert_array_equal(result,corr)

        # empty array
        test = np.array([[]])

        corr = np.array([[]])

        result = Cluster_Labeling(test).labels
        np.testing.assert_array_equal(result,corr)
    def testing_centroid_finding(self):
        test = np.array([[ 1, 1, 0, 1 ],
                         [ 0, 0, 1, 0 ],
                         [ 1, 0, 0, 1 ]])
        corr = {1: (1,2), 2: (2,0)}
        cl = Cluster_Labeling(test)




if __name__ == "__main__":
    pass