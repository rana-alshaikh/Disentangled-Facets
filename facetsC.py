import os
from sympy import Matrix


from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

import math
from operator import itemgetter
import scipy
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import hdbscan
from sklearn import linear_model
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
from numpy.linalg import svd
import time
import itertools
import tensorflow as tf
import re
#from scipy.stats import t
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist,squareform
import warnings

warnings.filterwarnings('ignore')
print('finish importing')


# In[ ]:



class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass
class Facets():
    def __init__(self,ent,fea,orderd_features,mdsMAIN,mds,mdsTest10_P,trainIndices,testIndices,train_labels,test_labels,
                 orderd_features_directions,orderd_features_directions_GLOVE,GloveVectors,
                 orderd_features_Kappa,orderd_features_positiveMovies,orderd_features_predictions,
                 numberOfFacets,dimOfFacets,facets_Path,remainder_Path):
        
       
        self.__fea = fea#features extracted from the corpus
        self.__ent = ent#documents names
        self.__orderd_features = orderd_features # features ordered based on Kappa score
        self.__orderd_features_directions = orderd_features_directions #directions for the self.__orderd_features
        self.__orderd_features_directions_GLOVE = orderd_features_directions_GLOVE #directions for the self.__orderd_features concatenated with the pre_trained Glove Vectors
        self.__orderd_features_Kappa = orderd_features_Kappa# Kappa score for each feature
        self.__orderd_features_positiveMovies = orderd_features_positiveMovies# movies that classified as positive by the linear classifier
        self.__orderd_features_predictions = orderd_features_predictions
        self.__GloVeVectorsForAll=GloveVectors# pre_trained word embedding vectors
        self.__trainIndices = trainIndices
        self.__testIndices = testIndices
        self.__mdsMAIN = mdsMAIN
        self.__mds = mds
        self.__mdsTest10_P = mdsTest10_P
        self.__train_labels = train_labels
        self.__test_labels = test_labels
        self.__numberOfFacets = numberOfFacets
        self.__dimOfFacets = dimOfFacets
        self.__fulldim=100
        self.__facetsPath =facets_Path
        self.__remainderPath =remainder_Path


    def decompose_Space(self):

        self.__directions = [list(i) for i in (self.__orderd_features_directions)[:500]]
        self.__PositiveInstance = self.__orderd_features_positiveMovies
        dirc = self.__orderd_features_directions[:500]
        dircGlove = self.__orderd_features_directions_GLOVE[:500]
        terms = self.__orderd_features[:500]
        self.__candidate_facet=self.__incAgg(dirc,dircGlove,terms)
        print(self.__candidate_facet)



    

        orthogonal_basis_all=[]
        self.__10dim_facets=[]
        space=self.__mds
        spaceTest = self.__mdsTest10_P
        find_proj = self.__mdsMAIN #For the first facet we use the entire space
        self._fulldim=100 # for the first facet the fulldim will be 100d for the second facet it will be 90d as we will remove the 10d of the first facet and so on
        self.__features_for_each_facet=[]
        self.__features_for_each_facet1=[]
        self.__featuresDirection_for_each_facet=[]
        self.__featuresKappa_for_each_facet=[]
        self.__featuresDirection_for_each_facet_glove=[]
        all_features=self.__orderd_features

        all_features=[x for x in self.__orderd_features   if 1.0  in (self.__train_labels[:,self.__fea.index(x)])]


        self.__facets=[]
        self.__facets.append(self.__candidate_facet)
        
        for i in range(self.__numberOfFacets):
            print('Fact: ',i)
            print(self.__candidate_facet)

            orthogonal_basis,correct_prediction_kapp5, test_y_kapp5=self.__finding_facet_space(space,spaceTest)
            dim_10,dim_100=projection(orthogonal_basis, find_proj)
            #print('dim10 and dim100',np.array(dim_10).shape,np.array(dim_100).shape)
            self.__10dim_facets.append(dim_10)
            if (i)!=9:
                #finding the orthogonal basis of the remainder space which is the orthogonal complement of the orthogonal basis of the 10dim(using the func. checkit)  
                nullspace_orthogonaBasis=checkit((orthogonal_basis))
                #project the instances to the facet's 10 d space
                nullspace_=projection_sub(np.transpose(nullspace_orthogonaBasis), find_proj)
                #print(np.array(nullspace_).shape)
                find_proj=np.array(nullspace_)#To repeat
                #print(rank(nullspace_),rank(self.__mds))
                self.__fulldim=rank(nullspace_)
                space=np.array(nullspace_)[self.__trainIndices]
                spaceTest=np.array(nullspace_)[self.__testIndices]

            #learning the directions of all the terms in the facet's space
            temp_f,temp_direction,temp_directionGlove,temp_kappa,PositiveInstance,predictions=finding_directions(self.__fea,np.array(dim_10)[self.__trainIndices],i,all_features,self.__GloVeVectorsForAll,self.__train_labels,self.__facetsPath)
            print('Top 100 features in the facet')
            print(temp_f[:100])
            self.__features_for_each_facet.append(temp_f)
            self.__features_for_each_facet1.append(temp_f[:20])#Facet extension Y 
            self.__featuresDirection_for_each_facet.append(temp_direction)
            self.__featuresDirection_for_each_facet_glove.append(temp_directionGlove)
            self.__featuresKappa_for_each_facet.append(temp_kappa)

            all_features=[x for x in all_features if x  not in self.__candidate_facet and x not in temp_f[:30]]

            flatten_facets = list(itertools.chain.from_iterable(self.__facets))
            flatten1_facetsExten = list(itertools.chain.from_iterable(self.__features_for_each_facet1))

            remainder_features=[x for x in self.__orderd_features if x  not in flatten_facets and x not in flatten1_facetsExten]

            if (i)!=9:
                #print('I am inside the if!=9')
                temp_f_rest,temp_direction_rest,temp_direction_rest_Glove,temp_kappa_rest,PositiveInstance,predictions=finding_directions(self.__fea,space,i,remainder_features,self.__GloVeVectorsForAll,self.__train_labels,self.__remainderPath)
                temp_f_rest_Glove=[]
                temp_f_rest_adjOnly=[]

                try:
                    self.__directions=[list(i) for i in temp_direction_rest[:500]]
                    self.__candidate_facet=self.__incAgg(temp_direction_rest[:500],
                                                                 temp_direction_rest_Glove[:500],
                                                                 temp_f_rest[:500])


                except:
                    self.__directions=[list(i) for i in temp_direction_rest[:1900]]
                    self.__candidate_facet=self.__incAgg(temp_direction_rest[:1900],
                                                                 temp_direction_rest_Glove[:1900],
                                                                 temp_f_rest[:1900])

                self.__facets.append(self.__candidate_facet)
                #print(self.__candidate_facet)


        

   
    
        
    def __dfun(self,d1, d2):
        u=self.__PositiveInstance[list(self.__directions).index(list(d1))]
        v=self.__PositiveInstance[list(self.__directions).index(list(d2))]
        overlap=np.min([len(intersection(u, v))/len(u),len(intersection(u,v))/len(v)])
        return overlap
    def __incAgg(self,directions,directions_Glove,terms):

        """
        This finction will first calculate the overlap score. Second it will create the disssimilarity matrix.
        Finally it will apply the AgglomerativeClustering.
        """
        ranked=[]
        ranked1=[]
        cosine_similarity_all= cosine_similarity(np.array(directions_Glove)[:,abs(len(directions_Glove[0])-50):])

        dm =squareform(pdist(np.array(directions), self.__dfun))



        sim_matrix=np.zeros((len(dm),len(dm)),dtype=np.float)

        cosine_similarity_all=np.array(cosine_similarity_all,dtype=np.float32)
        dm=np.array(dm,dtype=np.float32)
        for ii in range(len(dm)):
            if ii in [100,200,300,400,500,700,800,900]:#just to check the progress of the run
                print(ii)
            for jj in range(len(dm)):

                minimum=(1-cosine_similarity_all[ii][jj])
                if dm[ii][jj]<0.7:#The Overlap threshold 

                    sim_matrix[ii][jj]=minimum
                else:
                    sim_matrix[ii][jj]=1
        Nclusters=100
        cluster = AgglomerativeClustering(n_clusters=Nclusters, affinity='precomputed', linkage='average')  
        cluster.fit_predict(sim_matrix)
        labels = cluster.labels_


        n_clusters_ =Nclusters


        hdbclusters=[]
        for i in range (0,n_clusters_):

            tempf=[]

            for j in range(0,len(terms)):

                if labels[j]==i:

                    tempf.append(terms[j])
            if len(tempf)>10 and len(tempf)<=20:
                hdbclusters.append(tempf)
        best_cluster=self.__findingclusterDirections(hdbclusters,terms,directions)


        return best_cluster

    def __findingclusterDirections(self,clusters,terms,terms_Direc):
        """
        This func. will return the most inclusive cluster


        """
        clusterFeatuer=[]
        cluster1stLevel_index=[]
        cluster1stLevel_originalFeatuerDirection=[]
        cluster1sLevel_Pos=[]
        cluster1sLevel_AvgKappa=[]
        for clus in range(len (clusters)):
            #print('cluster#: ',clus)
            temp=[]
            tempf=[]
            tempDirections=[]
            temp_pos=[]
            tempTotalKappAvg=0.0
            for j in range(0,len(terms)):
                if terms[j] in clusters[clus] :

                    tempf.append(terms[j])
                    tempDirections.append(terms_Direc[j])
                    temp.append(self.__fea.index(terms[j]))
                    temp_pos=Union(list(self.__orderd_features_positiveMovies[self.__orderd_features.index(terms[j])])
                                   ,temp_pos)
                    kap=self.__orderd_features_Kappa[self.__orderd_features.index(terms[j])]
                    tempTotalKappAvg=tempTotalKappAvg+float(kap)
            if len(tempf)>0:
                cluster1stLevel_index.append(temp)
                clusterFeatuer.append(tempf)
                cluster1stLevel_originalFeatuerDirection.append(tempDirections)
                cluster1sLevel_Pos.append(len(temp_pos))
                cluster1sLevel_AvgKappa.append(tempTotalKappAvg/len(tempf))#just in case we want to extract the cluster with the highst avg kappa score

        return clusters[np.argmax(np.array(cluster1sLevel_Pos))]

    
    def __finding_facet_space(self,space,spaceTest):
        """
        This function will return the orthogonal basis of the facet

        Parameters
        ----------
        space : ndarray
            for the first facet it will be [num of training instances X 100d]
            for the second facet it will be [num of training instances X 90d] and so on
        spaceTest : ndarray
            for the first facet it will be [num of testing instances X 100d]
            for the second facet it will be [num of testing instances X 90d] and so on


        Return value
        ------------
        q.T: list
            the orthogonal basis of the facet.

        correct_prediction_kapp:
        the kappa score to evaluate how this facet seperate the space

        """

        train_y=[]
        test_y=[]
        train_X = space
        test_X = spaceTest

        for i in self.__candidate_facet:
            train_y.append(self.__train_labels[:,self.__fea.index(i)])
            test_y.append(self.__test_labels[:,self.__fea.index(i)])
        train_y = np.reshape(train_y,[len(self.__candidate_facet),len(train_X),1])
        test_y = np.reshape(test_y,[len(self.__candidate_facet),len(test_X),1])
        #print(np.array(train_y).shape)

        learning_rate = 0.1
        training_epochs = 50
        tf.reset_default_graph()

        def if_true():#Train
            return tf.constant(train_X,dtype=tf.float32)

        def if_false():#Test
            return  tf.constant(test_X,dtype=tf.float32)

        
        def  lrForTerm (vectors,b1,target):


            """
            This function will calculate the logistic regression for each term in the candidate facet and then send the loss for 
            the main function to optimize it with the loss functions of the other terms in one step.

            """
            tf.set_random_seed(1234)

            x = tf.cond(pred, if_true, if_false)#if pred==True return the training data else return testing data 
            vector1=tf.cast(tf.reshape(vectors,(-1,1)),dtype=tf.float32)
            modd=tf.matmul(x,vector1)
            #print(modd.shape)
            mod=(tf.matmul(x,vector1)+b1)
            prediction = tf.nn.sigmoid(mod)
            loss= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mod, labels=target))

            return loss,prediction,target
        def accuracy_fn(mod,target):
            prediction = tf.round(tf.sigmoid(mod))
            correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
            accuracy = tf.reduce_mean(correct)

            return accuracy
        num_features=self.__fulldim


        X = tf.placeholder(tf.float32, [None, num_features], name="X")
        Y = tf.placeholder(tf.float32, [len(self.__candidate_facet),None, 1], name="Y")
        pred = tf.placeholder(tf.bool) 
        

        FacetVectors=tf.Variable(tf.truncated_normal([self.__dimOfFacets, num_features],
                                        stddev=1./math.sqrt(self.__dimOfFacets)))
        lamda=tf.Variable(tf.random_normal(shape=[len(self.__candidate_facet), self.__dimOfFacets]),dtype=tf.float32)
        W=tf.matmul(lamda,FacetVectors)
        W=tf.reshape(W,[len(self.__candidate_facet),num_features, 1])

        b = tf.get_variable("b", [len(self.__candidate_facet),1], initializer = tf.zeros_initializer(),dtype=tf.float32)

        #usin the tf.map to fin the loss of the lojistc reg loss for each term
        final_result,prediction,nothing = tf.map_fn(lambda r:lrForTerm(r[0],r[1],r[2],),(W ,b,Y),
                                     dtype=(tf.float32, tf.float32,tf.int32))
       


        # Calculate the cost
        #final_result is matrix that contain the loss of each term 
        cost =tf.reduce_mean(final_result)# tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = Y))

        # Use Adam as optimization method
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

        cost_history = np.empty(shape=[1],dtype=float)

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(training_epochs):
                _, c ,FacetVectors1= sess.run([optimizer, cost,FacetVectors], feed_dict={X: train_X, Y: train_y, pred: True})

                cost_history = np.append(cost_history, c)


            # Calculate the correct predictions


            correct_prediction = tf.to_float(tf.greater(prediction, 0.5))


            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(Y, correct_prediction)))


            correct_prediction_kapp= correct_prediction.eval({X: test_X, Y: test_y, pred: False})
            #finding the orthogonal basis(to project the instances later) using QR factorization
	    #note you cannot project unless you have an orthogonal basis of the space, the FacetVectors are not orthogonal	
            q, r = np.linalg.qr(np.transpose(FacetVectors1))
            A=scipy.linalg.orth(FacetVectors1)

            return q.T,correct_prediction_kapp,test_y




    @property
    def facetsSpaces(self):
        if self.__10dim_facets is None:
            raise NotTrainedError("Need to train model first.")
        return self.__10dim_facets

    @property
    def featuresForFacets(self):
        if self.__features_for_each_facet1 is None:
            raise NotTrainedError("Need to train model first.")
        return self.__features_for_each_facet1
    @property
    def featuresDircForFacets(self):
        if self.__featuresDirection_for_each_facet is None:
            raise NotTrainedError("Need to train model first.")
        return self.__featuresDirection_for_each_facet

    @property
    def featuresGloveDircForFacets(self):
        if self.__featuresDirection_for_each_facet_glove is None:
            raise NotTrainedError("Need to train model first.")
        return self.__featuresDirection_for_each_facet_glove
    
    
    
def projection(orthogonal_basis, vectors):
    proj_10d=[]
    proj_100d=[]
    for y in vectors:
        #the prog_100d to help us to find the null space because we need a 100 dim vectors with rank 10 to find the remainder 90 dim
        proj_100d.append(np.array([np.dot(y,orthogonal_basis[i] )*orthogonal_basis[i]
                              for i in (range(len(orthogonal_basis))) ]).sum(axis=0))
        #only the dot because we want it to be 10 dim
        proj_10d.append(np.array([np.dot(y,orthogonal_basis[i] )
                                  for i in (range(len(orthogonal_basis))) ]))
    return(proj_10d,proj_100d)



def projection_sub(orthogonal_basis, vectors):
    '''we use this finction to project the instances to the remainder space 
    to find the top feature of the remainder space to find the next facet'''
    proj_10d=[]
    proj_100d=[]
    for y in vectors:
        proj_10d.append(np.array([np.dot(y,orthogonal_basis[i] )
                                  for i in (range(len(orthogonal_basis))) ]))
    return(proj_10d)

def intersection(lst1, lst2): 
  
    # Use of hybrid method 
    temp = set(lst2) 
    lst3 = [value for value in lst1 if value in temp] 
    return lst3 
                
def Union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list    




def rank(A, atol=0.09, rtol=0):
    """Estimate the rank (i.e. the dimension of the nullspace) of a matrix.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length n will be treated
        as a 2-D with shape (1, n)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    r : int
        The estimated rank of the matrix.

    See also
    --------
    numpy.linalg.matrix_rank
        matrix_rank is basically the same as this function, but it does not
        provide the option of the absolute tolerance.
    """

    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    return rank


def nullspace(A, atol=0.09, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def checkit(a):
    #print ("a:")
    #print (a)
    r = rank(a)
    print ("rank is", r)
    ns = nullspace(a)
    #print ("nullspace:")
    #print (ns)
    #print(rank(ns),ns.shape)
    if ns.size > 0:
        res = np.abs(np.dot(a, ns)).max()
        #print( "max residual is", res)
    return ns


def perf_measure(y_actual, y_hat): # Get the true positives etc
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == 1 and y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] == 0:
            FP += 1
        if y_actual[i] == 0 and y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] == 1:
            FN += 1

    return TP, FP, TN, FN



def runLR(vectors, classes):
    # Default is dual formulation, which is unusual. Balanced balances the classes
    clf = linear_model.LogisticRegression(class_weight="balanced", dual=False)
    y=range(len((vectors)))
    vectorsTrain, vectorsTest, y_train, y_test = train_test_split( vectors, y, random_state=42)

    classesTrain=classes[y_train]
    
    clf.fit(vectorsTrain, classesTrain)  # All of the vectors and classes. No need for training data.
    direction = clf.coef_.tolist()[0]  # Get the directionbais=clf.intercept_.tolist()[0]
    bais=clf.intercept_.tolist()

    predictedTest = clf.predict(vectorsTest)
    predictedTest = predictedTest.tolist()
    kappa_score_test = cohen_kappa_score(classes[y_test], predictedTest)
    
    probabilities=clf.predict_proba(vectors)
    predicted = clf.predict(vectors)
    predicted = predicted.tolist()  # Convert to list so we can calculate the scores
    f1 = f1_score(classes, predicted)
    kappa_score = cohen_kappa_score(classes, predicted)
    acc = accuracy_score(classes, predicted)
    
    TP, FP, TN, FN = perf_measure(classes, predicted)  # Get the True positive, etc
    return kappa_score_test,kappa_score, f1, direction, acc, TP, FP, TN, FN,predicted,probabilities




def finding_directions(fea,space,facet,featuresTerms,GloveVectors,train_labels,Path):
    '''
    This function will return the orthogonal basis of the facet
        
    Parameters
    ----------
    space: The vector space where we want to learn the directions
    facet: The facet's number or -1 if to learn the directions in the entire space
    featuresTerms: The terms that we want to learn the directions for
    GloveVectors: the pre_trained Glove vectors for all the features(not only theone in the cluster)
    train_labels: the labels to train the classifier
    Path: To detrmain where to save the files 


    Return value
    ----------
    #1-orderd_features: list containsthe terms orderd based on kappa score)
    #2-orderd_features_directions: ndarray contains the direction for each term
    #3-orderd_features_directions_GLOVE:
    #each line is a concatenation between the term's direction and the term's pre-trained word embedding vectors
    #4-orderd_features_Kappa(Kappa scores for each direction)
    #5-orderd_features_positiveMovies (movies that classified by the linear classifier as positive)
    ##############################################################################
    ##############################################################################

    '''
    
    base_folder_second = Path 
 
        
    directionsLGR=[]
    scores=[]
    predictions=[]
    kappa_Score_all=[]
    kappa_Score_all_test=[]
    probabilities_features=[]
    for i in  range(0,len(featuresTerms[:2000])):
        clusterpmiW=[]
        if i%200==0:
        
            start_time = time.time()
        if i %200==0:
            print("--- %s seconds ---" % (time.time() - start_time))

        if i%1000==0:
            print('finished: ',i)

        kappa_scoretest,kappa_score1, f1, direction, acc, TP, FP, TN, FN, predicted,probabilities = runLR(space,train_labels[:,fea.index(featuresTerms[i])])


        directionsLGR.append(direction)
        predictions.append(predicted)
        probabilities_features.append(probabilities)
        kappa_Score_all.append(kappa_score1)
        kappa_Score_all_test.append(kappa_scoretest)
        scores.append([f1,acc])

    #print('STEP4_directions')        


    #sort the directions
    ##############################################################################
    ##############################################################################    
    positiveMovies=[]
    for i in range(0,len(featuresTerms[:2000])):
        temp=[]
        for j in range(0,len(predictions[i])):
            if predictions[i][j]==1:
                temp.append(j)
        positiveMovies.append(temp)

    filtered_directions=[]
    accuracy=[]
    for i in range(0,len(featuresTerms[:2000])):
        accuracy.append(kappa_Score_all_test[i])
    orderd_accuracy_index=np.argsort(accuracy)[::-1]
    #print('finish sorting')



    #writing the directions after sorting
    ##############################################################################
    ##############################################################################
    orderd_features1=[]
    orderd_features_directions1=[]
    orderd_features_kappa=[]
    orderd_features_directions_GLOVE1=[]
    orderd_features_positiveMovies=[]
    orderd_features_predictions=[]
    for j in orderd_accuracy_index:
        orderd_features1.append(featuresTerms[int(j)])
        orderd_features_kappa.append(kappa_Score_all_test[int(j)])
        orderd_features_directions1.append(directionsLGR[int(j)])
        orderd_features_directions_GLOVE1.append(np.concatenate((directionsLGR[int(j)],np.array(GloveVectors[fea.index(featuresTerms[int(j)])]))))
        orderd_features_positiveMovies.append(positiveMovies[j])
        orderd_features_predictions.append(predictions[j])
    if facet==-1:#learning the directions in the entire space
        write2dArray(orderd_features_positiveMovies, base_folder_second+'orderd_featuresPositiveReclean')    
        write1dArray(orderd_features1, base_folder_second+'orderd_featuresReclean')
        write2dArray(orderd_features_directions1, base_folder_second+'orderd_features_directionsReclean')
        write2dArray(orderd_features_directions_GLOVE1, base_folder_second+'orderd_features_directions_GLOVEReclean')
        write1dArray(orderd_features_kappa, base_folder_second+'orderd_features_kappaReclean')
    else:
        write1dArray(orderd_features1, base_folder_second+'orderd_features_for_facet_'+str(facet))
        write2dArray(orderd_features_directions1, base_folder_second+'orderd_features_directions_for_facet_'+str(facet))
        write2dArray(orderd_features_directions_GLOVE1, base_folder_second+'orderd_features_directions_GLOVE_for_facet_'+str(facet))
        write1dArray(orderd_features_kappa, base_folder_second+'orderd_features_kappa_for_facet_'+str(facet))

    

    return orderd_features1,orderd_features_directions1,orderd_features_directions_GLOVE1,orderd_features_kappa,orderd_features_positiveMovies,orderd_features_predictions



# In[ ]:


def import2dArray(file_name, file_type="f", return_sparse=False):
    if file_name[-4:] == ".npz":
        print("Loading sparse array")
        array = sp.load_npz(file_name)
        if return_sparse is False:
            array = array.toarray()
    elif file_name[-4:] == ".npy":
        print("Loading numpy array")
        array = np.load(file_name)#
    else:
        with open(file_name, "r") as infile:
            if file_type == "i":
                array = [list(map(int, line.strip().split())) for line in infile]
            elif file_type == "f":
                array = [list(map(float, line.strip().split())) for line in infile]
            elif file_type == "discrete":
                array = [list(line.strip().split()) for line in infile]
                for dv in array:
                    for v in range(len(dv)):
                        dv[v] = int(dv[v][:-1])
            else:
                array = np.asarray([list(line.strip().split()) for line in infile])
        array = np.asarray(array)
    print("successful import", file_name)
    return array


def write1dArray(array, name):
    try:
        file = open(name, "w")
        print("starting array")
        for i in range(len(array)):
            file.write(str(array[i]))
            file.write("\n")
        file.close()
    except FileNotFoundError:
        print("FAILURE")
   

    print("successful write", name)
    
    
def write2dArray(array, name):
    try:
        file = open(name, "w")
        print("starting array")
        for i in range(len(array)):
            for n in range(len(array[i])):
                file.write(str(array[i][n]) + " ")
            file.write("\n")
        file.close()
    except FileNotFoundError:
        print("FAILURE")
    

    print("successful write", name)    

