import random

class Kmeans(object):
    """
    k-means algorithm
    """
    
    def __init__(self, data, clusters, metric, avg):
        
        self.centroids = random.sample(data, clusters)

        err = -1
        while True:
            bins = [set() for k in xrange(clusters)]
            
            for i in data:
                ml = [(metric(c,i), ic) for ic,c in enumerate(self.centroids)]
                ml_min,_ =  min(ml)
                c = random.choice( [k for m,k in ml if m == ml_min] )
                bins[c].add(i)

            for bi,b in enumerate(bins):
                self.centroids[bi] = avg(b)
                
            olderr = err
            err = sum( [sum( [metric(d,self.centroids[c]) for d in b] )
                        for c,b in enumerate(bins)] )
            
            if err - olderr < 0.0005:
                break

    def result(self):
        return self.centroids


def choose_initial(data, k, distfunc=None):
    """
    Choose randomly k different centroids.

    :data The elements being clustered
    :k The number of clusters
    :distfunc Ignored
    """
    return random.sample(data, k)

def choose_initial_pp(data, k, distfunc):
    """
    Choose randomly k different centroids using the kmeans++ heuristic
    by David Arthur and Sergei Vassilvitskii (see the article "k-means++:
    The Advantages of Careful Seeding".

    :data The elements being clustered
    :k The number of clusters
    :distfunc Function to calculate the distance between two elements.
    """

    # Calculate squared distance
    distance2 = lambda(c, x): calcdist.setdefault((c, x),
            calcdist.setdefault((x, c), distfunc(x, c)))

    # The first centroid is a random one
    centroids = [random.choice(data)]
    # Table to store the calculated values, avoiding duplicated calculations
    calcdists = dict()

    while len(centroids) < k:
        mindists = [min((distance2(c, x), x) for c in centroids)
                for x in data if x is not c]
        # Divide because we add it twice: first for (c, x) and then for (x, c)
        totaldist = float(sum(mindists) / 2)
        for d, x in mindists:
            if x not in centroids and random.random() < d / totaldist:
                centroids.append(x)
                break

    return centroids

def calc_centroid(cluster):
    """
    Calculate the centroid from the given cluster.

    Note: This function assumes that the methods __add__ and __div__ are
    correctly set on the cluster's elements. Otherwise, use your own function.
    """

    return sum(cluster) / len(cluster)