from queue import Queue

class DBSCAN:
    
    """Class for performing Density-Based Spatial Clustering of Applications with Noise (DBSCAN)"""

    def __init__(self, costs, max_size = 10, anti_noiser = True):
        self.anti_noiser = anti_noiser
        self.max_size = max_size
        self.max_dist = 2 * max(map(max, costs))
        self.weights = None

        self.solution = None

    # Returns subset of dests with elements x that satisfies
    # costs[source][x] + costs[x][source] <= 2 * radius
    def range_query(self, dests, costs, source, radius):
        result = list()
        for dest in dests:
            if (costs[source][dest] + costs[dest][source]) / 2 <= radius:
                result.append(dest)
        return result
    
    # Standard dbscan clustering dests.
    # Returns list of clusters.
    def dbscan(self, dests, costs, radius = 10, min_size = 1):
        clusters_num = -1

        states = dict()
        # Undifined cluster.
        for d in dests:
            states[d] = -2

        for d in dests:
            neighbours = self.range_query(dests, costs, d, radius)
            if len(neighbours) < min_size:
                states[d] = -1

        for dest in dests:
            if states[dest] != -2:
                continue

            clusters_num += 1
            q = Queue()
            q.put(dest)

            while not q.empty():
                dest2 = q.get()
                states[dest2] = clusters_num
                neighbours = self.range_query(dests, costs, dest2, radius)
                for v in neighbours:
                    if states[v] == -2:
                        q.put(v)

        for dest in dests: 
            if states[dest] == -1:
                min_dist = self.max_dist
                best_neighbour = -1
                for d in dests:
                    if states[d] != -1:
                        if costs[d][dest] < min_dist:
                            best_neighbour = d
                            min_dist = costs[d][dest]
                if best_neighbour == -1:
                    clusters_num += 1
                    states[dest] = clusters_num
                else:
                    states[dest] = states[best_neighbour]

        clusters = list()
        for i in range(clusters_num + 1):
            clusters.append(list())
        for dest in dests:
            cl = states[dest]
            clusters[cl].append(dest)

        self.solution = clusters
        return clusters

    # Recursive dbscan. Returns list of clusters.
    # dests - set that need to be clustered.
    # costs - array with costs between dests.
    # min_radius, max_radius - lower and upper bound for radius parameter in dbscan.
    # clusters_num - expected maximum number of clusters. It is not guaranteed that 
    # function won't return more clusters.
    # max_size - maximum size of a cluster. It is guaranteed that every cluster will
    # have at most max_size elements.
    # max_weight - maximum sum of deliveries' weights of a cluster. It is guaranteed that every cluster will
    # have at most max_weight sum of weights.
    def recursive_dbscan(self, dests, costs, min_radius=5, max_radius=15, clusters_num=50, max_size=10, max_weight=100):
        best_res = [[d] for d in dests]

        min_r = min_radius
        max_r = max_radius
        curr_r = max_r

        # Searching best radius with binary search.
        while min_r + 1 < max_r:
            curr_r = (min_r + max_r) / 2

            clusters = self.dbscan(dests, costs, curr_r, 1)

            if len(clusters) < clusters_num:
                max_r = curr_r
            else:
                min_r = curr_r
                if len(clusters) < len(best_res):
                    best_res = clusters
        print("Radius:", curr_r)

        # Recursive dbscan on clusters with too many elements.
        for cluster in best_res:
            weight = 0
            if self.weights is not None:
                for dest in cluster:
                    weight += self.weights[dest]
            if len(cluster) > max_size or weight > max_weight:
                best_res.remove(cluster)
                best_res += self.recursive_dbscan(cluster, costs, 0., self.max_dist, 2, max_size, max_weight)

        # Removing singleton clusters while they are and there is more than clusters_num clusters.
        if self.anti_noiser:
            while len(best_res) > clusters_num:
                singleton = [0]
                for cluster in best_res:
                    if len(cluster) == 1:
                        singleton = cluster
                        break

                if singleton == [0]:
                    break

                best_res.remove(singleton)

                one = singleton[0]
                best_cluster = []
                best_dist = self.max_dist

                for cluster in best_res:
                    if len(cluster) == max_size or cluster == singleton:
                        continue

                    weight = 0
                    min_dist = self.max_dist

                    for dest in cluster:
                        if self.weights is not None:
                            weight += self.weights[dest]
                        min_dist = min(min_dist, costs[dest][one])
                    if self.weights is None or weight + self.weights[one] <= max_weight:
                        if best_dist > min_dist:
                            best_dist = min_dist
                            best_cluster = cluster

                if best_cluster == []:
                    best_res.append(singleton)
                    break
                best_res.remove(best_cluster)
                best_res.append(best_cluster + singleton)

        self.solution = best_res
        return best_res