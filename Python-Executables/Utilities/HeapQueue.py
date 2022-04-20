from heapq import heapify, heappush, heappop

# Heap Element
class hElem:
    def __init__(self, subset, merit):
        self.subset = subset
        self.merit = merit

    def __eq__(self, other):
        return set(self.subset) == set(other.subset)

    def __str__(self):
        return f"Subset: {self.merit}, Merit: {self.merit}"

    def __lt__(self, other):
        return self.merit > other.merit
    
# Heap Queue
class HeapQueue:
    def  __init__(self):
        self.queue = []
        heapify(self.queue)
        
    def __len__(self):
        return len(self.queue)
    
    def __iter__(self):
        return iter(self.queue)
    
    def __getitem__(self, key):
        # if key is of invalid type or value, the list values will raise the error
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __delitem__(self, key):
        del self.values[key]

    def isEmpty(self):
        return len(self.queue) == 0
    
    def push(self, subset, merit):
        """
        item already in priority queue with smaller priority:
        -> update its priority
        item already in priority queue with higher priority:
        -> do nothing
        if item not in priority queue:
        -> push it
        """
        item = hElem(subset, merit)
        if item in self.queue:
            ind = self.queue.index(item)
            ori_merit = self.queue[ind].merit
            if merit >= ori_merit:
                self.queue[ind] = item
            return
        
        heappush(self.queue, item)
        
    def pop(self):
        """
        return item with highest priority and remove it from queue
        """
        item = heappop(self.queue)
        return (item.subset, item.merit)