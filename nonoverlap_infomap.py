from infomap import Infomap 

class InfoMap:
    def __init__(self, G):
        self._G = G

    def excuate(self):
        infomap = Infomap() 
        infomap.multiLevel = True  # 或者 False，取决于你的需求  
        infomap.teleportationProbability = 0.50  
        infomap.physical = True  # 设置为物理模式  
        infomap.coarseTuneLevel = 2  # 设置粗调谐级别  
        infomap.seedToRandomNumberGenerator = 42  # 设置随机数生成器的种子 
        for edge in self._G.edges():  
            infomap.add_link(*edge)  
        infomap.run()  
        infomap_communities = infomap.get_modules()
        node_key = {f'{value}': index for index, value in enumerate(self._G.nodes())}
        output_data = {}
        for i in infomap_communities.keys():
            output_data[str(i)] = []
            output_data[str(i)].append(infomap_communities[i])
            
        # 进行排序
        max_cluster = max(output_data.values())[0] 
        for i in range(len(self._G.nodes())):
            # output_data[str(i)] = [output_data[str(i)]]
            if str(i) not in output_data.keys():
                output_data[str(i)] = [max_cluster]
                max_cluster = max_cluster + 1
        output_data =  dict(sorted(output_data.items(), key=lambda x: int(x[0])))
        return output_data

