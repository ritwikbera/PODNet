class config():
    def __init__(self,dataset):
        if dataset=='minigrid':
            self.categorical_dim=3  #go to key/door/goal
            self.MAX_LENGTH = 100
            self.SEGMENT_SIZE = 100
            self.batch_size = 4

        elif dataset=='circleworld':
            self.categorical_dim=2
            self.MAX_LENGTH=500
            self.SEGMENT_SIZE=100
            self.batch_size=4

        elif dataset=='robotarium':
            self.categorical_dim=4
            self.MAX_LENGTH=10240
            self.SEGMENT_SIZE=512
            self.batch_size=2

if __name__=='__main__':
    conf = config('circleworld')
    print(conf.SEGMENT_SIZE)
