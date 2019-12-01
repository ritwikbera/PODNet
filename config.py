class config():
    def __init__(self,dataset):
        if dataset=='minigrid':
            self.state_dim=3
            self.action_dim=7
            self.latent_dim=1
            self.categorical_dim=4
            self.MAX_LENGTH = 100
            self.SEGMENT_SIZE = 100
            self.batch_size = 4

        elif dataset=='circleworld':
            self.state_dim=2
            self.action_dim=2
            self.latent_dim=1
            self.categorical_dim=2
            self.MAX_LENGTH=1024
            self.SEGMENT_SIZE=512
            self.batch_size=1

        elif dataset=='robotarium':
            self.state_dim=2
            self.action_dim=2
            self.latent_dim=1
            self.categorical_dim=4
            self.MAX_LENGTH=10240
            self.SEGMENT_SIZE=512
            self.batch_size=2

if __name__=='__main__':
    conf = config('circleworld')
    print(conf.SEGMENT_SIZE)
