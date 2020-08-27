class Config(object):
    def __init__(self):
        self.ENV = 'default'            #当前的环境参数
        self.Introduce = 'Not at the moment'    #对此次实验的描述
        self.VERSION = 1                #当前版本


        #------------------------------------------------GPU配置
        self.GPU = dict(
            use_gpu = True,             #是否使用GPU，True表示使用
            device_id = [0],            #所使用的GPU设备号，type=list
        )


        self.CONFIG = dict(
            dataset_name = 'MovieLen',     #所选择的数据集的名称
            model_name = 'Baseline',       #攻击模型的名称
            criterion_name = 'MSELoss',       #损失函数的名称
            optimizer_name = 'Adam',     #优化器的名称（torch.nn中）
            metrics = ['mse', 'rmse'],        #评价标准的名称（metric文件夹中）
            adjust_lr = True,               #是否自动的变化学习率
            load_model = True,              #是否加载预训练模型（测试、迁移）
        )

        #------------------------------------------------训练参数设置
        self.ARG = dict(
            epoch = 30,         #训练epoch
            batch_size = 256,    #训练集batch_size
        )

        #------------------------------------------------损失函数选择
        
        
        
        #------------------------------------------------网络模型
        self.Baseline = dict(
            embedding_num = 400,
            output_dim = 6,
        )


        #------------------------------------------------优化器
        self.Adam = dict(
            lr = 0.001,                  #学习率
            weight_decay = 5e-4,        #权重衰减
        )


        #------------------------------------------------数据集
        #--------------------------------数据集参数
        self.MovieLen = dict(
            dirname = '/home/baiding/Desktop/Study/Deep/datasets/recommend/MovieLen/ml-1m',            #数据集存放的文件夹
            prop = 0.7,         #训练集所占的比例
        )

        
        #------------------------------------------------学习率变化
        self.LrAdjust = dict(
            lr_step = 10,                   #学习率变化的间隔
            lr_decay = 0.1,                 #学习率变化的幅度
            increase_bottom = 5,            #退火前学习率增加的上界
            increase_amp = 1.1,             #学习率增加的幅度
        )


        #------------------------------------------------模型加载
        self.LoadModel = dict(
            filename = './checkpoint/MovieLen_Baseline_V1/Baseline_Epoch2.pkl',     #加载模型的位置，与上面模型要对应
        )


        #------------------------------------------------checkpoint
        self.Checkpoint = dict(
            checkpoint_dir = './checkpoint/{}_{}_V{}'.format(
                self.CONFIG['dataset_name'], self.CONFIG['model_name'],
                self.VERSION),                          # checkpoint 所在的文件夹
            checkpoint_file_format = self.CONFIG['model_name']+'_Epoch{}.pkl',     #模型文件名称格式，分别表示模型名称、Epoch
            model_best = 'model_best.ptk',            #最好的模型名称，暂时未用到
            log_file = 'log.log',                         #log文件名称
            save_period = 1,                            #模型的存储间隔
        )


    def log_output(self):
        log = {}
        log['ENV'] = self.ENV
        log['Introduce'] = self.Introduce
        log['CONFIG'] = self.CONFIG
        for name,value in self.CONFIG.items():
            if type(value) is str and hasattr(self,value):
                log[value] = getattr(self,value)
            else:
                log[name] = value
        for name,value in self.ARG.items():
            log[name] = value
        return log