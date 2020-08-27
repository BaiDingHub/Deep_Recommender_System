import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import *


def movielen_data_make(source_file, prop, title_length):
    """[制作movielen数据集所需要的所有数据，并保存在文件夹中]
    将rating分为训练集和测试集，存储到data_split.p文件中，加载方式如下：
        train_data, test_data = pickle.load(open(os.path.join(source_file,'data_split.p'), 'rb'))
        train_data占ratings的prop比例，为(num,3)矩阵, 包括UserID，MovieID,ratings
    
    将user特征及相应参数存储到users_feature.p文件中，加载方式如下：
        users, users_orig = pickle.load(open(os.path.join(source_file, 'users_feature.p'), 'rb'))
        users : (user_num, 4)DataFrame，包括UserID、Gender、Age、Occupation，全是整数
        usres_orig : 未处理前的users‘s DataFrame
        
    将movie特征及相应参数存储到movies_feature.p文件中，加载方式如下：
        movies, movies_orig, title_length, title_voc_num, genres_num = pickle.load( 
                open(os.path.join(source_file, 'movies_feature.p'), 'rb'))
        movies : (movie_num, 3)DataFrame，包括MovieID、Title、Genres，MovieID为整数，
            Title为title_length长度的整数List，Genres为genres_num长度的整数List
        movies_orig : 未处理前的movies‘s DataFrame
        title_length : Title特征List的长度
        title_voc_num : Title特征中单词种类的数目   目前长度5217
        genres_num : Genres特征中单词种类的数目   目前长度17
        
        
    Args:
        source_file (str): [MovieLen数据集所在的文件夹]
        prop ([float]): 训练集所占的比例
        title_length ([int]): Title特征List的长度

    Returns:
        ratings [DataFrame]: [ratings(全数字)]
    """
    USER_DATA_FILE = os.path.join(source_file, 'users.dat')
    MOVIE_DATA_FILE = os.path.join(source_file, 'movies.dat')
    RATING_DATA_FILE = os.path.join(source_file, 'ratings.dat')
    
    users, users_orig = user_data_processing(USER_DATA_FILE)
    movies, movies_orig, title2int, genres2int = movie_data_processing(MOVIE_DATA_FILE, title_length)
    ratings = rating_data_processing(RATING_DATA_FILE)
    data = ratings.values
    
    # 划分训练集和测试集(ratings)
    np.random.seed(2020)
    n_examples = data.shape[0]
    n_train = int(n_examples * prop)
    train_idx = np.random.choice(range(0, n_examples),size = n_train,replace = False)
    test_idx = list(set(range(0,n_examples)) - set(train_idx))
    train_data = data[train_idx]
    test_data = data[test_idx]
    pickle.dump([train_data, test_data], open(os.path.join(source_file, 'data_split.p'), 'wb'))
    
    # 对用户特征进行初步分析，并存储到文件中
    pickle.dump((users, users_orig), open(os.path.join(source_file, 'users_feature.p'), 'wb'))
    
    # 对电影特征进行初步分析，并存储到文件中
    title_voc_num = len(title2int)
    genres_num = len(genres2int)
    pickle.dump((movies, movies_orig, title_length, title_voc_num, genres_num), 
                open(os.path.join(source_file, 'movies_feature.p'), 'wb'))
    
    
    
    
    


def user_data_processing(filepath):
    """[对user.dat中的数据进行处理]
    - UserID: 不做处理
    - Gender字段：需要将‘F’和‘M’转换成0和1
    - Age字段：要转成7个连续数字0~6
    - Occupation：不做处理
    - 舍弃： Zip-code
    
    Args:
        filepath (str): [user.dat文件的位置]

    Returns:
        users [DataFrame]: [users(全数字)]
        users_orig [arr]: [users.dat与users对应的原数据]
    """
    user_title = ['UserID','Gender','Age','Occupation','Zip-code']
    users = pd.read_csv(filepath, 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names= user_title)
    users_orig = users
    
    # 舍弃Zip-code字段
    users = users.filter(regex='UserID|Gender|Age|Occupation')
    
    # 处理Gender字段
    users_orig = users.values
    gender_to_int = {'F':0, 'M':1}
    users['Gender'] = users['Gender'].map(gender_to_int)
    # 处理Occupation字段
    age2int = {val:ii for ii, val in enumerate(sorted(set(users['Age'])))}
    users['Age'] = users['Age'].map(age2int)
    
    return users, users_orig

def movie_data_processing(filepath, title_length = 16):
    """[对movie.dat中的数据进行处理]
    - MovieID字段: 不做处理
    - Title字段: 首先去除掉title中的year。然后将title映射成数字列表。（int映射粒度为单词而不是整个title）
    - Genres字段: 进行int映射，因为有些电影是多个Genres的组合,需要再将每个电影的Genres字段转成数字列表.
    - Genres和Title字段需要将长度统一，这样在神经网络中方便处理。
    - 注意，我们需要将Title、Genres的内容转换成等长度的list，使用Title所对应的数字进行填充，不够的部分用‘< PAD >’对应的数字填充。
    
    Args:
        filepath (str): [movie.dat文件的位置]
        title_length (int, optional): [title's list的场地]. Defaults to 16.

    Returns:
        movies [DataFrame]: [movies(全数字)]
        movies_orig [arr]: [movies.dat与movies对应的原数据]
        title2int [dict]: [每个title单词所对应的标号字典，可用于预测]
        genres2int [dict]: [每个genres单词所对应的标号字典，可用于预测]
    """
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_csv(filepath, 
                        sep='::', 
                        engine='python', 
                        encoding='latin-1',
                        names= movies_title)
    movies_orig = movies
    
    # 划分title字段
    ## 使用正则表达式去掉年份
    pattern = re.compile('^(.*)\((\d+)\)$')
    title_re_year = {val:pattern.match(val).group(1) for val in set(movies['Title'])}
    movies['Title'] = movies['Title'].map(title_re_year)
    ## 将title的单词分开
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)
    title_set.add('PADDING')
    ##　将每个单词对应成数字
    title2int = {val: idx for idx, val in enumerate(title_set)}
    title_map = {val: [title2int[row] for row in val.split()] 
                    for val in set(movies['Title'])}
    ## 把title字段转换成list,不够长度的部分用PADDING所对应的数字填充
    for key in title_map.keys():
        padding_length = title_length - len(title_map[key])
        padding = [title2int['PADDING']] * padding_length
        title_map[key].extend(padding)
    movies['Title'] = movies['Title'].map(title_map)
    
    # 划分genres字段
    ## 将genres的单词分开
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)
    genres_set.add('PADDING')
    ## 将每个单词对应成数字
    genres2int = {val: idx for idx, val in enumerate(genres_set)}
    genres_map = {val:[genres2int[row] for row in val.split('|')]
                 for val in set(movies['Genres'])}
    ## 把movie字段转换成list,不够长度的部分用PADDING所对应的数字填充
    for key in genres_map:
        padding_length = len(genres_set) - len(genres_map[key])
        padding = [genres2int['PADDING']] * padding_length
        genres_map[key].extend(padding)
    movies['Genres'] = movies['Genres'].map(genres_map)
    
    return movies, movies_orig, title2int, genres2int

def rating_data_processing(filepath):
    """[rating.dat中的数据进行处理]
    - 只需要将timstaps社区，保留其他属性即可

    Args:
        filepath (str): [ratings.dat文件的位置]

    Returns:
        ratings [DataFrame]: [ratings(全数字)]
    """
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_csv(filepath, 
                        sep='::', 
                        engine='python', 
                        encoding='latin-1',
                        names= ratings_title)
    ratings = ratings.filter(regex='UserID|MovieID|ratings')
    
    return ratings



class MovieLenTrainSet(Dataset):
    """[加载MovieLen数据集，此处使用的是ML1M数据集，包含rating.data, users.data, movies.dat]

    Args:
        dirname ([str]): [MovieLen数据集所在的文件夹]
        prop ([float]): 训练集所占的比例
        title_length ([int]): Title特征List的长度
    """
    def __init__(self, dirname, prop=0.7, title_length = 16):

        if not os.path.exists(os.path.join(dirname, 'data_split.p')):
            movielen_data_make(dirname, 0.7, 16)
        # 打开训练文件data_split.p
        train_data, test_data = pickle.load(open(os.path.join(dirname,'data_split.p'), 'rb'))
        # 打开用户特征文件users_feature.p
        users, users_orig = pickle.load(open(os.path.join(dirname, 'users_feature.p'), 'rb'))
        # 打开电影特征文件movies_feature.p
        movies, movies_orig, title_length, title_voc_num, genres_num = pickle.load( 
                open(os.path.join(dirname, 'movies_feature.p'), 'rb'))
        
        self.datas = train_data
        self.users = users.values
        self.movies = movies.values

        # 分析用户各项特征的最大可能取值数目，记录用户的UserID、Gender、Age、Occupation可能取值的数目
        self.users_max_info = []
        for i in range(1,users.shape[1]):
            self.users_max_info.append(len(set(self.users[:,i])))
        # 分析电影各项特征的最大可能取值数目，记录电影的MovieID、Title、Genres可能取值的数目
        # ID_num = max(train_data)
        self.movies_max_info = [title_voc_num, genres_num]


    def __getitem__(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Returns:
            data [dict]: [包含用户特征user_feature, 电影特征movie_feature]
            rating [int]: [对应用户和电影下的评分]
        """
        rating = self.datas[idx][2]
        user_id = self.datas[idx][0]
        movie_id = self.datas[idx][1]

        user_gender = self.users[self.users[:,0] == user_id][0][1]
        user_age = self.users[self.users[:,0] == user_id][0][2]
        user_occupation = self.users[self.users[:,0] == user_id][0][3]

        movie_title = self.movies[self.movies[:,0] == movie_id][0][1]
        movie_genres = self.movies[self.movies[:,0] == movie_id][0][2]

        user_feature = {
            'gender' : torch.LongTensor([user_gender]),
            'age' : torch.LongTensor([user_age]),
            'occupation' : torch.LongTensor([user_occupation]),
        }
        movie_feature = {
            'title' : torch.LongTensor(movie_title),
            'genres' : torch.LongTensor(movie_genres),
        }

        # data = {'user_feature':user_feature, 'movie_feature': movie_feature}
        return (user_feature,movie_feature, rating)
    
    def __len__(self):
        return len(self.datas)
    
    def get_model_info(self):
        """[summary]

        Returns:
            users_max_info [list]: [用户的每一项特征可能取值的数目]
            movies_max_info [list]: [电影的每一项特征可能取值的数目]
        """
        info = {
            'users_max_info' : self.users_max_info,
            'movies_max_info' : self.movies_max_info,
        }
        return info





class MovieLenTestSet(Dataset):
    """[加载MovieLen数据集，此处使用的是ML1M数据集，包含rating.data, users.data, movies.dat]

    Args:
        dirname ([str]): [MovieLen数据集所在的文件夹]
        prop ([float]): 训练集所占的比例
        title_length ([int]): Title特征List的长度
    """
    def __init__(self, dirname, prop=0.7, title_length = 16):

        if not os.path.exists(os.path.join(dirname, 'data_split.p')):
            movielen_data_make(dirname, prop, title_length)
        # 打开训练文件data_split.p
        train_data, test_data = pickle.load(open(os.path.join(dirname,'data_split.p'), 'rb'))
        # 打开用户特征文件users_feature.p
        users, users_orig = pickle.load(open(os.path.join(dirname, 'users_feature.p'), 'rb'))
        # 打开电影特征文件movies_feature.p
        movies, movies_orig, title_length, title_voc_num, genres_num = pickle.load( 
                open(os.path.join(dirname, 'movies_feature.p'), 'rb'))
        
        self.datas = test_data
        self.users = users.values
        self.movies = movies.values

        # 分析用户各项特征的最大可能取值数目，记录用户的Gender、Age、Occupation可能取值的数目
        self.users_max_info = []
        for i in range(1,users.shape[1]):
            self.users_max_info.append(len(set(self.users[:,i])))
        # 分析电影各项特征的最大可能取值数目，记录电影的Title、Genres可能取值的数目
        self.movies_max_info = [title_voc_num, genres_num]


    def __getitem__(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Returns:
            data [dict]: [包含用户特征user_feature, 电影特征movie_feature]
            rating [int]: [对应用户和电影下的评分]
        """
        rating = self.datas[idx][2]
        user_id = self.datas[idx][0]
        movie_id = self.datas[idx][1]

        user_gender = self.users[self.users[:,0] == user_id][0][1]
        user_age = self.users[self.users[:,0] == user_id][0][2]
        user_occupation = self.users[self.users[:,0] == user_id][0][3]

        movie_title = self.movies[self.movies[:,0] == movie_id][0][1]
        movie_genres = self.movies[self.movies[:,0] == movie_id][0][2]

        user_feature = {
            'gender' : torch.LongTensor([user_gender]),
            'age' : torch.LongTensor([user_age]),
            'occupation' : torch.LongTensor([user_occupation]),
        }
        movie_feature = {
            'title' : torch.LongTensor(movie_title),
            'genres' : torch.LongTensor(movie_genres),
        }

        # data = {'user_feature':user_feature, 'movie_feature': movie_feature}
        return (user_feature,movie_feature, rating)
    
    def __len__(self):
        return len(self.datas)
    
    def get_model_info(self):
        """[summary]

        Returns:
            users_max_info [list]: [用户的每一项特征可能取值的数目]
            movies_max_info [list]: [电影的每一项特征可能取值的数目]
        """
        info = {
            'users_max_info' : self.users_max_info,
            'movies_max_info' : self.movies_max_info,
        }
        return info


if __name__ == "__main__":
    train_set = MovieLenTrainSet('/home/baiding/Desktop/Study/Deep/datasets/recommend/MovieLen/ml-1m')
    test_set = MovieLenTestSet('/home/baiding/Desktop/Study/Deep/datasets/recommend/MovieLen/ml-1m')
    print(len(train_set))
    print(train_set.__getitem__(0))


